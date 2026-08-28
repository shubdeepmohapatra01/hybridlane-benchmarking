# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Multivariate continuous optimization: m qumodes vs. m qubit registers.

The single-variable comparison (`hyqbench_benchmarks.cv_qaoa` and
`continuous_dv`) shows a constant-factor gate advantage for the qumode. This
module is where the two costs are expected to *diverge*, because both axes of
the qubit encoding compound with dimension:

- **Width.** ``m`` variables cost the CV side ``m`` qumodes and the DV side
  ``m * n_q`` qubits.
- **Terms.** A degree-``d`` polynomial in ``m`` variables discretized onto
  ``m * n_q`` qubits expands into ``O((m n_q)^d)`` Pauli-``Z`` strings, against
  one native gate per monomial on the CV side.

Polynomial representation
-------------------------
``terms`` maps an exponent tuple to a coefficient, so ``f(x, y) = x^2 + 0.5xy``
is ``{(2, 0): 1.0, (1, 1): 0.5}``. This is sparse in the monomials actually
present, which matters: the CV gate count follows the number of monomials while
the DV term count follows the *degree*, and conflating the two would flatter one
side.

What is native on the CV side
-----------------------------
Every monomial in the ``x_i`` commutes with every other (they are all functions
of position operators, which commute across modes), so the cost unitary
factorizes **exactly** at any degree, in any dimension -- no Trotter error.
Per monomial:

===========  ===============  ====================================================
monomial     unitary          native gates
===========  ===============  ====================================================
``x_i``      displacement     1 (``Displacement``)
``x_i^2``    shear            2 (``Rotation``, ``Squeezing``)
``x_i^3``    cubic phase      1 (``CubicPhase``)
``x_i x_j``  two-mode phase   3 (``Rotation``, ``TwoModeSum``, ``Rotation``)
===========  ===============  ====================================================

The cross term is the one that needed deriving. ``TwoModeSum(lam)`` implements
``|x_a>|x_b> -> |x_a>|x_b + lam x_a>``, i.e. ``exp(-i lam x_a p_b)``; conjugating
mode ``b`` by a quarter phase-space rotation sends ``p_b -> -x_b``, giving
``exp(-i s x_a x_b)``. Verified to overlap 1.0 against a direct matrix
exponential in `test_continuous_nd.py`.

Higher mixed monomials (``x_i^2 x_j`` and beyond) are non-Gaussian *and*
multi-mode; they have no such short native form and are rejected rather than
silently approximated.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DOMAIN = (-6.0, 6.0)


# ---------------------------------------------------------------------------
# Polynomials in several variables
# ---------------------------------------------------------------------------


def n_vars(terms) -> int:
    """Number of variables the term dict is written over."""
    return len(next(iter(terms)))


def degree(terms) -> int:
    """Total degree, i.e. the largest sum of exponents over all monomials."""
    return max(sum(e) for e in terms)


def poly_eval_nd(terms, points) -> np.ndarray:
    """Evaluate ``f`` at ``points`` of shape ``(..., m)``."""
    pts = np.asarray(points, dtype=float)
    total = np.zeros(pts.shape[:-1], dtype=float)
    for exponents, coeff in terms.items():
        contribution = np.full(pts.shape[:-1], float(coeff))
        for axis, power in enumerate(exponents):
            if power:
                contribution = contribution * pts[..., axis] ** power
        total = total + contribution
    return total


def quadratic_with_coupling(centre=(2.0, -1.0), coupling=0.5):
    """``sum_i (x_i - c_i)^2 + coupling * prod_i (x_i - c_i)`` as a term dict.

    The cross term is the point: it is what forces a two-mode gate on the CV
    side and cross-register Pauli strings on the DV side, so a problem without
    one would understate the dimensional cost on both.

    The minimum stays at ``centre`` provided the quadratic form is positive
    definite, i.e. ``|coupling| < 2`` for two variables.
    """
    m = len(centre)
    if m != 2:
        raise ValueError("this helper builds the two-variable case")
    cx, cy = float(centre[0]), float(centre[1])
    k = float(coupling)
    # (x-cx)^2 + (y-cy)^2 + k (x-cx)(y-cy), expanded
    return {
        (2, 0): 1.0,
        (0, 2): 1.0,
        (1, 1): k,
        (1, 0): -2 * cx - k * cy,
        (0, 1): -2 * cy - k * cx,
        (0, 0): cx**2 + cy**2 + k * cx * cy,
    }


def minimum_nd(terms, domain=DEFAULT_DOMAIN, n_restarts=40, seed=0):
    """Locate the minimizer numerically, from many starts inside ``domain``.

    Multi-start rather than a single descent because a general polynomial in
    several variables can have several wells, and silently returning a local one
    as ``x*`` would make every reported error meaningless.
    """
    from scipy.optimize import minimize

    m = n_vars(terms)
    rng = np.random.default_rng(seed)
    best_x, best_f = None, np.inf
    for _ in range(n_restarts):
        x0 = rng.uniform(domain[0], domain[1], size=m)
        res = minimize(
            lambda z: float(poly_eval_nd(terms, z)),
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
        )
        if np.all(res.x >= domain[0]) and np.all(res.x <= domain[1]) and res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)
    return best_x, best_f


def is_bounded_below_nd(terms) -> bool:
    """True only if every top-degree direction grows, checked by sampling.

    A sufficient-condition check would be involved for a general multivariate
    polynomial; this samples the top-degree homogeneous part on a sphere, which
    is exact for detecting a *negative* direction (the failure that matters) and
    conservative otherwise.
    """
    d = degree(terms)
    if d % 2 == 1:
        return False
    m = n_vars(terms)
    top = {e: c for e, c in terms.items() if sum(e) == d}
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(4000, m))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return bool(poly_eval_nd(top, directions).min() > 0)


# ---------------------------------------------------------------------------
# CV side: m qumodes
# ---------------------------------------------------------------------------

#: Monomials with a short native form. Anything else raises rather than being
#: silently approximated -- an approximation here would show up as a resource
#: advantage that the hardware cannot actually deliver.
NATIVE_MONOMIALS = {1, 2, 3}


def cv_gate_count(terms) -> dict:
    """Native CV gates per cost layer, by monomial type."""
    counts = {"displacement": 0, "shear": 0, "cubic": 0, "cross": 0}
    for exponents in terms:
        total = sum(exponents)
        support = [i for i, e in enumerate(exponents) if e]
        if total == 0:
            continue  # global phase
        if len(support) == 1:
            if total == 1:
                counts["displacement"] += 1
            elif total == 2:
                counts["shear"] += 1
            elif total == 3:
                counts["cubic"] += 1
            else:
                raise ValueError(f"monomial {exponents} has no native form")
        elif len(support) == 2 and total == 2:
            counts["cross"] += 1
        else:
            raise ValueError(
                f"mixed monomial {exponents} is non-Gaussian and multi-mode; "
                "it has no short native decomposition"
            )
    gates = counts["displacement"] + 2 * counts["shear"] + counts["cubic"] + 3 * counts["cross"]
    return {
        **counts,
        "cost_gates_per_layer": gates,
        "mixer_gates_per_layer": 2 * n_vars(terms),
        "gates_per_layer": gates + 2 * n_vars(terms),
    }


def cv_simulator(terms, cutoff, squeeze=1.0):
    """Split-operator simulator on ``m`` qumodes, exact within the truncation.

    Same construction as the single-mode version: every cost monomial is a
    function of the ``x_i`` alone and every mixer term a function of the ``p_i``
    alone, so each layer is a basis change per mode plus a diagonal phase. The
    state is held as an ``(N,) * m`` tensor and the per-mode transforms are
    applied with `tensordot`, which keeps the cost at ``O(m N^(m+1))`` rather
    than building the ``N^m`` square matrix.
    """
    from scipy.linalg import expm

    m = n_vars(terms)
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    ad = a.conj().T
    x = (a + ad) / np.sqrt(2)
    p = (a - ad) / (1j * np.sqrt(2))
    lam_x, vec_x = np.linalg.eigh(x)
    lam_p, vec_p = np.linalg.eigh(p)

    mesh = np.meshgrid(*([lam_x] * m), indexing="ij")
    f_grid = poly_eval_nd(terms, np.stack(mesh, axis=-1))
    kinetic = sum(np.meshgrid(*([0.5 * lam_p**2] * m), indexing="ij"))

    vacuum = np.zeros(cutoff, dtype=complex)
    vacuum[0] = 1.0
    single = expm(0.5 * (-squeeze * (a @ a) + squeeze * (ad @ ad))) @ vacuum
    psi0 = single
    for _ in range(m - 1):
        psi0 = np.tensordot(psi0, single, axes=0)
    psi0 = psi0.reshape((cutoff,) * m)

    def to_basis(psi, mat):
        for axis in range(m):
            psi = np.moveaxis(np.tensordot(mat, psi, axes=([1], [axis])), 0, axis)
        return psi

    def simulate(etas, gammas):
        psi = psi0.copy()
        for eta, gamma in zip(etas, gammas, strict=True):
            psi = to_basis(psi, vec_x.conj().T)
            psi = np.exp(-1j * float(eta) * f_grid) * psi
            psi = to_basis(psi, vec_x)
            psi = to_basis(psi, vec_p.conj().T)
            psi = np.exp(-1j * float(gamma) * kinetic) * psi
            psi = to_basis(psi, vec_p)
        return psi

    return simulate, (lam_x, vec_x, f_grid)


def cv_moments(psi, terms, cutoff):
    """``<x_i>``, ``Var(x_i)`` and ``<f>`` for an ``(N,)*m`` CV state."""
    m = n_vars(terms)
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    x = (a + a.conj().T) / np.sqrt(2)
    lam_x, vec_x = np.linalg.eigh(x)

    amp = psi
    for axis in range(m):
        amp = np.moveaxis(np.tensordot(vec_x.conj().T, amp, axes=([1], [axis])), 0, axis)
    probs = np.abs(amp) ** 2
    probs = probs / probs.sum()

    mesh = np.meshgrid(*([lam_x] * m), indexing="ij")
    means = np.array([float((probs * g).sum()) for g in mesh])
    second = np.array([float((probs * g**2).sum()) for g in mesh])
    return {
        "mean": means,
        "var": second - means**2,
        "energy": float((probs * poly_eval_nd(terms, np.stack(mesh, axis=-1))).sum()),
        "probs": probs,
        "grid": lam_x,
    }


# ---------------------------------------------------------------------------
# DV side: m registers of n_q qubits
# ---------------------------------------------------------------------------


def dv_grids(n_qubits, domain=DEFAULT_DOMAIN):
    """Position and momentum grids for one register; every variable shares them."""
    from .continuous_dv import momentum_grid, position_grid

    return position_grid(n_qubits, domain), momentum_grid(n_qubits, domain)


def dv_cost_operator(terms, n_qubits, domain=DEFAULT_DOMAIN):
    """``f(x_1..x_m)`` as commuting Pauli-``Z`` strings over ``m * n_q`` qubits.

    Register ``i`` occupies qubits ``[i*n_q, (i+1)*n_q)``. The Walsh-Hadamard
    expansion is taken over the *joint* diagonal, so cross terms appear as Pauli
    strings spanning two registers -- which is exactly the cost that grows with
    dimension.
    """
    from .continuous_dv import diagonal_to_pauli

    m = n_vars(terms)
    x = dv_grids(n_qubits, domain)[0]
    mesh = np.meshgrid(*([x] * m), indexing="ij")
    diagonal = poly_eval_nd(terms, np.stack(mesh, axis=-1)).ravel()
    return diagonal_to_pauli(diagonal, m * n_qubits)


def dv_term_counts(terms, n_qubits, domain=DEFAULT_DOMAIN) -> dict:
    """Pauli-term census over the joint register."""
    op = dv_cost_operator(terms, n_qubits, domain)
    weights = [sum(1 for c in str(p) if c == "Z") for p in op.paulis]
    return {
        "n_qubits": n_vars(terms) * n_qubits,
        "n_nonidentity": int(sum(1 for w in weights if w > 0)),
        "max_weight": int(max(weights)) if weights else 0,
        "cnot_cost": int(sum(2 * (w - 1) for w in weights if w > 1)),
    }


def dv_resource_summary(terms, depth, n_qubits, domain=DEFAULT_DOMAIN) -> dict:
    """Transpiled gate counts for the multivariate DV circuit."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate, QFTGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter

    from .continuous_dv import diagonal_to_pauli
    from .resources import count_dv

    m = n_vars(terms)
    total_qubits = m * n_qubits
    cost = dv_cost_operator(terms, n_qubits, domain)

    # Kinetic term: sum_i p_i^2 / 2, each acting on its own register.
    p = dv_grids(n_qubits, domain)[1]
    single = diagonal_to_pauli(0.5 * p**2, n_qubits)
    kinetic = SparsePauliOp.sum(
        [
            SparsePauliOp(
                [
                    "I" * (total_qubits - (i + 1) * n_qubits) + str(lbl) + "I" * (i * n_qubits)
                    for lbl in single.paulis
                ],
                single.coeffs,
            )
            for i in range(m)
        ]
    ).simplify(1e-12)

    qc = QuantumCircuit(total_qubits)
    synth = LieTrotter(reps=1)
    for _ in range(depth):
        qc.append(PauliEvolutionGate(cost, time=0.3, synthesis=synth), range(total_qubits))
        for i in range(m):
            qc.append(QFTGate(n_qubits), range(i * n_qubits, (i + 1) * n_qubits))
        qc.append(PauliEvolutionGate(kinetic, time=0.3, synthesis=synth), range(total_qubits))
        for i in range(m):
            qc.append(QFTGate(n_qubits).inverse(), range(i * n_qubits, (i + 1) * n_qubits))

    counts = count_dv(qc)
    return {
        "n_qubits": counts["n_qubits"],
        "n_gates": counts["n_gates"],
        "n_two_qubit": counts["n_two_qubit"],
        "depth": counts["depth"],
        "terms": dv_term_counts(terms, n_qubits, domain),
    }


def dv_simulator(terms, n_qubits, domain=DEFAULT_DOMAIN, init_var=3.695):
    """Fast exact DV simulator: split-operator with an FFT mixer.

    Mathematically identical to the Qiskit circuit in `dv_resource_summary` --
    the cost is diagonal in position, the mixer diagonal in momentum, and the
    QFT between them *is* the DFT -- but it evolves the amplitude vector
    directly instead of applying thousands of gates through `Statevector`. That
    keeps the optimizer loop tractable at ``m * n_q`` qubits, where the circuit
    path would dominate the runtime.

    Both sides of this comparison therefore use a split-operator simulator, so
    neither is handicapped by its simulation method.
    `test_dv_fast_matches_circuit` pins the two together.
    """
    m = n_vars(terms)
    x, p = dv_grids(n_qubits, domain)
    n_points = x.size

    mesh = np.meshgrid(*([x] * m), indexing="ij")
    f_grid = poly_eval_nd(terms, np.stack(mesh, axis=-1))
    kinetic = sum(np.meshgrid(*([0.5 * p**2] * m), indexing="ij"))

    centre = 0.5 * (domain[0] + domain[1])
    amp1 = np.exp(-((x - centre) ** 2) / (4.0 * init_var))
    amp1 = amp1 / np.linalg.norm(amp1)
    psi0 = amp1
    for _ in range(m - 1):
        psi0 = np.tensordot(psi0, amp1, axes=0)
    psi0 = psi0.reshape((n_points,) * m).astype(complex)

    axes = tuple(range(m))

    def simulate(etas, gammas):
        psi = psi0.copy()
        for eta, gamma in zip(etas, gammas, strict=True):
            psi = np.exp(-1j * float(eta) * f_grid) * psi
            psi = np.fft.fftn(psi, axes=axes)
            psi = np.exp(-1j * float(gamma) * kinetic) * psi
            psi = np.fft.ifftn(psi, axes=axes)
        return psi

    return simulate, (x, f_grid)


def moments_nd(probs, grid, terms):
    """``<x_i>``, ``Var(x_i)`` and ``<f>`` from a joint probability grid."""
    m = n_vars(terms)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    mesh = np.meshgrid(*([grid] * m), indexing="ij")
    means = np.array([float((probs * g).sum()) for g in mesh])
    second = np.array([float((probs * g**2).sum()) for g in mesh])
    return {
        "mean": means,
        "var": second - means**2,
        "energy": float((probs * poly_eval_nd(terms, np.stack(mesh, axis=-1))).sum()),
    }


def run_nd(
    terms, side, depth, size, seeds, domain=DEFAULT_DOMAIN, maxiter=250, squeeze=1.0, init_var=3.695
):
    """Multistart on either side. ``side`` is ``"cv"`` or ``"dv"``.

    ``size`` is the Fock cutoff for CV and the qubits-per-variable for DV. Both
    minimize the same localization objective against the same ``x*``, so the
    only difference is the encoding.
    """
    from scipy.optimize import minimize

    x_star, _ = minimum_nd(terms, domain)
    m = n_vars(terms)
    if side == "cv":
        simulate, (grid, vec_x, _) = cv_simulator(terms, size, squeeze)

        def measure(params):
            # `cv_simulator` returns the state in the **Fock** basis, so it must
            # be rotated into the x eigenbasis before |amp|^2 can be read as a
            # position distribution. Skipping this pairs Fock amplitudes with
            # position eigenvalues and silently produces nonsense -- it cost a
            # debugging round, and is why the DV branch (whose simulator already
            # works on the position grid) looked fine while this one did not.
            amp = simulate(params[:depth], params[depth:])
            for axis in range(m):
                amp = np.moveaxis(
                    np.tensordot(vec_x.conj().T, amp, axes=([1], [axis])), 0, axis
                )
            return moments_nd(np.abs(amp) ** 2, grid, terms)

    elif side == "dv":
        simulate, (grid, _) = dv_simulator(terms, size, domain, init_var)

        def measure(params):
            psi = simulate(params[:depth], params[depth:])
            return moments_nd(np.abs(psi) ** 2, grid, terms)

    else:
        raise ValueError(f"unknown side {side!r}")

    def loss(params):
        mom = measure(params)
        return float(np.sum(mom["var"]) + np.sum((mom["mean"] - x_star) ** 2))

    out = []
    for seed in seeds:
        x0 = np.random.default_rng(seed).uniform(-1.0, 1.0, size=2 * depth)
        res = minimize(loss, x0, method="BFGS", options={"maxiter": maxiter})
        mom = measure(res.x)
        out.append(
            {
                "seed": int(seed),
                "params": res.x,
                "mean": mom["mean"],
                "var": mom["var"],
                "energy": mom["energy"],
                "x_error": float(np.linalg.norm(mom["mean"] - x_star)),
                "localization": float(np.sum(mom["var"]) + np.sum((mom["mean"] - x_star) ** 2)),
            }
        )
    return out, x_star
