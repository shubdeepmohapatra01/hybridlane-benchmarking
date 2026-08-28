# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Pure-DV (qubit-only) continuous optimization, as a baseline for CV-QAOA.

This is the qubit-only counterpart to `hyqbench_benchmarks.cv_qaoa`: it
minimizes the *same* polynomial ``f(x)`` with the *same* QAOA structure, so any
difference in resources is attributable to the encoding rather than to a
different algorithm.

A qubit machine has no continuous degree of freedom, so ``x`` is discretized
onto a fixed-point grid of ``2**n_qubits`` points across ``domain``::

    x_j = x_min + j * delta,    delta = (x_max - x_min) / 2**n_qubits

and the computational basis state ``|j>`` *is* the position ``x_j``.

Both QAOA unitaries are then **exact**, which is worth stating plainly because
it makes this a strong baseline rather than a strawman:

``exp(-i eta f(x))``
    Diagonal in the computational basis, so it expands into Pauli-``Z`` strings
    that all mutually commute. Trotterizing a set of commuting terms is exact,
    so there is no splitting error here at any degree -- unlike the JCH
    benchmark, where the DV side pays one.

``exp(-i gamma p^2 / 2)``
    Diagonal in the *Fourier* basis, so it is QFT -> diagonal -> QFT-dagger,
    again exact.

Where the cost actually goes
---------------------------
Since ``x`` is affine in the bit variables and ``Z_k^2 = I``, the operator
``x^m`` expands into every product of at most ``m`` distinct ``Z``s. A degree
``d`` polynomial therefore needs

    sum_{k=1}^{d} C(n_qubits, k)  =  O(n_qubits^d)

Pauli terms, against ``O(d)`` native gates on a qumode. That gap -- polynomial
of degree ``d`` in the register width, versus linear in the degree -- is the
comparison this module exists to measure, and it is why the cubic case
separates the two encodings much more sharply than the quadratic.

The expansion is computed by a fast Walsh-Hadamard transform of the diagonal
rather than by symbolic algebra: it is exact, runs in ``O(n 2^n)``, and cannot
drift from the diagonal it is supposed to represent. `SparsePauliOp.from_operator`
would also be exact but costs ``O(4^n)``, which is unusable past a few qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, QFTGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter

DEFAULT_DOMAIN = (-6.0, 6.0)


# ---------------------------------------------------------------------------
# Grid and Pauli expansion
# ---------------------------------------------------------------------------


def position_grid(n_qubits: int, domain=DEFAULT_DOMAIN) -> np.ndarray:
    """The ``2**n_qubits`` grid points, indexed so that ``|j>`` holds ``x[j]``."""
    x_min, x_max = domain
    n_points = 1 << n_qubits
    delta = (x_max - x_min) / n_points
    return x_min + delta * np.arange(n_points)


def momentum_grid(n_qubits: int, domain=DEFAULT_DOMAIN) -> np.ndarray:
    """Conjugate momenta, in the index order the QFT produces.

    The grid is centred: index ``k`` above ``N/2`` represents a *negative*
    momentum. Getting this wrong is silent -- the circuit still runs and still
    conserves probability, it just simulates a different kinetic term -- so it
    is pinned by `test_dv_matches_cv_on_quadratic`.
    """
    x_min, x_max = domain
    n_points = 1 << n_qubits
    length = x_max - x_min
    k = np.arange(n_points)
    k = np.where(k >= n_points // 2, k - n_points, k)
    return 2.0 * np.pi * k / length


def _fwht(vector: np.ndarray) -> np.ndarray:
    """In-place-style fast Walsh-Hadamard transform (unnormalized)."""
    out = np.array(vector, dtype=float, copy=True)
    n = out.size
    step = 1
    while step < n:
        for start in range(0, n, 2 * step):
            a = out[start : start + step].copy()
            b = out[start + step : start + 2 * step].copy()
            out[start : start + step] = a + b
            out[start + step : start + 2 * step] = a - b
        step *= 2
    return out


def diagonal_to_pauli(diagonal: np.ndarray, n_qubits: int, atol=1e-12) -> SparsePauliOp:
    """Exact Pauli-``Z`` expansion of a diagonal operator.

    The Walsh-Hadamard transform *is* this expansion: the coefficient of
    ``prod_{k in S} Z_k`` is the transform coefficient at the index whose set
    bits are ``S``, because ``Z_k`` acts on ``|j>`` as ``(-1)**bit_k(j)``.
    """
    coeffs = _fwht(np.asarray(diagonal, dtype=float)) / diagonal.size
    labels, values = [], []
    for index, value in enumerate(coeffs):
        if abs(value) < atol:
            continue
        # Qiskit label strings are big-endian: position 0 is the highest qubit.
        label = "".join("Z" if (index >> k) & 1 else "I" for k in reversed(range(n_qubits)))
        labels.append(label)
        values.append(value)
    if not labels:
        labels, values = ["I" * n_qubits], [0.0]
    return SparsePauliOp(labels, np.asarray(values, dtype=complex))


def poly_diagonal(coeffs, n_qubits: int, domain=DEFAULT_DOMAIN) -> np.ndarray:
    """``f(x_j)`` on the grid, ascending polynomial coefficients."""
    x = position_grid(n_qubits, domain)
    return sum(float(c) * x**k for k, c in enumerate(coeffs))


def cost_operator(coeffs, n_qubits: int, domain=DEFAULT_DOMAIN) -> SparsePauliOp:
    """``f(x)`` as a sum of commuting Pauli-``Z`` strings."""
    return diagonal_to_pauli(poly_diagonal(coeffs, n_qubits, domain), n_qubits)


def momentum_operator(n_qubits: int, domain=DEFAULT_DOMAIN) -> SparsePauliOp:
    """``p^2 / 2`` in the Fourier basis, as commuting Pauli-``Z`` strings."""
    p = momentum_grid(n_qubits, domain)
    return diagonal_to_pauli(0.5 * p**2, n_qubits)


def term_counts(coeffs, n_qubits: int, domain=DEFAULT_DOMAIN) -> dict:
    """Pauli-term census, the quantity that scales as ``O(n_qubits^degree)``."""
    op = cost_operator(coeffs, n_qubits, domain)
    weights = [sum(1 for c in str(p) if c == "Z") for p in op.paulis]
    return {
        "n_terms": len(op),
        "n_nonidentity": int(sum(1 for w in weights if w > 0)),
        "max_weight": int(max(weights)) if weights else 0,
        "cnot_cost": int(sum(2 * (w - 1) for w in weights if w > 1)),
    }


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------


def dv_qaoa_circuit(
    coeffs,
    etas,
    gammas,
    n_qubits: int,
    domain=DEFAULT_DOMAIN,
    initial="uniform",
    cx_structure: str = "chain",
) -> QuantumCircuit:
    """Build the qubit-only QAOA circuit for ``f(x)``.

    Initial state options:

    ``"uniform"``
        A Hadamard on every qubit. This is the *principled* QAOA choice -- the
        uniform superposition in position is the ground state of the kinetic
        mixer ``p^2/2`` (zero momentum), exactly as the CV side's x-antisqueezed
        vacuum approximates the same thing.

    ``("gaussian", var)``
        A discretized Gaussian of position variance ``var``. Use this to *match*
        the CV side's initial width: the uniform state spans the whole domain
        (variance ``L^2/12``, i.e. 12 on the default domain) while a squeezed
        vacuum at ``s = 1`` has variance ``e^2/2 ~ 3.7``. Comparing the two
        encodings from differently-shaped starts measures the initial condition
        as much as the encoding, so the notebook reports both.

    Its state-preparation cost is *excluded* from the gate counts, on both
    sides -- see `resource_summary`.

    Every `PauliEvolutionGate` here wraps a set of *commuting* terms, so the
    `LieTrotter` synthesis is exact and `reps=1` costs no accuracy.

    `cx_structure` selects the CNOT ladder shape, as in `jch_dv.py`: ``"chain"``
    walks each Pauli string's support next-neighbour to next-neighbour,
    ``"fountain"`` fans every leg into one target. Both realize the *same*
    unitary -- it is a compilation choice, not a second approximation -- and on
    the JCH benchmark the fountain shape plus `optimization_level=3` removed
    roughly half the CNOTs. `resource_summary` reports both.
    """
    qc = QuantumCircuit(n_qubits)
    if initial == "uniform":
        qc.h(range(n_qubits))
    elif isinstance(initial, tuple) and initial[0] == "gaussian":
        qc.prepare_state(gaussian_amplitudes(n_qubits, initial[1], domain), range(n_qubits))
    elif initial != "zero":
        raise ValueError(f"unknown initial state {initial!r}")

    cost = cost_operator(coeffs, n_qubits, domain)
    kinetic = momentum_operator(n_qubits, domain)
    synth = LieTrotter(reps=1, cx_structure=cx_structure)

    for eta, gamma in zip(etas, gammas, strict=True):
        qc.append(PauliEvolutionGate(cost, time=float(eta), synthesis=synth), range(n_qubits))
        qc.append(QFTGate(n_qubits), range(n_qubits))
        qc.append(PauliEvolutionGate(kinetic, time=float(gamma), synthesis=synth), range(n_qubits))
        qc.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    return qc


def gaussian_amplitudes(n_qubits, variance, domain=DEFAULT_DOMAIN) -> np.ndarray:
    """Normalized real amplitudes of a discretized Gaussian centred on the domain.

    Used to give the DV side an initial position spread matching the CV side's
    squeezed vacuum, so the comparison is not confounded by the start.
    """
    x = position_grid(n_qubits, domain)
    centre = 0.5 * (domain[0] + domain[1])
    amps = np.exp(-((x - centre) ** 2) / (4.0 * variance))
    return amps / np.linalg.norm(amps)


def run_state(coeffs, etas, gammas, n_qubits, domain=DEFAULT_DOMAIN, initial="uniform"):
    """Statevector after the circuit, as probabilities over the position grid."""
    qc = dv_qaoa_circuit(coeffs, etas, gammas, n_qubits, domain, initial)
    sv = Statevector.from_int(0, 2**n_qubits).evolve(qc)
    return np.abs(np.asarray(sv.data)) ** 2


# ---------------------------------------------------------------------------
# Fast exact simulator
# ---------------------------------------------------------------------------
#
# `run_state` builds the Qiskit circuit and evolves a `Statevector` through it,
# which is the right thing when the point is that the *circuit* computes this --
# but it costs 28 s per evaluation at 14 qubits, because each cost layer's
# hundreds of Pauli strings are expanded gate by gate and the QFT adds O(n^2)
# more. A BFGS run needs ~1000 evaluations, so anything past 12 qubits is out of
# reach that way, and the precision study needs 17 and beyond.
#
# The circuit's unitary has a closed form. Both `PauliEvolutionGate`s here wrap
# *commuting* terms, so each is exactly a diagonal phase -- in position for the
# cost and in momentum for the mixer -- and the QFT is exactly the DFT that maps
# between them. So one layer is: multiply by a phase, FFT, multiply by a phase,
# inverse FFT. That is O(N log N) instead of O(N * n_terms), and it is the same
# unitary rather than an approximation of it -- `test_fast_matches_circuit`
# holds the two to 1e-12.


def _phase_diagonals(coeffs, n_qubits, domain):
    """``(f(x_j), p_k^2/2)`` -- the two diagonals a layer applies."""
    return poly_diagonal(coeffs, n_qubits, domain), 0.5 * momentum_grid(n_qubits, domain) ** 2


def run_state_fast(coeffs, etas, gammas, n_qubits, domain=DEFAULT_DOMAIN, initial="uniform"):
    """Position-basis probabilities after the circuit, via FFT.

    Same signature and same answer as :func:`run_state`; use this one whenever
    the result rather than the gate sequence is what is wanted.
    """
    n_points = 1 << n_qubits
    if initial == "uniform":
        psi = np.full(n_points, 1.0 / np.sqrt(n_points), dtype=complex)
    elif isinstance(initial, tuple) and initial[0] == "gaussian":
        psi = gaussian_amplitudes(n_qubits, initial[1], domain).astype(complex)
    elif initial == "zero":
        psi = np.zeros(n_points, dtype=complex)
        psi[0] = 1.0
    else:
        raise ValueError(f"unknown initial state {initial!r}")

    fx, p2 = _phase_diagonals(coeffs, n_qubits, domain)
    for eta, gamma in zip(etas, gammas, strict=True):
        psi *= np.exp(-1j * float(eta) * fx)
        # Qiskit's QFT convention is the inverse DFT up to normalization, and
        # `momentum_grid` is already ordered to match numpy's frequency layout.
        # The pairing is fixed by `test_fast_matches_circuit` rather than by
        # argument: a swapped direction still conserves probability and still
        # spreads the packet, so it would not show up as an obvious failure.
        psi = np.fft.fft(psi, norm="ortho")
        psi *= np.exp(-1j * float(gamma) * p2)
        psi = np.fft.ifft(psi, norm="ortho")
    return np.abs(psi) ** 2


def state_moments(probs, coeffs, n_qubits, domain=DEFAULT_DOMAIN) -> dict:
    """``<x>``, ``Var(x)`` and ``<f(x)>`` from grid probabilities."""
    x = position_grid(n_qubits, domain)
    fx = poly_diagonal(coeffs, n_qubits, domain)
    mean_x = float(probs @ x)
    return {
        "mean_x": mean_x,
        "var_x": float(probs @ (x**2)) - mean_x**2,
        "energy": float(probs @ fx),
        # Probability that ran off the end of the grid. The QFT mixer is
        # periodic, so amplitude leaving one edge reappears at the other; if
        # this is not small the domain is too tight and the wraparound is
        # shaping the answer.
        "edge_mass": float(probs[:2].sum() + probs[-2:].sum()),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_dv_qaoa(
    coeffs,
    depth=5,
    n_qubits=8,
    domain=DEFAULT_DOMAIN,
    objective=None,
    maxiter=400,
    x0=None,
    seed=None,
    initial="uniform",
    fast=False,
):
    """Optimize the DV-QAOA angles for one polynomial.

    Deliberately mirrors `hyqbench_benchmarks.cv_qaoa.run_cv_qaoa` argument for
    argument -- same objectives, same defaults, same initialization scheme -- so
    that a difference in the reported result is a difference in the encoding and
    nothing else.

    ``fast=True`` evaluates through :func:`run_state_fast` instead of building
    and evolving the Qiskit circuit. Same unitary, same answer to 1e-12, and the
    only way registers past ~12 qubits are reachable at all. The default stays
    ``False`` so the headline comparison keeps running the actual circuit.
    """
    from scipy.optimize import minimize

    from hyqbench_benchmarks import cv_qaoa as cq

    c = np.asarray(coeffs, dtype=float)
    info = cq.describe(c, domain)
    if objective is None:
        objective = "energy" if info["bounded_below"] else "localization"
    if objective == "energy" and not info["bounded_below"]:
        raise ValueError(
            "objective='energy' is ill-posed for a polynomial that is not "
            "bounded below; use objective='localization'."
        )

    history = []
    runner = run_state_fast if fast else run_state

    def loss(params):
        probs = runner(c, params[:depth], params[depth:], n_qubits, domain, initial)
        moments = state_moments(probs, c, n_qubits, domain)
        value = cq.objective_value(moments, info["x_star"], objective)
        history.append(value)
        return value

    if x0 is None:
        x0 = (
            cq.golden_angle_init(depth)
            if seed is None
            else np.random.default_rng(seed).uniform(-1.0, 1.0, size=2 * depth)
        )

    result = minimize(
        loss, np.asarray(x0, dtype=float), method="BFGS", options={"maxiter": maxiter}
    )

    probs = runner(c, result.x[:depth], result.x[depth:], n_qubits, domain, initial)
    moments = state_moments(probs, c, n_qubits, domain)
    return {
        "params": result.x,
        "fast": fast,
        "depth": depth,
        "n_qubits": n_qubits,
        "objective": objective,
        "problem": info,
        "history": np.asarray(history),
        "n_iterations": int(result.nit),
        **moments,
        "localization": moments["var_x"] + (moments["mean_x"] - info["x_star"]) ** 2,
        "x_error": abs(moments["mean_x"] - info["x_star"]),
        "grid_spacing": (domain[1] - domain[0]) / (1 << n_qubits),
        # The Pauli census is the DV cost that grows with the register; skipped
        # past 18 qubits, where building the operator to count its terms costs
        # more than the optimization it annotates (the count is C(n,1)+C(n,2)
        # for a quadratic, which `test_pauli_expansion_is_exact` pins).
        "terms": term_counts(c, n_qubits, domain) if n_qubits <= 18 else None,
    }


# ---------------------------------------------------------------------------
# Parallel multistart, and the shared evaluation scheme
# ---------------------------------------------------------------------------


def _run_dv_job(kwargs: dict) -> dict:
    """Worker: one DV-QAOA start. Module-level so it survives process spawn."""
    result = run_dv_qaoa(**kwargs)
    result.pop("history", None)
    result["problem"] = {k: v for k, v in result["problem"].items() if k != "coeffs"}
    return result


def sweep_dv_qaoa(jobs: list[dict], max_workers: int | None = None) -> list[dict]:
    """Run `run_dv_qaoa` over a list of kwarg dicts, in parallel."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    if len(jobs) <= 1:
        return [_run_dv_job(job) for job in jobs]
    if max_workers is None:
        max_workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_run_dv_job, jobs))
    except (OSError, RuntimeError):
        return [_run_dv_job(job) for job in jobs]


#: "Converged" for continuous optimization, the analogue of the knapsack
#: study's ``P(optimal) >= 0.9``. An absolute position tolerance is used rather
#: than a relative one because the domain is fixed and both encodings are being
#: asked for the same number.
SUCCESS_TOL = 0.05


def summarize_starts(results: list[dict], tol: float = SUCCESS_TOL) -> dict:
    """Distribution over random starts, in the form the VQE study reports.

    Single-run numbers on these landscapes are not reproducible conclusions --
    that is the lesson `vqe_resource_comparison.ipynb` was rebuilt around, and
    the CV-QAOA depth sweep reproduced the same trap (its ``x_error`` was
    non-monotonic in depth, which is optimizer scatter rather than a resource
    limit). So every headline number here is a distribution: success rate,
    mean +/- standard error, median, and best.
    """
    if not results:
        return {}
    x_err = np.array([r["x_error"] for r in results], dtype=float)
    loc = np.array([r["localization"] for r in results], dtype=float)
    energy = np.array([r["energy"] for r in results], dtype=float)
    n = x_err.size
    hits = int((x_err <= tol).sum())
    return {
        "n_starts": n,
        "n_converged": hits,
        "success_rate": hits / n,
        "x_error_mean": float(x_err.mean()),
        "x_error_se": float(x_err.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "x_error_median": float(np.median(x_err)),
        "x_error_best": float(x_err.min()),
        "localization_median": float(np.median(loc)),
        "localization_best": float(loc.min()),
        "energy_best": float(energy.min()),
        "tol": tol,
    }


def resource_summary(coeffs, depth, n_qubits, domain=DEFAULT_DOMAIN) -> dict:
    """Transpiled gate counts for the DV circuit, both compilations.

    Counts the *whole* ``depth``-layer circuit including the QFT mixers, minus
    state preparation (excluded on both sides so the comparison is of the
    algorithm rather than of how each stack loads its initial state).
    """
    from .resources import count_dv

    etas = np.full(depth, 0.3)
    gammas = np.full(depth, 0.3)
    out = {}
    for structure in ("chain", "fountain"):
        qc = dv_qaoa_circuit(
            coeffs,
            etas,
            gammas,
            n_qubits,
            domain,
            initial="zero",
            cx_structure=structure,
        )
        counts = count_dv(qc)
        out[structure] = {
            "n_qubits": counts["n_qubits"],
            "n_gates": counts["n_gates"],
            "n_two_qubit": counts["n_two_qubit"],
            "depth": counts["depth"],
        }
    out["terms"] = term_counts(coeffs, n_qubits, domain)
    out["saving"] = (
        1 - out["fountain"]["n_two_qubit"] / out["chain"]["n_two_qubit"]
        if out["chain"]["n_two_qubit"]
        else 0.0
    )
    return out
