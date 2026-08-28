# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Pure-DV (qubit-only) baselines for the binary knapsack QUBO.

This is the baseline that `sandia.ecd_vqe_sandia`'s ECD-VQE is compared
against. The QUBO is not re-derived here: `qubo_matrix` reads the problem
constants and `knapsack_cost` straight out of `sandia.ecd_vqe_sandia`, so both
sides provably minimize the *same* cost function. `verify_against_sandia`
asserts that agreement over all `2**n` assignments.

Variable ordering (matching `sandia.ecd_vqe_sandia.decode_fock`)::

    z = [x_0, ..., x_{k-1},  x_{n_items-1},  y_0, ..., y_{k-1}]
         \_ m0's k item bits _/  \_ qubit _/  \_ m1's slack bits _/

For `knapsack4b` that is 4 item variables + 3 slack variables = **7 qubits**,
exactly the width of the CV-DV encoding's 7 binary variables.

Two DV ansaetze:

- `hardware_efficient_ansatz` — the standard RY/RZ + CZ-ladder ansatz, the
  qubit-native counterpart of the ECD layer stack.
- `qaoa_ansatz` — QUBO-to-Ising QAOA, the baseline published alongside
  ECD-VQE (arXiv:2501.11735).

Initialization is deterministic (golden-angle schedule), mirroring
`sandia.ecd_vqe_sandia_jax.golden_angle_init`: no random restarts, and never
all-zeros, which is a stationary point for these ansaetze too.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_GOLDEN_ANGLE = 2.399963229728653  # pi * (3 - sqrt(5)), same constant as sandia

#: Largest register for which `verify_against_sandia` builds the dense Ising
#: operator as a cross-check. `SparsePauliOp.to_matrix()` costs 16 * 4**n bytes:
#: 268 MB at 12 variables, 4.3 GB at 14, 68.7 GB at 16. Twelve keeps the check
#: free everywhere it runs; `cost_diagonal` is checked against `knapsack_cost`
#: at every size regardless.
DENSE_VERIFY_MAX_VARS = 12


def _sandia(problem: str = "knapsack4b"):
    """Import `sandia.ecd_vqe_sandia` and switch it to `problem`."""
    from sandia import ecd_vqe_sandia as s

    s.set_problem(problem)
    return s


# ---------------------------------------------------------------------------
# QUBO
# ---------------------------------------------------------------------------


def problem_spec(problem: str = "knapsack4b") -> dict:
    """Problem constants, read from `sandia.ecd_vqe_sandia`."""
    s = _sandia(problem)
    return dict(  # noqa: C408
        problem=s.PROBLEM,
        n_items=s.N_ITEMS,
        values=list(s.VALUES),
        weights=list(s.WEIGHTS),
        max_weight=s.MAX_WEIGHT,
        l_val=s.L_VAL,
        n_bits_m0=s.N_BITS_M0,
        n_bits_m1=s.N_BITS_M1,
        h_opt=s.H_OPT,
        n_depth=s.N_DEPTH,
        # n_bits_m0 item bits on m0 + 1 on the qubit + n_bits_m1 slack bits.
        n_vars=s.N_VARS,
        primary_levels=s.PRIMARY_LEVELS,
        aux_levels=s.AUX_LEVELS,
        max_fock=(s.MAX_FOCK_M0, s.MAX_FOCK_M1),
    )


def split_vars(z, spec: dict) -> tuple[list, list]:
    """Split a flat assignment `z` into knapsack `(x, y)` variables."""
    n_items = spec["n_items"]
    return list(z[:n_items]), list(z[n_items:])


def cost_of(z, problem: str = "knapsack4b") -> float:
    """Classical QUBO cost of an assignment, via `sandia`'s own function."""
    s = _sandia(problem)
    x, y = split_vars(z, problem_spec(problem))
    return float(s.knapsack_cost(x, y))


def qubo_matrix(problem: str = "knapsack4b") -> tuple[np.ndarray, float]:
    """Expand the knapsack cost into `z^T Q z + offset` (Q upper-triangular).

    Uses `z_i^2 = z_i` for binary variables, so linear terms live on `Q`'s
    diagonal. Verified against `sandia.ecd_vqe_sandia.knapsack_cost` by
    `verify_against_sandia`.
    """
    spec = problem_spec(problem)
    n_items, k = spec["n_items"], spec["n_bits_m1"]
    n = spec["n_vars"]
    values, weights = spec["values"], spec["weights"]
    lam, w_max = spec["l_val"], spec["max_weight"]

    # coefficient of z_i inside the residual (W_max - weight - slack)
    resid = np.zeros(n)
    resid[:n_items] = [-w for w in weights]
    resid[n_items:] = [-(2**b) for b in range(k)]

    quad = np.zeros((n, n))
    offset = lam * w_max**2
    for i in range(n):
        quad[i, i] += -values[i] if i < n_items else 0.0
        quad[i, i] += lam * (resid[i] ** 2 + 2 * w_max * resid[i])
        for j in range(i + 1, n):
            quad[i, j] += 2 * lam * resid[i] * resid[j]
    return quad, offset


def qubo_to_ising(qubo: np.ndarray, offset: float) -> tuple[SparsePauliOp, float]:
    """Map `z^T Q z + offset` to an Ising `SparsePauliOp` with `z = (I - Z)/2`.

    Returns `(operator, constant)`; the operator already carries the constant
    as an identity term, and `constant` is returned separately for reference.
    """
    n = qubo.shape[0]
    terms: dict[tuple[int, ...], float] = {}

    def add(qubits, coeff):
        key = tuple(sorted(qubits))
        terms[key] = terms.get(key, 0.0) + coeff

    add((), offset)
    for i in range(n):
        add((), qubo[i, i] / 2)
        add((i,), -qubo[i, i] / 2)
        for j in range(i + 1, n):
            q = qubo[i, j] / 4
            add((), q)
            add((i,), -q)
            add((j,), -q)
            add((i, j), q)

    labels, coeffs = [], []
    for qubits, coeff in terms.items():
        if abs(coeff) < 1e-12:
            continue
        label = ["I"] * n
        for q in qubits:
            label[q] = "Z"
        labels.append("".join(reversed(label)))
        coeffs.append(coeff)
    const = terms.get((), 0.0)
    return SparsePauliOp(labels, np.array(coeffs)), const


def brute_force(problem: str = "knapsack4b") -> dict:
    """Enumerate all `2**n` assignments; return costs, optimum, and spectrum."""
    spec = problem_spec(problem)
    n = spec["n_vars"]
    assignments = np.array(list(itertools.product([0, 1], repeat=n)))
    costs = np.array([cost_of(z, problem) for z in assignments])
    best = costs.min()
    optimal = np.flatnonzero(np.isclose(costs, best))
    return {
        "assignments": assignments,
        "costs": costs,
        "best_cost": float(best),
        "worst_cost": float(costs.max()),
        "optimal_indices": optimal,
        "optimal_assignments": assignments[optimal],
        "degeneracy": int(optimal.size),
    }


def cost_diagonal(problem: str = "knapsack4b") -> np.ndarray:
    """Diagonal of the Ising cost operator, in statevector index order.

    The obvious spelling, ``np.diag(ising.to_matrix())``, allocates a dense
    ``2**n x 2**n`` complex operator to read ``2**n`` real numbers off it. That
    is 268 MB at 12 variables and **68.7 GB at 16**, which is what OOM-killed
    the first 16-variable DV sweep (MaxRSS 63.7 GB against a 64 GB request)
    after it had already spent 14 hours on the CV-DV half of the job. More
    memory is not the fix: the same expression needs 17.6 TB at 20 variables.

    The diagonal is just the classical cost of each assignment, which
    :func:`brute_force` already computes from ``sandia``'s ``knapsack_cost``,
    so this scatters those costs into little-endian state indices instead --
    ``2**n`` floats, milliseconds, and no operator at all.
    ``verify_against_sandia`` checks this against the dense construction at
    every size where the dense one is affordable.
    """
    bf = brute_force(problem)
    n = problem_spec(problem)["n_vars"]
    # Statevector index i carries qubit q as bit q (little-endian); the
    # assignment vectors are in the same q-order, so this is a plain dot.
    idx = bf["assignments"] @ (1 << np.arange(n))
    diag = np.empty(2**n, dtype=float)
    diag[idx] = bf["costs"]
    return diag


def verify_against_sandia(problem: str = "knapsack4b") -> None:
    """Assert the QUBO matrix and Ising operator reproduce `knapsack_cost`."""
    qubo, offset = qubo_matrix(problem)
    ising, _ = qubo_to_ising(qubo, offset)
    bf = brute_force(problem)
    n = qubo.shape[0]

    values = np.array([z @ qubo @ z + offset for z in bf["assignments"]])
    if not np.allclose(values, bf["costs"]):
        raise AssertionError("QUBO matrix disagrees with sandia's knapsack_cost")

    # The cheap path, checked at every size.
    diag = cost_diagonal(problem)
    # Statevector index i has qubit q as bit q (little-endian); z-order matches.
    for idx, z in enumerate(bf["assignments"]):
        state_index = sum(int(b) << q for q, b in enumerate(z))
        if not np.isclose(diag[state_index], bf["costs"][idx]):
            raise AssertionError(
                f"cost_diagonal disagrees at z={z.tolist()}: "
                f"{diag[state_index]} != {bf['costs'][idx]}"
            )

    # And the dense Ising construction, wherever it still fits. `to_matrix()`
    # is 16 * 4**n bytes -- 268 MB at 12 variables, 68.7 GB at 16 -- so above
    # DENSE_VERIFY_MAX_VARS the operator is left unbuilt rather than allowed to
    # OOM the job. The identity above is the one the solver depends on; this is
    # the independent cross-check on `qubo_to_ising`, and skipping it at 16 is
    # sound because it holds at every size where it can be evaluated.
    if n <= DENSE_VERIFY_MAX_VARS:
        dense = np.real(np.diag(ising.to_matrix()))
        if not np.allclose(dense, diag):
            raise AssertionError(
                f"Ising operator disagrees with cost_diagonal at n={n}: "
                f"max |d| = {np.abs(dense - diag).max()}"
            )
    if not np.isclose(bf["best_cost"], problem_spec(problem)["h_opt"]):
        raise AssertionError(
            f"brute-force optimum {bf['best_cost']} != documented "
            f"H_opt {problem_spec(problem)['h_opt']}"
        )
    _ = n  # keep the signature honest about what was checked


def optimal_state_indices(problem: str = "knapsack4b") -> list[int]:
    """Statevector indices of the optimal assignment(s)."""
    bf = brute_force(problem)
    return [sum(int(b) << q for q, b in enumerate(z)) for z in bf["optimal_assignments"]]


def optimal_item_indices(problem: str = "knapsack4b") -> np.ndarray:
    """Statevector indices whose *item* variables are optimal, any slack.

    The knapsack's answer is the item selection. The slack variables exist only
    to turn "weight <= W_max" into the equality a QUBO needs, they are never
    read out, and given the items their correct value is forced and recoverable
    classically in constant time. So "did this run solve the knapsack" is the
    marginal over the slack register, not the probability of the exact
    all-variable assignment.

    Used for the ``p_items`` metric, which is reported **alongside** ``p_optimal``
    rather than instead of it -- the strict metric is the one a skeptic will ask
    for, and the two differ a lot on one side and not at all on the other. See
    `optimal_state_indices` for the strict version.

    This must be applied identically on both stacks or it is a thumb on the
    scale; `cvdv_multistart.one_start` computes the same marginal for CV-DV.
    """
    spec = problem_spec(problem)
    n_items, n_vars = spec["n_items"], spec["n_vars"]
    z_opt = brute_force(problem)["optimal_assignments"][0]
    idx = np.arange(2**n_vars)
    bits = (idx[:, None] >> np.arange(n_vars)[None, :]) & 1
    return np.flatnonzero(np.all(bits[:, :n_items] == z_opt[:n_items], axis=1))


# ---------------------------------------------------------------------------
# Ansaetze
# ---------------------------------------------------------------------------


def hardware_efficient_ansatz(
    n_qubits: int, n_layers: int, entangler: str = "ladder"
) -> QuantumCircuit:
    """RY/RZ rotations with a CZ entangler, `2*n_qubits*(n_layers+1)` params.

    `entangler` is ``"ladder"`` (linear chain, hardware-realistic) or
    ``"ring"`` (adds the wrap-around bond).
    """
    n_params = 2 * n_qubits * (n_layers + 1)
    theta = ParameterVector("t", n_params)
    qc = QuantumCircuit(n_qubits, name=f"he_L{n_layers}")
    p = 0

    def rotation_block():
        nonlocal p
        for q in range(n_qubits):
            qc.ry(theta[p], q)
            p += 1
            qc.rz(theta[p], q)
            p += 1

    rotation_block()
    for _ in range(n_layers):
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
        if entangler == "ring" and n_qubits > 2:
            qc.cz(n_qubits - 1, 0)
        rotation_block()
    return qc


def qaoa_ansatz(ising: SparsePauliOp, p: int) -> QuantumCircuit:
    """Standard QAOA: `|+>^n`, then `p` rounds of cost then mixer. `2p` params."""
    n = ising.num_qubits
    gammas = ParameterVector("g", p)
    betas = ParameterVector("b", p)
    qc = QuantumCircuit(n, name=f"qaoa_p{p}")
    qc.h(range(n))
    for layer in range(p):
        for pauli, coeff in zip(ising.paulis, ising.coeffs, strict=True):
            zs = [q for q in range(n) if str(pauli)[::-1][q] == "Z"]
            angle = 2 * float(np.real(coeff)) * gammas[layer]
            if len(zs) == 1:
                qc.rz(angle, zs[0])
            elif len(zs) == 2:
                qc.rzz(angle, zs[0], zs[1])
            # identity term is a global phase; skip
        qc.rx(2 * betas[layer], range(n))
    return qc


def golden_angle_init(n_params: int, offset: float = 0.4) -> np.ndarray:
    """Deterministic parameter schedule: `theta_k = offset + k * golden_angle`.

    Mirrors `sandia.ecd_vqe_sandia_jax.golden_angle_init`'s rationale: the
    all-zeros point is an exact stationary point of these ansaetze (every
    gradient vanishes), and random restarts make results irreproducible, so
    spread the initial angles quasi-uniformly instead.
    """
    return (offset + np.arange(n_params) * _GOLDEN_ANGLE) % (2 * np.pi)


def qaoa_ramp_init(p: int, ising: SparsePauliOp | None = None) -> np.ndarray:
    """Annealing-inspired linear-ramp initialization for QAOA's `2p` angles.

    QAOA is *not* served by the golden-angle schedule the other ansaetze use.
    Its landscape is structured, and the standard good starting point mirrors a
    discretized adiabatic sweep: `gamma` ramps up from 0 while `beta` ramps
    down to 0. Starting from scattered angles instead makes QAOA look worse
    than it is, which would strawman the baseline.

    `gamma` is scaled by the mean magnitude of the cost coefficients, since the
    knapsack QUBO's penalty term makes them large enough that an unscaled angle
    of order 1 wraps many times.
    """
    steps = (np.arange(p) + 0.5) / p
    scale = 1.0
    if ising is not None:
        weights = np.abs(np.real(np.asarray(ising.coeffs)))
        weights = weights[weights > 1e-12]
        if weights.size:
            scale = 1.0 / float(np.mean(weights))
    gammas = steps * scale
    betas = (1.0 - steps) * (np.pi / 4)
    # qaoa_ansatz declares ParameterVector "b" before "g"; Qiskit sorts a
    # circuit's parameters by name, so betas come first in the flat vector.
    return np.concatenate([betas, gammas])


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def run_dv_vqe(
    ansatz: QuantumCircuit,
    ising: SparsePauliOp,
    problem: str = "knapsack4b",
    maxiter: int = 2000,
    method: str = "BFGS",
    x0: np.ndarray | None = None,
) -> dict:
    """Minimize `<ising>` over `ansatz` parameters with an exact statevector.

    Returns final energy, probability of the optimal assignment(s), the
    approximation ratio, the most-likely bitstring, and the energy history.
    """
    n_params = ansatz.num_parameters
    if x0 is None:
        x0 = golden_angle_init(n_params)
    # `ising` no longer builds the energy -- see below -- but it still has to
    # describe the same register, and a mismatch here would otherwise surface
    # as a silently wrong energy rather than an error.
    n_vars = problem_spec(problem)["n_vars"]
    if ising.num_qubits != n_vars or ansatz.num_qubits != n_vars:
        raise ValueError(
            f"register mismatch for {problem} ({n_vars} variables): operator "
            f"has {ising.num_qubits} qubits, ansatz has {ansatz.num_qubits}"
        )
    # Diagonal in the computational basis, so the dense operator that
    # `ising.to_matrix()` would build is both unaffordable above 12 variables
    # (68.7 GB at 16) and unnecessary: <psi|H|psi> is a weighted sum of
    # probabilities. This also turns an O(4**n) matvec into an O(2**n) one.
    ham_diag = cost_diagonal(problem)
    opt_idx = optimal_state_indices(problem)
    bf = brute_force(problem)
    history: list[float] = []

    def energy(params):
        psi = Statevector.from_instruction(ansatz.assign_parameters(params)).data
        e = float(np.real(np.abs(psi) ** 2 @ ham_diag))
        history.append(e)
        return e

    result = minimize(energy, x0, method=method, options={"maxiter": maxiter})

    psi = Statevector.from_instruction(ansatz.assign_parameters(result.x)).data
    probs = np.abs(psi) ** 2
    e_final = float(probs @ ham_diag)
    best, worst = bf["best_cost"], bf["worst_cost"]
    return {
        "energy": e_final,
        "p_optimal": float(probs[opt_idx].sum()),
        "approx_ratio": float((worst - e_final) / (worst - best)),
        "most_likely_index": int(np.argmax(probs)),
        "most_likely_is_optimal": bool(int(np.argmax(probs)) in opt_idx),
        "probs": probs,
        "params": result.x,
        "n_params": n_params,
        "energy_history": np.array(history),
        "n_evaluations": int(result.nfev),
        "n_iterations": int(result.nit),
        "scipy_result": result,
    }


# ---------------------------------------------------------------------------
# Parallel sweeps
# ---------------------------------------------------------------------------
#
# A layer sweep is a set of completely independent optimizations. Running them
# serially leaves most of a multi-core machine idle. As in `jch_dv`, the worker
# lives at module level because macOS spawns fresh interpreters for worker
# processes and cannot pickle a function defined in a notebook cell.


def _run_sweep_job(job: dict) -> dict:
    """Worker: build one ansatz from a spec and optimize it.

    The ansatz is rebuilt inside the worker rather than shipped in, so only
    plain data crosses the process boundary.
    """
    problem = job.get("problem", "knapsack4b")
    qubo, offset = qubo_matrix(problem)
    ising, _ = qubo_to_ising(qubo, offset)

    x0 = None
    if job["kind"] == "hardware_efficient":
        ansatz = hardware_efficient_ansatz(
            job["n_qubits"], job["layers"], job.get("entangler", "ladder")
        )
    elif job["kind"] == "qaoa":
        ansatz = qaoa_ansatz(ising, job["p"])
        # QAOA gets the annealing-style ramp, not the golden-angle schedule --
        # see qaoa_ramp_init for why using the latter would understate QAOA.
        x0 = qaoa_ramp_init(job["p"], ising)
    else:
        raise ValueError(f"unknown ansatz kind {job['kind']!r}")

    # A `seed` turns the job into a random-start sample, which is how the
    # landscape study measures success *rate* rather than a single outcome.
    # Without one the deterministic initialization is used.
    if job.get("seed") is not None:
        rng = np.random.default_rng(job["seed"])
        x0 = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)

    from .resources import count_dv

    result = run_dv_vqe(ansatz, ising, problem, maxiter=job.get("maxiter", 4000), x0=x0)
    counts = count_dv(ansatz)
    counts.pop("transpiled", None)
    result["counts"] = counts
    result.pop("scipy_result", None)  # not reliably picklable
    result.update({k: v for k, v in job.items() if k != "problem"})
    return result


def sweep_vqe(jobs: list[dict], max_workers: int | None = None) -> list[dict]:
    """Optimize a list of ansatz specs in parallel, preserving order.

    Each job is a dict with `kind` (``"hardware_efficient"`` or ``"qaoa"``)
    plus that ansatz's parameters (`n_qubits`/`layers`, or `p`), and optionally
    `maxiter` and `problem`.
    """
    from .jch_dv import _parallel_map

    return _parallel_map(_run_sweep_job, jobs, max_workers)


# ---------------------------------------------------------------------------
# Optimizer-matched DV VQE
# ---------------------------------------------------------------------------
#
# `run_dv_vqe` above uses scipy BFGS with finite-difference gradients, because
# Qiskit's `Statevector` is not differentiable. The CV-DV side (`sandia`) uses
# JAX backprop with optax Adam. Comparing the two therefore confounds "which
# encoding" with "which optimizer", which is a real problem when the headline
# claim is about parameter efficiency.
#
# The functions below remove that confound: the same hardware-efficient ansatz,
# optimized with the *identical* optimizer, schedule, and iteration count that
# `sandia.ecd_vqe_sandia_jax.optimize_problem` uses on the CV-DV side.
#
# There is a pleasing symmetry that makes this exact rather than approximate:
# the knapsack Ising Hamiltonian is diagonal in the computational basis, so on
# both stacks the energy is literally `probs @ cost_diagonal`. The only
# difference left is the ansatz.

#: Adam settings copied from `sandia.ecd_vqe_sandia_jax.optimize_problem`.
MATCHED_N_ITERS = 8000
MATCHED_LR = 0.02
MATCHED_SCHEDULE = {0.5: 0.25, 0.82: 0.2}  # fraction of run -> LR multiplier


def _he_statevector_jax(params, n_qubits: int, n_layers: int):
    """RY/RZ + CZ-ladder ansatz as a raw JAX statevector.

    Written directly rather than through a simulator so the whole optimization
    compiles into one `jax.lax.scan`, matching how the CV-DV side is run.
    """
    import jax.numpy as jnp

    psi = jnp.zeros(2**n_qubits, dtype=jnp.complex128).at[0].set(1.0)
    psi = psi.reshape([2] * n_qubits)
    p = params.reshape(n_layers + 1, n_qubits, 2)

    def ry(state, q, theta):
        state = jnp.moveaxis(state, q, 0)
        c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
        out = jnp.stack([c * state[0] - s * state[1], s * state[0] + c * state[1]])
        return jnp.moveaxis(out, 0, q)

    def rz(state, q, theta):
        state = jnp.moveaxis(state, q, 0)
        phase = jnp.exp(-0.5j * theta)
        out = jnp.stack([phase * state[0], jnp.conj(phase) * state[1]])
        return jnp.moveaxis(out, 0, q)

    def cz(state, a, b):
        state = jnp.moveaxis(jnp.moveaxis(state, a, 0), b, 1)
        state = state.at[1, 1].set(-state[1, 1])
        return jnp.moveaxis(jnp.moveaxis(state, 1, b), 0, a)

    for layer in range(n_layers + 1):
        for q in range(n_qubits):
            psi = ry(psi, q, p[layer, q, 0])
            psi = rz(psi, q, p[layer, q, 1])
        if layer < n_layers:
            for q in range(n_qubits - 1):
                psi = cz(psi, q, q + 1)
    return psi.reshape(-1)


def run_dv_vqe_adam(
    n_layers: int,
    seed: int | None = None,
    problem: str = "knapsack4b",
    n_iters: int = MATCHED_N_ITERS,
    x0=None,
) -> dict:
    """Optimize the hardware-efficient ansatz with the CV-DV side's optimizer.

    Adam with the same piecewise-constant schedule (0.02 -> 0.005 -> 0.001),
    the same 8000 steps, and analytic backprop gradients. Pass `seed` for a
    random start, or `x0` for an explicit one; with neither, the deterministic
    golden-angle schedule is used.
    """
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", True)

    spec = problem_spec(problem)
    n_qubits = spec["n_vars"]
    cost_diag = jnp.array(cost_diagonal(problem))
    opt_indices = optimal_state_indices(problem)
    item_indices = optimal_item_indices(problem)
    bf = brute_force(problem)

    n_params = 2 * n_qubits * (n_layers + 1)
    if x0 is None:
        if seed is None:
            x0 = golden_angle_init(n_params)
        else:
            x0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, n_params)
    params = jnp.array(x0)

    schedule = optax.piecewise_constant_schedule(
        MATCHED_LR, {int(n_iters * k): v for k, v in MATCHED_SCHEDULE.items()}
    )
    optimizer = optax.adam(schedule)
    state = optimizer.init(params)

    def energy(p):
        psi = _he_statevector_jax(p, n_qubits, n_layers)
        return jnp.real(jnp.sum(jnp.abs(psi) ** 2 * cost_diag))

    value_and_grad = jax.value_and_grad(energy)

    def step(carry, _):
        p, st = carry
        value, grad = value_and_grad(p)
        updates, st = optimizer.update(grad, st, p)
        return (optax.apply_updates(p, updates), st), value

    (params, _), history = jax.lax.scan(step, (params, state), None, length=n_iters)

    psi = np.asarray(_he_statevector_jax(params, n_qubits, n_layers))
    probs = np.abs(psi) ** 2
    e_final = float(np.sum(probs * np.asarray(cost_diag)))
    best, worst = bf["best_cost"], bf["worst_cost"]
    return {
        "energy": e_final,
        "p_optimal": float(probs[opt_indices].sum()),
        # The knapsack answer, marginalizing the slack -- see optimal_item_indices.
        "p_items": float(probs[item_indices].sum()),
        "approx_ratio": float((worst - e_final) / (worst - best)),
        "probs": probs,
        "params": np.asarray(params),
        "n_params": n_params,
        "layers": n_layers,
        "seed": seed,
        "n_iterations": n_iters,
        "energy_history": np.asarray(history),
    }


def convergence_iters(history, tolerances=(1.0, 0.1, 0.01)) -> dict:
    """First iteration whose energy is within `tol` of the run's final energy.

    Reported at several tolerances on purpose: a single one hides whether a
    method gets *close* quickly and then crawls, or arrives late and stops
    dead. Both stacks compute this identically (see
    `cvdv_vs_dv/cvdv_multistart.py`), so the numbers are comparable.
    """
    history = np.asarray(history)
    final = history[-1]
    out = {}
    for tol in tolerances:
        hits = np.flatnonzero(history <= final + tol)
        out[str(tol)] = int(hits[0]) if hits.size else int(history.size)
    return out


def _run_adam_job(job: dict) -> dict:
    """One sweep entry, leaving no JAX state behind.

    `run_dv_vqe_adam` builds its `lax.scan` from a closure over `cost_diag`,
    `opt_indices` and `x0`, which XLA bakes in as compile-time constants. Every
    call therefore compiles a *new* executable that JAX's compilation cache
    holds on to, and a sweep runs 350 of them back to back in one process
    whenever `max_workers <= 1` -- which is exactly what the GPU path forces,
    since a forked CUDA context is unusable.

    Measured at 12 variables and 150 iterations: ~88 MB retained per job,
    growing without bound. That is ~30 GB across a full sweep at a toy
    iteration count, and the real 8000-step scans retain far more -- it is what
    OOM-killed both production sweeps at 63.7 GB of a 64 GB request, after
    n=8 (a 16x smaller state) had sailed through the same code minutes earlier.

    `jax.clear_caches()` releases them. It costs nothing here because the
    baked-in constants differ per job, so nothing was ever being reused.
    """
    import jax

    result = run_dv_vqe_adam(
        n_layers=job["layers"],
        seed=job.get("seed"),
        problem=job.get("problem", "knapsack4b"),
        n_iters=job.get("n_iters", MATCHED_N_ITERS),
    )
    history = result["energy_history"]
    result["convergence_iters"] = convergence_iters(history)
    result["history"] = [float(x) for x in np.asarray(history)[::50]]
    result.pop("probs", None)
    result.pop("energy_history", None)
    result.pop("params", None)
    jax.clear_caches()
    return result


def sweep_dv_vqe_adam(jobs: list[dict], max_workers: int | None = None) -> list[dict]:
    """Optimizer-matched sweep: `[{"layers": L, "seed": s}, ...]`, in parallel."""
    from .jch_dv import _parallel_map

    return _parallel_map(_run_adam_job, jobs, max_workers)
