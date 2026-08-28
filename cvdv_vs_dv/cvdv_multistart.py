# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""CV-DV ECD-VQE multi-start landscape probe, parallel over seeds.

Measures the CV-DV side the *same* way the DV side is measured, so the two
success rates are comparable: random initial parameters, same optimizer
settings as the production run, distribution over many starts.

Why this exists: `vqe_resource_comparison.ipynb` measures the DV ansatz over
random starts but the CV-DV production optimizer is deterministic, so comparing
them directly is not like-for-like. This re-runs the ECD ansatz from random
starts under otherwise identical settings, so both sides can be reported as
distributions.

Run from the repo root:
    python cvdv_vs_dv/cvdv_multistart.py <first_seed> <last_seed> [n_workers]

50 seeds takes roughly 20 minutes across 6 workers. The notebook caches the
result rather than re-running it.
"""

import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Keep each worker single-threaded; oversubscribing BLAS/XLA across processes
# is slower than running them serially.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

SANDIA = str(pathlib.Path(__file__).resolve().parent.parent / "sandia")
DEPTHS = (7,)  # depths to probe; the shipped knapsack4b configuration
N_ITERS = 8000
PROBLEM = "knapsack4b"


def one_start(args):
    """One random-start run. ``args`` is ``(depth, seed)`` or
    ``(depth, seed, problem)`` or ``(depth, seed, problem, n_iters)``.

    The Fock cutoffs are **not** arguments: they follow the problem, because
    each instance's encoding window sets what a valid cutoff even is (a 16-
    variable instance needs 256 levels on m0 where a 7-variable one needs 8).
    ``ecd_vqe_sandia_jax.resolve_fock`` is the single place that mapping lives.
    """
    depth, seed = args[0], args[1]
    problem = args[2] if len(args) > 2 else PROBLEM
    n_iters = args[3] if len(args) > 3 else N_ITERS
    if SANDIA not in sys.path:
        sys.path.insert(0, SANDIA)

    import jax
    import numpy as np
    import optax

    jax.config.update("jax_enable_x64", True)

    import ecd_vqe_sandia as base

    base.set_problem(problem)
    import ecd_vqe_sandia_jax as sj

    from hybridlane import random_ecd_params

    opt_fock = sj.resolve_fock(None, "opt")
    verify_fock = sj.resolve_fock(None)
    h_opt, _ = sj.build_cost_diagonal_jax(opt_fock, base.WIRES, 5.0)
    qn_opt = sj.get_qnode(opt_fock, base.WIRES)
    h_ver, iw_ver = sj.build_cost_diagonal_jax(verify_fock, base.WIRES, 5.0)
    qn_ver = sj.get_qnode(verify_fock, base.WIRES)

    params = jax.numpy.array(random_ecd_params(depth, rng=np.random.default_rng(seed)))
    schedule = optax.piecewise_constant_schedule(
        0.02, {int(n_iters * 0.5): 0.25, int(n_iters * 0.82): 0.2}
    )
    optimizer = optax.adam(schedule)
    state = optimizer.init(params)

    def energy(p):
        return sj.energy_jax(p, depth, h_opt, qn_opt, base.WIRES)

    value_and_grad = jax.jit(jax.value_and_grad(energy))

    def step(carry, _):
        p, st = carry
        value, g = value_and_grad(p)
        updates, st = optimizer.update(g, st, p)
        return (optax.apply_updates(p, updates), st), value

    (params, _), history = jax.lax.scan(step, (params, state), None, length=n_iters)
    history = np.asarray(history)

    sv = qn_ver(params, depth, base.WIRES)
    probs = np.real(np.asarray(sv * np.conj(sv)))
    q, m0, m1 = base.TARGET
    f0, f1 = verify_fock
    idx = q * f0 * f1 + m0 * f1 + m1
    # The knapsack answer lives on the qubit and m0 (the item variables); m1
    # carries only the slack, which is a reformulation device and is never read
    # out. `p_items` marginalizes over every *decodable* slack -- out-of-window
    # m1 is leakage and does not decode to a slack value at all, so including it
    # would credit leaked amplitude as a solution. `knapsack_dv.optimal_item_indices`
    # is the DV counterpart; the metric is meaningless unless both sides use it.
    flat = np.arange(2 * f0 * f1)
    items_mask = (
        (flat // (f0 * f1) == q)
        & ((flat // f1) % f0 == m0)
        & (flat % f1 < base.AUX_LEVELS)
    )
    return {
        "depth": depth,
        "seed": seed,
        "problem": problem,
        "n_vars": int(base.N_VARS),
        "params": [float(x) for x in np.asarray(params)],
        "confinement": float(np.sum(probs * np.asarray(iw_ver))),
        "energy": float(probs @ np.asarray(h_ver)),
        "p_optimal": float(probs[idx]),
        "p_items": float(probs[items_mask].sum()),
        # Convergence speed: first iteration whose energy is within `tol` of the
        # run's own final energy. Reported at several tolerances because a
        # single one hides whether a method gets *close* fast then crawls, or
        # arrives late and stops.
        "convergence_iters": convergence_iters(history),
        # Downsampled trace for plotting; the full 8000 points per run would
        # make the cached JSON unwieldy.
        "history": [float(x) for x in history[::50]],
    }


def convergence_iters(history, tolerances=(1.0, 0.1, 0.01)):
    """First iteration reaching within `tol` of the run's final energy."""
    history = np.asarray(history)
    final = history[-1]
    out = {}
    for tol in tolerances:
        hits = np.flatnonzero(history <= final + tol)
        out[str(tol)] = int(hits[0]) if hits.size else int(history.size)
    return out


def sweep_multistart(
    seeds: range | list[int],
    depths: tuple[int, ...] = DEPTHS,
    max_workers: int = 6,
    verbose: bool = True,
    problem: str = PROBLEM,
    n_iters: int = N_ITERS,
) -> list[dict]:
    """Run `one_start` over seeds in parallel and return the results in memory.

    The notebook calls this directly rather than shelling out to the CLI below:
    a subprocess writing JSON to a hard-coded `/tmp` path is not reproducible
    on a machine where that file is stale or absent, and the notebook then has
    no way to tell the two cases apart.

    `max_workers` defaults to 6 rather than the core count because each worker
    holds a JAX/XLA context at Fock 64; oversubscribing them thrashes memory.
    """
    jobs = [(d, s, problem, n_iters) for d in depths for s in seeds]
    if verbose:
        print(f"{len(jobs)} runs, {max_workers} workers", flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for res in pool.map(one_start, jobs):
            out.append(res)
            if verbose:
                print(
                    f"  depth {res['depth']} seed {res['seed']:2d}: "
                    f"E={res['energy']:9.4f} P(opt)={res['p_optimal']:.4f}",
                    flush=True,
                )
    return sorted(out, key=lambda r: (r["depth"], r["seed"]))


if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out = sweep_multistart(range(first, last + 1), max_workers=workers)

    path = f"/tmp/cvdv_multistart_{first}_{last}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
