# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""
Build every cached dataset the VQE scaling study needs, one size at a time.

`vqe_scaling_comparison.ipynb` reads these caches and does not compute them:
the full set is several hours of optimization, which is not something a
notebook should do inside a cell that a reader might re-run by accident. Run
this first, then the notebook renders from `cvdv_vs_dv/data/`.

    uv run python -m cvdv_vs_dv.run_scaling_sweeps 4 8 12
    uv run python -m cvdv_vs_dv.run_scaling_sweeps 16 --starts 12 --device gpu

``--device`` picks the JAX backend (``auto`` / ``cpu`` / ``gpu``). The small
sizes run on a laptop CPU, which is what the demo notebooks reproduce; 12 and 16
variables belong on a GPU node -- see ``cvdv_vs_dv/hazel/README.md`` for the
Slurm scripts. Nothing about the physics changes with the backend, and the
caches are interchangeable between them.

Each dataset is written to its own ``.npz`` and skipped if already present, so
an interrupted run resumes where it stopped and adding a size never invalidates
the sizes already computed. Pass ``--force`` to recompute.

**Why the depth sweep comes first and the multistart second.** The ECD ansatz
needs a different depth at each problem size -- the knee moves from 5 at 4
variables to 12 at 8 -- so a shared default depth would either handicap the
large instances or overstate the cost of the small ones. The sweep locates each
instance's own knee (the shallowest depth reaching ``KNEE_THRESHOLD``), and the
multistart then runs there.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Must precede any import that pulls in JAX -- it fixes the platform, which JAX
# caches at import time. The sweep functions import jax lazily inside their
# workers for the same reason.
from cvdv_vs_dv import backend

REPO_ROOT = Path(__file__).resolve().parent.parent
SANDIA = str(REPO_ROOT / "sandia")
for _p in (str(REPO_ROOT), SANDIA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = REPO_ROOT / "cvdv_vs_dv" / "data"

#: Every register size in the study, smallest first so a run that is cut short
#: has produced the cheap points rather than none of them.
SIZES = (4, 7, 8, 10, 12, 16)

#: Problem name per register size. 7 is `knapsack4b`, the instance
#: `vqe_resource_comparison.ipynb` already settled -- it joins the series as a
#: fifth point rather than being re-run, see `adopt_knapsack4b`.
PROBLEMS = {4: "knapsack_n4", 7: "knapsack4b", 8: "knapsack_n8",
            10: "knapsack_n10", 12: "knapsack_n12", 16: "knapsack_n16"}

#: Depths to try when locating each instance's knee. Deeper instances need
#: deeper ansaetze, and a sweep that stopped at 8 everywhere would report the
#: larger ones as unsolvable when they are merely deeper.
DEPTH_GRID = {
    4: (1, 2, 3, 4, 5, 6, 7, 8),
    # knapsack4b. Normally adopted from vqe_resource_comparison.ipynb rather
    # than recomputed (see `adopt_knapsack4b`), but a grid is defined so that
    # `--force 7` can rebuild it from scratch on the same footing as the others.
    7: (1, 2, 3, 4, 5, 6, 7, 8),
    8: (4, 6, 8, 10, 12, 14),
    # 10 fills the gap between 8 and 12, where the random-start success rate
    # falls from 44% to 4%; that is too coarse a step to locate where the ECD
    # ansatz starts failing. Same grid as 12, since the two instances have the
    # same simulation cost (Fock cutoff product 8192 for both).
    10: (6, 8, 10, 12, 14, 16, 20),
    12: (6, 8, 10, 12, 14, 16, 20),
    16: (8, 12, 16, 20),
}

#: Initial displacement magnitudes swept by the deterministic initializer. This
#: is `sandia`'s production default, and it is not a formality: at 8 variables
#: and depth 10, beta = 0.8 alone reaches P(optimal) = 0.937 where beta = 0.6
#: reaches 0.989. Sweeping one beta per size would make the knee an artifact of
#: which beta happened to suit that instance.
BETAS = (0.6, 0.8, 1.0)

#: DV hardware-efficient layer counts, matching vqe_resource_comparison.ipynb.
DV_LAYERS = (1, 2, 3, 4, 6, 8, 12)

KNEE_THRESHOLD = 0.9       # P(optimal) defining "this depth works"
N_STARTS = 50              # random starts per configuration
DEPTH_SWEEP_ITERS = 8000
MULTISTART_ITERS = 8000
WORKERS = 5


def cache_status(sizes=SIZES) -> list[dict]:
    """Which datasets exist and which are missing, per size."""
    kinds = ("cvdv_depth", "cvdv_multistart", "dv_adam")
    out = []
    for size in sizes:
        have = {k: (DATA_DIR / f"scaling_{k}_n{size}.npz").exists() for k in kinds}
        out.append({"size": size, **have, "complete": all(have.values())})
    return out


def print_plan(sizes, force: bool = False) -> None:
    """Say what this run will and will not recompute, before it starts.

    Existing caches are **kept** unless ``--force`` is passed: a sweep costs
    hours and silently redoing one that already exists is the expensive kind of
    mistake. This prints the decision so it is visible in the job log rather
    than inferred from timings afterwards.
    """
    print("\nplan (existing caches are kept; pass --force to recompute):",
          flush=True)
    for row in cache_status(sizes):
        todo = [k for k in ("cvdv_depth", "cvdv_multistart", "dv_adam") if not row[k]]
        if force:
            state = "RECOMPUTE ALL (--force)"
        elif not todo:
            state = "complete, nothing to do"
        else:
            state = "will compute: " + ", ".join(todo)
        print(f"  n={row['size']:<3} {state}", flush=True)
    print(flush=True)


def cache(name: str, compute, force: bool = False) -> dict:
    """Compute-and-save, or load. Same contract as the notebooks' `cached`."""
    path = DATA_DIR / f"{name}.npz"
    if path.exists() and not force:
        print(f"[cache] {name}", flush=True)
        return dict(np.load(path, allow_pickle=False))
    DATA_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"[compute] {name} ...", flush=True)
    data = {k: np.asarray(v) for k, v in compute().items()}
    np.savez_compressed(path, **data)
    print(f"[compute] {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return data


def cvdv_depth_sweep(size: int, force: bool = False) -> dict:
    """Re-optimize the ECD ansatz at every depth in this size's grid.

    Deterministic (golden-angle init, no RNG), so this reproduces exactly. The
    optimal parameters are saved alongside the metrics -- the point of the cache
    is that nobody has to re-run the optimizer to redraw a figure or to hand the
    converged angles to someone with hardware.
    """
    problem = PROBLEMS[size]

    def _compute():
        import ecd_vqe_sandia as s
        import ecd_vqe_sandia_jax as sj

        s.set_problem(problem)
        rows = {k: [] for k in
                ("depth", "n_params", "n_ecd", "energy", "p_optimal", "confinement")}
        params = []
        for d in DEPTH_GRID[size]:
            r = sj.optimize_problem(ndepth=d, n_iters=DEPTH_SWEEP_ITERS,
                                    betas=BETAS, verbose=False)
            rows["depth"].append(d)
            rows["n_params"].append(int(np.asarray(r["params"]).size))
            rows["n_ecd"].append(2 * d)
            rows["energy"].append(float(r["energy"]))
            rows["p_optimal"].append(float(r["p_optimal"]))
            rows["confinement"].append(float(r["confinement"]))
            params.append(np.asarray(r["params"]))
            print(f"    depth {d:2d}: E={r['energy']:10.4f} P(opt)={r['p_optimal']:.4f}",
                  flush=True)
        # Ragged (one vector per depth), so pad to a rectangle and record the
        # true lengths rather than saving an object array -- `np.load` with
        # allow_pickle=False is what keeps these caches safe to redistribute.
        width = max(p.size for p in params)
        padded = np.full((len(params), width), np.nan)
        for i, p in enumerate(params):
            padded[i, : p.size] = p
        rows["params"] = padded
        rows["param_lengths"] = [p.size for p in params]
        return rows

    return cache(f"scaling_cvdv_depth_n{size}", _compute, force)


def knee_depth(sweep: dict, threshold: float = KNEE_THRESHOLD) -> int:
    """Shallowest swept depth reaching `threshold`, else the best one swept."""
    depths = np.asarray(sweep["depth"])
    p = np.asarray(sweep["p_optimal"])
    hits = np.flatnonzero(p >= threshold)
    return int(depths[hits[0]] if hits.size else depths[int(np.argmax(p))])


def cvdv_multistart(size: int, depth: int, n_starts: int, force: bool = False) -> dict:
    """N random starts at `depth`, the CV-DV half of the matched comparison."""
    problem = PROBLEMS[size]

    def _compute():
        from cvdv_vs_dv.cvdv_multistart import sweep_multistart

        res = sweep_multistart(range(n_starts), depths=(depth,), max_workers=WORKERS,
                               problem=problem, n_iters=MULTISTART_ITERS, verbose=True)
        return {
            "seed": [r["seed"] for r in res],
            "depth": [r["depth"] for r in res],
            "energy": [r["energy"] for r in res],
            "p_optimal": [r["p_optimal"] for r in res],
            "p_items": [r["p_items"] for r in res],
            "confinement": [r["confinement"] for r in res],
            "conv_1p0": [r["convergence_iters"]["1.0"] for r in res],
            "conv_0p1": [r["convergence_iters"]["0.1"] for r in res],
            "conv_0p01": [r["convergence_iters"]["0.01"] for r in res],
            "params": np.array([r["params"] for r in res]),
        }

    return cache(f"scaling_cvdv_multistart_n{size}", _compute, force)


def dv_adam_sweep(size: int, n_starts: int, force: bool = False) -> dict:
    """Hardware-efficient VQE under the *same* optimizer, over layers and seeds."""
    problem = PROBLEMS[size]

    def _compute():
        from cvdv_vs_dv import knapsack_dv as kd

        jobs = [{"layers": L, "seed": sd, "problem": problem}
                for L in DV_LAYERS for sd in range(n_starts)]
        res = kd.sweep_dv_vqe_adam(jobs, max_workers=WORKERS)
        return {
            "layers": [r["layers"] for r in res],
            "n_params": [r["n_params"] for r in res],
            "energy": [r["energy"] for r in res],
            "p_optimal": [r["p_optimal"] for r in res],
            "p_items": [r["p_items"] for r in res],
            "approx_ratio": [r["approx_ratio"] for r in res],
            "conv_1p0": [r["convergence_iters"]["1.0"] for r in res],
            "conv_0p1": [r["convergence_iters"]["0.1"] for r in res],
            "conv_0p01": [r["convergence_iters"]["0.01"] for r in res],
        }

    return cache(f"scaling_dv_adam_n{size}", _compute, force)


def adopt_knapsack4b(force: bool = False) -> None:
    """Re-file `vqe_resource_comparison.ipynb`'s 7-variable caches into this series.

    That notebook already ran `knapsack4b` under **exactly** this protocol -- 50
    random starts, Adam, 8000 steps, backprop gradients, success at
    P(optimal) >= 0.9, and the same hardware-efficient layer counts -- so
    re-running it would burn an hour to reproduce numbers that already exist,
    and would risk reporting a *different* number for the same instance in two
    notebooks.

    The old caches predate this module, so two fields have to be supplied:
    ``depth``, which was implicit in the filename (`..._depth7`), and the
    per-run parameter vectors, which that sweep did not save. Anything reading
    these files must tolerate a missing ``params`` -- the scaling notebook does,
    and says so where it matters.
    """
    pairs = [
        ("cvdv_depth_sweep", "scaling_cvdv_depth_n7"),
        ("cvdv_multistart_depth7", "scaling_cvdv_multistart_n7"),
        ("dv_adam_matched", "scaling_dv_adam_n7"),
    ]
    for old, new in pairs:
        src_path, dst_path = DATA_DIR / f"{old}.npz", DATA_DIR / f"{new}.npz"
        if dst_path.exists() and not force:
            print(f"[cache] {new}", flush=True)
            continue
        if not src_path.exists():
            raise FileNotFoundError(
                f"{src_path.name} is missing -- run vqe_resource_comparison.ipynb "
                "first, or drop 7 from the sizes"
            )
        data = dict(np.load(src_path, allow_pickle=False))
        if new.endswith("multistart_n7"):
            data["depth"] = np.full(data["seed"].shape, 7)
            data.pop("history", None)   # only the 7-variable cache carries it
        np.savez_compressed(dst_path, **data)
        print(f"[adopt] {old} -> {new}", flush=True)


def backfill_p_items(sizes=(4, 7, 8, 12, 16), verbose: bool = True) -> None:
    """Add ``p_items`` to CV-DV caches written before the metric existed.

    Cheap, because the multistart caches store the converged parameter vectors:
    the states are re-evaluated, not re-optimized. A 50-seed sweep backfills in
    seconds where re-running it would take hours.

    **The DV side is not backfilled here**, and that is deliberate rather than an
    omission: `sweep_dv_vqe_adam` discards its probability vectors, so there is
    nothing to recompute from and the runs have to be redone. They are cheap at
    4-8 variables and expensive at 12, so they ride along with the next sweep
    instead. Until then a cache may legitimately have ``p_items`` on one side
    only, and anything reporting the metric must require it on *both* -- applying
    the marginal to CV-DV alone would be a thumb on the scale.
    """
    import ecd_vqe_sandia as s
    import ecd_vqe_sandia_jax as sj

    for size in sizes:
        path = DATA_DIR / f"scaling_cvdv_multistart_n{size}.npz"
        if not path.exists():
            continue
        data = dict(np.load(path, allow_pickle=False))
        if "p_items" in data:
            if verbose:
                print(f"[skip] n={size} already has p_items", flush=True)
            continue
        if "params" not in data:
            # The 7-variable cache was adopted from vqe_resource_comparison.ipynb,
            # whose sweep predates saving parameters. Nothing to recompute from.
            if verbose:
                print(f"[skip] n={size}: no stored parameters (adopted cache)", flush=True)
            continue

        s.set_problem(PROBLEMS[size])
        depth = int(data["depth"][0])
        f0, f1 = sj.resolve_fock(None)
        qn = sj.get_qnode(None)
        tq, tm0, _ = s.TARGET
        flat = np.arange(2 * f0 * f1)
        items_mask = (
            (flat // (f0 * f1) == tq)
            & ((flat // f1) % f0 == tm0)
            & (flat % f1 < s.AUX_LEVELS)
        )

        p_items = []
        for params in data["params"]:
            probs = np.abs(np.asarray(qn(np.asarray(params), depth, s.WIRES))) ** 2
            p_items.append(float(probs[items_mask].sum()))
        data["p_items"] = np.asarray(p_items)
        np.savez_compressed(path, **data)
        if verbose:
            arr = np.asarray(p_items)
            print(f"[backfill] n={size}: p_items mean={arr.mean():.3f} "
                  f"converged={int((arr >= 0.9).sum())}/{arr.size} "
                  f"(p_optimal was {data['p_optimal'].mean():.3f}, "
                  f"{int((data['p_optimal'] >= 0.9).sum())}/{arr.size})", flush=True)


def run_size(size: int, n_starts: int = N_STARTS, force: bool = False) -> dict:
    print(f"\n=== {size} variables ({PROBLEMS[size]}) ===", flush=True)
    if size == 7:
        adopt_knapsack4b(force)
        return {"size": 7, "depth": 7}
    sweep = cvdv_depth_sweep(size, force)
    depth = knee_depth(sweep)
    print(f"  knee at depth {depth} ({8 * depth} parameters)", flush=True)
    cvdv_multistart(size, depth, n_starts, force)
    dv_adam_sweep(size, n_starts, force)
    return {"size": size, "depth": depth}


def main():  # pragma: no cover - driver
    global WORKERS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sizes", nargs="*", type=int, default=list(SIZES),
                    help=f"register sizes to build (default: all of {list(SIZES)})")
    ap.add_argument("--starts", type=int, default=N_STARTS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="auto", choices=backend.VALID,
                    help="JAX backend (default: auto-detect)")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="worker processes for the multistart sweeps; use 1 on "
                         "GPU, where the processes would contend for one card")
    args = ap.parse_args()
    print_plan(args.sizes, args.force)
    device = backend.select(args.device)
    # select() runs before JAX exists and can only see whether the *machine* has
    # a GPU; verify() checks where JAX actually landed, which is the only way to
    # catch a CUDA-less JAX quietly using the CPU on a GPU node. Fatal on
    # purpose: that failure costs the whole allocation and is invisible in the
    # output files.
    backend.verify()
    print(backend.report(), flush=True)

    # One process on GPU: several would queue on the same card and each would
    # preallocate against it.
    WORKERS = 1 if device == "gpu" else args.workers

    for size in args.sizes:
        run_size(size, args.starts, args.force)
    print("\nall done", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
