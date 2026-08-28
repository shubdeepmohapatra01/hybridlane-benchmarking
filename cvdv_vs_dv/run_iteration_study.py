# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""
What is the cheapest (depth, iterations) pair that solves each instance?

The scaling sweeps pin ``DEPTH_SWEEP_ITERS = MULTISTART_ITERS = 8000`` at every
size and every depth, and at 12 variables that budget is provably binding: in
``scaling_cvdv_multistart_n12.npz`` *every* run is still descending when it
stops -- median ``conv_0p01`` is 7776 of 8000 for the two successes, 7564 for
the 0.30-plateau runs and 7874 for the failures, with a correlation to outcome
of only -0.20. At 16 variables it is worse (median 7945, and 18 of 50 runs end
at *positive* energy). Nothing has converged anywhere, so those success rates
describe an under-trained ansatz rather than the ansatz itself.

That also confounds every statement about depth. The budget is fixed while the
parameter count is not, so a deeper circuit gets the same number of steps to
fit more parameters and the depth sweep systematically penalises depth. The
n=12 result "0.9995 at depth 16, 0.5671 at depth 20" may say nothing about
depth 20's capacity -- only that it ran out of steps. Depth and iterations have
to be varied together or neither can be read.

**Why this is a real grid and not one long run.** It is tempting to run once at
32000 steps and read every shorter budget off the trajectory. That is wrong
here: the learning rate is
``optax.piecewise_constant_schedule(0.02, {0.5 * n_iters: 0.25, 0.82 * n_iters: 0.2})``,
whose decay points are *fractions* of the budget. A 32000-step run is therefore
not a 8000-step run continued -- it is a different schedule that is still at
the initial learning rate where the 8000-step run has already decayed twice.
Each budget has to be run.

**The descent monitor.** Every cell records how much energy was still being
gained over the last fifth of its run (``tail_drop``) and where the run first
reached the optimum (``iters_to_target``). A cell that succeeds with a flat
tail has converged and its budget is sufficient; a cell that fails with a steep
tail was merely truncated and deserves a larger budget before being called a
failure. This is the distinction that 8000-step data cannot make.

**Reading success.** The study's criterion is ``P(optimal) >= 0.9``, evaluated
once at the end of each run. Energy is *not* a safe substitute mid-run: across
the 150 archived multistart runs, ``E <= h_opt + 2.0`` never fired on a failure
(no false positives) but missed 5 of 20 genuine successes at n=8 -- the reported
energy carries the confinement penalty, so a state can be 95% on the optimum
and still sit above the target. ``iters_to_target`` is therefore recorded as a
conservative lower bound on when the run arrived, and the verdict for a cell is
always its final ``p_optimal``.

    # the grid, one size per job
    python -m cvdv_vs_dv.run_iteration_study --stage a --size 10 \
        --depths 6 8 10 12 14 16 --iters 8000 16000 32000 --device gpu

    # what it found
    python -m cvdv_vs_dv.run_iteration_study --report --sizes 8 10 12

    # reliability at the winning cell
    python -m cvdv_vs_dv.run_iteration_study --stage b --size 12 \
        --depth 10 --iters 32000 --seeds 20 --device gpu

Nothing here reads or writes ``cvdv_vs_dv/data/scaling_*.npz``; output goes to
``cvdv_vs_dv/data/iteration_study/``, one file per cell, so a run that hits its
time limit keeps everything it finished and re-running only fills the gaps.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Must precede any import that pulls in JAX -- it fixes the platform, which JAX
# caches at import time. Same contract as `run_scaling_sweeps`.
from cvdv_vs_dv import backend

REPO_ROOT = Path(__file__).resolve().parent.parent
SANDIA = str(REPO_ROOT / "sandia")
for _p in (str(REPO_ROOT), SANDIA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cvdv_vs_dv.run_scaling_sweeps import PROBLEMS  # noqa: E402

OUT_DIR = REPO_ROOT / "cvdv_vs_dv" / "data" / "iteration_study"

SUCCESS = 0.9          # P(optimal) defining "this cell works"; matches KNEE_THRESHOLD
HIST_POINTS = 512      # stored trace width, so cells of different budgets stack
TAIL_FRACTION = 0.2    # last fifth of a run defines its "still descending?" tail

#: Conservative energy target for `iters_to_target`, as `h_opt + ENERGY_TOL`.
#: Calibrated on the 150 archived multistart runs: at this tolerance the test
#: fired on no failure at any size, but missed 5 of 20 real successes at n=8.
#: One-sided by construction -- when it fires the run had arrived, when it does
#: not the run may still have arrived.
ENERGY_TOL = 2.0

#: Wall-clock minutes per (depth x 1000 iterations x beta), measured from the
#: production runs. The n=12 deterministic sweep took 123.8 min over depths
#: (6,8,10,12,14,16,20) at 8000 iters and three betas -> 0.060; its multistart
#: independently gives 8.29 min per depth-16 run at one beta -> 0.065, so the
#: linear model holds across both stages. n=8, n=10 and n=12 share it because
#: their Fock cutoff products are identical (128x64, 64x128, 128x64 = 8192);
#: n=16 is 512x256 = 131072, i.e. 16x the state, and gets its own rate from its
#: 143.2 min sweep.
MIN_PER_DEPTH_KILOITER = {8: 0.060, 10: 0.060, 12: 0.060, 16: 0.107}


def cell_path(size: int, depth: int, n_iters: int, beta_tag: str) -> Path:
    return OUT_DIR / f"cell_n{size}_d{depth}_it{n_iters}_b{beta_tag}.npz"


def _beta_tag(betas) -> str:
    return "-".join(f"{b:g}" for b in betas)


def _downsample(history, points: int = HIST_POINTS) -> np.ndarray:
    """Fixed-width trace, so cells with different budgets share one array."""
    h = np.asarray(history, dtype=float)
    if h.size == 0:
        return np.full(points, np.nan)
    return h[np.linspace(0, h.size - 1, points).round().astype(int)]


def descent_metrics(history, h_opt: float, tol: float = ENERGY_TOL) -> dict:
    """Did this run converge, or was it still going when the budget ended?

    ``tail_drop`` is the energy still being gained over the final
    ``TAIL_FRACTION`` of the run. Near zero means the optimizer has settled and
    a larger budget would not have helped; large means the cell was truncated
    and its failure says nothing about its depth.
    """
    h = np.asarray(history, dtype=float)
    n = h.size
    if n == 0:
        return {"tail_drop": np.nan, "iters_to_target": -1, "final_energy": np.nan}
    cut = max(0, int(n * (1 - TAIL_FRACTION)) - 1)
    hits = np.flatnonzero(h <= h_opt + tol)
    return {
        "tail_drop": float(h[cut] - h[-1]),
        "iters_to_target": int(hits[0]) if hits.size else -1,
        "final_energy": float(h[-1]),
    }


def run_cell(size: int, depth: int, n_iters: int, betas, force: bool = False) -> dict:
    """One deterministic optimization; its own file, so the grid is resumable."""
    import ecd_vqe_sandia as s
    import ecd_vqe_sandia_jax as sj

    path = cell_path(size, depth, n_iters, _beta_tag(betas))
    if path.exists() and not force:
        d = dict(np.load(path, allow_pickle=False))
        print(f"[cache] d={depth:2d} it={n_iters:6d}  E={float(d['energy']):9.4f} "
              f"P(opt)={float(d['p_optimal']):.4f}", flush=True)
        return d

    s.set_problem(PROBLEMS[size])
    h_opt = float(s._PROBLEMS[PROBLEMS[size]]["h_opt"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[compute] d={depth:2d} it={n_iters:6d} ...", flush=True)
    r = sj.optimize_problem(ndepth=depth, n_iters=n_iters, betas=tuple(betas),
                            verbose=False)
    wall = (time.time() - t0) / 60
    m = descent_metrics(r["energy_history"], h_opt)

    out = dict(
        size=size, depth=depth, n_iters=n_iters, h_opt=h_opt,
        n_params=int(np.asarray(r["params"]).size), n_ecd=2 * depth,
        energy=float(r["energy"]), p_optimal=float(r["p_optimal"]),
        confinement=float(r["confinement"]), beta=float(r["beta"]),
        params=np.asarray(r["params"]),
        energy_history=_downsample(r["energy_history"]),
        confinement_history=_downsample(r["confinement_history"]),
        history_len=int(np.asarray(r["energy_history"]).size),
        wall_min=wall, **m,
    )
    np.savez_compressed(path, **out)
    print(f"[compute] d={depth:2d} it={n_iters:6d}  E={out['energy']:9.4f} "
          f"P(opt)={out['p_optimal']:.4f} beta={out['beta']:.1f} "
          f"tail_drop={out['tail_drop']:8.4f} "
          f"{'SOLVED' if out['p_optimal'] >= SUCCESS else ''} ({wall:.1f} min)",
          flush=True)
    return out


def stage_a(size, depths, iters, betas, force=False) -> None:
    """The depth x iterations grid, cheapest cells first.

    Ordered by (iters, depth) so a job that runs out of wall-clock has finished
    whole low-budget rows rather than a ragged corner of the grid.
    """
    for n_iters in sorted(iters):
        for depth in sorted(depths):
            run_cell(size, depth, n_iters, betas, force)


def stage_b(size, depth, n_iters, n_seeds, workers, force=False) -> None:
    """Random starts at one cell: is the winning cell *reliably* winning?"""
    from cvdv_vs_dv.cvdv_multistart import sweep_multistart

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"multi_n{size}_d{depth}_it{n_iters}_s{n_seeds}.npz"
    if path.exists() and not force:
        print(f"[cache] {path.name}", flush=True)
        return

    t0 = time.time()
    print(f"[compute] multistart d={depth} it={n_iters} seeds=0..{n_seeds - 1}",
          flush=True)
    res = sweep_multistart(range(n_seeds), depths=(depth,), max_workers=workers,
                           problem=PROBLEMS[size], n_iters=n_iters, verbose=True)
    np.savez_compressed(
        path,
        seed=[r["seed"] for r in res], depth=[r["depth"] for r in res],
        n_iters=np.full(len(res), n_iters),
        energy=[r["energy"] for r in res],
        p_optimal=[r["p_optimal"] for r in res],
        p_items=[r["p_items"] for r in res],
        confinement=[r["confinement"] for r in res],
        conv_1p0=[r["convergence_iters"]["1.0"] for r in res],
        conv_0p1=[r["convergence_iters"]["0.1"] for r in res],
        conv_0p01=[r["convergence_iters"]["0.01"] for r in res],
        params=np.array([r["params"] for r in res]),
        # `run_scaling_sweeps` discards these; here they are the point, since
        # the question is where a truncated trajectory was heading.
        history=np.array([_downsample(r["history"]) for r in res]),
        history_len=[len(r["history"]) for r in res],
    )
    n_ok = sum(r["p_optimal"] >= SUCCESS for r in res)
    print(f"[compute] done in {(time.time() - t0) / 60:.1f} min -- "
          f"{n_ok}/{len(res)} reached P(opt) >= {SUCCESS}", flush=True)


def report(sizes) -> None:
    """Print the grid and the cheapest solving cell per size."""
    for size in sizes:
        cells = sorted(OUT_DIR.glob(f"cell_n{size}_*.npz"))
        if not cells:
            print(f"\nn={size}: no cells yet")
            continue
        rows = [dict(np.load(c, allow_pickle=False)) for c in cells]
        print(f"\n=== n={size} ({PROBLEMS[size]}, h_opt={float(rows[0]['h_opt']):.1f}) ===")
        print(f"{'depth':>6}{'iters':>8}{'n_par':>7}{'energy':>10}{'P(opt)':>9}"
              f"{'tail_drop':>11}{'to_target':>11}  verdict")
        for r in sorted(rows, key=lambda r: (int(r["depth"]), int(r["n_iters"]))):
            p, tail = float(r["p_optimal"]), float(r["tail_drop"])
            tgt = int(r["iters_to_target"])
            if p >= SUCCESS:
                verdict = "SOLVED" + ("" if tail < 0.05 else "  (tail still moving)")
            elif tail >= 0.05:
                verdict = "truncated -- give it more iterations"
            else:
                verdict = "converged to a non-solution"
            print(f"{int(r['depth']):6d}{int(r['n_iters']):8d}{int(r['n_params']):7d}"
                  f"{float(r['energy']):10.4f}{p:9.4f}{tail:11.4f}"
                  f"{(tgt if tgt >= 0 else -1):11d}  {verdict}")

        solved = [r for r in rows if float(r["p_optimal"]) >= SUCCESS]
        if not solved:
            trunc = [r for r in rows if float(r["tail_drop"]) >= 0.05]
            print(f"  --> nothing solved. {len(trunc)}/{len(rows)} cells were still "
                  f"descending; raise --iters before concluding anything about depth.")
            continue
        # Cheapest = fewest parameters, then fewest iterations. Depth is the
        # hardware cost (ECD gates on a real device); iterations are only
        # simulation time, so depth breaks the tie first.
        best = min(solved, key=lambda r: (int(r["n_params"]), int(r["n_iters"])))
        print(f"  --> cheapest solving cell: depth {int(best['depth'])} "
              f"({int(best['n_params'])} params, {2 * int(best['depth'])} ECD gates) "
              f"at {int(best['n_iters'])} iterations, P(opt)={float(best['p_optimal']):.4f}")
        min_d = min(int(r["depth"]) for r in solved)
        at_min = [r for r in solved if int(r["depth"]) == min_d]
        print(f"  --> shallowest solving depth: {min_d}, needing "
              f"{min(int(r['n_iters']) for r in at_min)} iterations")


def _estimate_minutes(size, depth, n_iters, n_runs=1, betas=1) -> float:
    return MIN_PER_DEPTH_KILOITER.get(size, 0.060) * depth * (n_iters / 1000) \
        * n_runs * betas


def main():  # pragma: no cover - driver
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("a", "b", "ab"), default="a")
    ap.add_argument("--report", action="store_true",
                    help="print the grid found so far and exit")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12],
                    help="--report only")
    ap.add_argument("--size", type=int, default=12, choices=sorted(PROBLEMS))
    ap.add_argument("--depths", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16])
    ap.add_argument("--iters", type=int, nargs="+", default=[8000, 16000, 32000],
                    help="budgets to run; stage B uses the last one")
    ap.add_argument("--depth", type=int, default=None,
                    help="stage B: the single depth to draw random starts at")
    ap.add_argument("--seeds", type=int, default=20,
                    help="stage B starts. Five would not resolve a rate near "
                         "5%%: it returns zero successes 77%% of the time")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.8],
                    help="stage A initial displacements. Each beta is a full "
                         "optimization, so the production default (0.6 0.8 1.0) "
                         "is 3x the cost; 0.8 alone is the grid default and the "
                         "winning cell can be re-run with all three")
    ap.add_argument("--device", default="auto", choices=backend.VALID)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.report:
        report(args.sizes)
        return

    if args.dry_run:
        total = 0.0
        tag = _beta_tag(args.betas)
        print(f"\nplan for n={args.size} ({PROBLEMS[args.size]}), "
              f"betas={args.betas}:", flush=True)
        if "a" in args.stage:
            for it in sorted(args.iters):
                for d in sorted(args.depths):
                    done = cell_path(args.size, d, it, tag).exists()
                    m = _estimate_minutes(args.size, d, it, 1, len(args.betas))
                    total += 0 if done else m
                    print(f"  A  d={d:2d} it={it:6d}  ~{m:6.1f} min"
                          f"{'  [cached]' if done else ''}")
        if "b" in args.stage:
            d = args.depth if args.depth is not None else max(args.depths)
            m = _estimate_minutes(args.size, d, sorted(args.iters)[-1], args.seeds)
            total += m
            print(f"  B  d={d:2d} it={sorted(args.iters)[-1]:6d} x {args.seeds} "
                  f"seeds  ~{m:6.1f} min")
        print(f"\n  estimated total: {total / 60:.1f} h\n", flush=True)
        return

    backend.select(args.device)
    backend.verify()
    print(backend.report(), flush=True)
    # Forking a live CUDA context yields workers whose context is unusable, so
    # GPU runs stay serial -- same constraint as the production sweeps.
    workers = 1 if backend.select(args.device) == "gpu" else args.workers

    if "a" in args.stage:
        print(f"\n=== stage A: depth x iterations, n={args.size} ===", flush=True)
        stage_a(args.size, args.depths, args.iters, tuple(args.betas), args.force)
        report([args.size])
    if "b" in args.stage:
        depth = args.depth if args.depth is not None else max(args.depths)
        print(f"\n=== stage B: multistart, n={args.size} d={depth} ===", flush=True)
        stage_b(args.size, depth, sorted(args.iters)[-1], args.seeds, workers,
                args.force)
    print("\nall done", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
