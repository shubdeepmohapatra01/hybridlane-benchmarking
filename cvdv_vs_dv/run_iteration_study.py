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

from cvdv_vs_dv.run_scaling_sweeps import DV_LAYERS, PROBLEMS  # noqa: E402

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

#: Same idea for the DV arm, per (1000 iterations x seed), independent of layer
#: count -- the cost is dominated by the 2**n statevector, not the parameter
#: count. n=8 is measured (350 jobs at 8000 iters in 210.8 min -> 0.075); the
#: larger sizes are extrapolated by state size and are rough, since the DV
#: sweeps at 12 and 16 have never completed.
DV_MIN_PER_KILOITER = {8: 0.075, 10: 0.15, 12: 0.30, 16: 1.2}


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

    out = {
        "size": size, "depth": depth, "n_iters": n_iters, "h_opt": h_opt,
        "n_params": int(np.asarray(r["params"]).size), "n_ecd": 2 * depth,
        "energy": float(r["energy"]), "p_optimal": float(r["p_optimal"]),
        "confinement": float(r["confinement"]), "beta": float(r["beta"]),
        "params": np.asarray(r["params"]),
        "energy_history": _downsample(r["energy_history"]),
        "confinement_history": _downsample(r["confinement_history"]),
        "history_len": int(np.asarray(r["energy_history"]).size),
        "wall_min": wall, **m,
    }
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


def multistart_cell(size, depth, n_iters, n_seeds, workers, force=False,
                    seed_start: int = 0) -> dict | None:
    """Random starts at one (depth, budget), cached like a stage-A cell.

    This is the measurement that matters for the study's headline: the CV-DV
    ansatz is *deployed* from random parameters, so its success rate under
    random init -- not what the golden-angle initializer can reach -- is what
    compares against the DV arm's own random-start distribution.
    """
    from cvdv_vs_dv.cvdv_multistart import sweep_multistart

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Seeds 0..n-1 keep the original filename so the cells already computed
    # stay discoverable; a chunk starting elsewhere gets a `_from{k}` suffix.
    # `report_multistart` stitches the chunks of one (depth, budget) back
    # together, so splitting a 50-seed cell across jobs is invisible downstream.
    suffix = "" if seed_start == 0 else f"_from{seed_start}"
    path = OUT_DIR / f"multi_n{size}_d{depth}_it{n_iters}_s{n_seeds}{suffix}.npz"
    if path.exists() and not force:
        d = dict(np.load(path, allow_pickle=False))
        p = np.asarray(d["p_optimal"])
        print(f"[cache] d={depth:2d} it={n_iters:6d}  "
              f"{int((p >= SUCCESS).sum())}/{p.size} solved", flush=True)
        return d

    t0 = time.time()
    seeds = range(seed_start, seed_start + n_seeds)
    print(f"[compute] d={depth:2d} it={n_iters:6d} seeds {seeds.start}"
          f"..{seeds.stop - 1} ...", flush=True)
    res = sweep_multistart(seeds, depths=(depth,), max_workers=workers,
                           problem=PROBLEMS[size], n_iters=n_iters, verbose=True)
    p_opt = np.array([r["p_optimal"] for r in res])
    # conv_* are relative to each run's *own* final energy, so a value near the
    # budget means that run was still descending when it stopped. Averaged over
    # seeds this is the direct test of "is the budget the binding constraint?".
    conv = np.array([r["convergence_iters"]["0.01"] for r in res], dtype=float)
    out = {
        "size": size, "depth": depth, "n_iters": n_iters, "n_seeds": n_seeds,
        "seed_start": seed_start,
        "seed": [r["seed"] for r in res],
        "energy": [r["energy"] for r in res], "p_optimal": p_opt,
        "p_items": [r["p_items"] for r in res],
        "confinement": [r["confinement"] for r in res],
        "conv_1p0": [r["convergence_iters"]["1.0"] for r in res],
        "conv_0p1": [r["convergence_iters"]["0.1"] for r in res],
        "conv_0p01": conv,
        "params": np.array([r["params"] for r in res]),
        "history": np.array([_downsample(r["history"]) for r in res]),
        "history_len": [len(r["history"]) for r in res],
        # fraction of the budget consumed before the run stopped improving
        "truncation": float(np.mean(conv / n_iters)),
        "wall_min": (time.time() - t0) / 60,
    }
    np.savez_compressed(path, **out)
    n_ok = int((p_opt >= SUCCESS).sum())
    print(f"[compute] d={depth:2d} it={n_iters:6d}  {n_ok}/{n_seeds} solved, "
          f"best P(opt)={p_opt.max():.4f}, truncation={out['truncation']:.2f} "
          f"({out['wall_min']:.1f} min)", flush=True)
    return out


def stage_b(size, depths, iters, n_seeds, workers, force=False,
            seed_start: int = 0) -> None:
    """The random-start grid: every (depth, budget), `n_seeds` starts each.

    Ordered cheapest-first (budget outer, depth inner) so a job that runs out
    of wall-clock has finished whole low-budget rows rather than a ragged
    corner, and every cell is cached independently so a resubmit resumes.
    """
    for n_iters in sorted(iters):
        for depth in sorted(depths):
            multistart_cell(size, depth, n_iters, n_seeds, workers, force,
                            seed_start)


def dv_cell(size, layers, n_iters, n_seeds, workers, force=False) -> dict | None:
    """DV hardware-efficient VQE at one (layers, budget), `n_seeds` starts.

    Exists so the comparison stays *optimizer-matched*. `vqe_resource_comparison`
    rests on both arms getting the same optimizer, the same step count and the
    same success criterion; the moment CV-DV is given 32000 iterations and DV is
    left at `MATCHED_N_ITERS = 8000`, the headline gap is partly a budget gap and
    the claim is no longer defensible.

    In practice this should not move the DV numbers -- DV's median `conv_0p01`
    is 386 at n=8 and 1954 at n=12, so it has long since converged by 8000 and
    the extra steps are wasted on it. That is the point: "both arms were offered
    32000 steps, DV plateaued at ~2000 and CV-DV used all of them" is a far
    stronger statement than one where only CV-DV was measured at the larger
    budget.
    """
    from cvdv_vs_dv.knapsack_dv import sweep_dv_vqe_adam

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"dv_n{size}_L{layers}_it{n_iters}_s{n_seeds}.npz"
    if path.exists() and not force:
        d = dict(np.load(path, allow_pickle=False))
        p = np.asarray(d["p_optimal"])
        print(f"[cache] L={layers:2d} it={n_iters:6d}  "
              f"{int((p >= SUCCESS).sum())}/{p.size} solved", flush=True)
        return d

    t0 = time.time()
    print(f"[compute] DV L={layers:2d} it={n_iters:6d} x {n_seeds} seeds ...",
          flush=True)
    jobs = [{"layers": layers, "seed": sd, "problem": PROBLEMS[size],
             "n_iters": n_iters} for sd in range(n_seeds)]
    res = sweep_dv_vqe_adam(jobs, max_workers=workers)
    p_opt = np.array([r["p_optimal"] for r in res])
    conv = np.array([r["convergence_iters"]["0.01"] for r in res], dtype=float)
    out = {
        "size": size, "layers": layers, "n_iters": n_iters, "n_seeds": n_seeds,
        "n_params": [r["n_params"] for r in res],
        "energy": [r["energy"] for r in res], "p_optimal": p_opt,
        "p_items": [r["p_items"] for r in res],
        "approx_ratio": [r["approx_ratio"] for r in res],
        "conv_1p0": [r["convergence_iters"]["1.0"] for r in res],
        "conv_0p1": [r["convergence_iters"]["0.1"] for r in res],
        "conv_0p01": conv,
        "truncation": float(np.mean(conv / n_iters)),
        "wall_min": (time.time() - t0) / 60,
    }
    np.savez_compressed(path, **out)
    n_ok = int((p_opt >= SUCCESS).sum())
    print(f"[compute] DV L={layers:2d} it={n_iters:6d}  {n_ok}/{n_seeds} solved, "
          f"best P(opt)={p_opt.max():.4f}, truncation={out['truncation']:.2f} "
          f"({out['wall_min']:.1f} min)", flush=True)
    return out


def stage_dv(size, layers, iters, n_seeds, workers, force=False) -> None:
    """The DV arm over the same budgets the CV-DV grid uses."""
    for n_iters in sorted(iters):
        for n_layers in sorted(layers):
            dv_cell(size, n_layers, n_iters, n_seeds, workers, force)


def report_dv(sizes) -> None:
    """DV success rate per (layers, budget) -- the matched comparison arm."""
    for size in sizes:
        cells = sorted(OUT_DIR.glob(f"dv_n{size}_*.npz"))
        if not cells:
            continue
        rows = [dict(np.load(c, allow_pickle=False)) for c in cells]
        print(f"\n=== n={size} DV arm ({PROBLEMS[size]}) ===")
        print(f"{'layers':>7}{'n_par':>7}{'iters':>8}{'seeds':>7}{'solved':>8}"
              f"{'rate':>8}{'bestP':>8}{'trunc':>8}")
        for r in sorted(rows, key=lambda r: (int(r["layers"]), int(r["n_iters"]))):
            p = np.asarray(r["p_optimal"])
            npar = int(np.asarray(r["n_params"])[0])
            ok = int((p >= SUCCESS).sum())
            print(f"{int(r['layers']):7d}{npar:7d}{int(r['n_iters']):8d}{p.size:7d}"
                  f"{ok:8d}{ok / p.size:8.0%}{p.max():8.4f}"
                  f"{float(r['truncation']):8.2f}")


#: A seed counts as solved only if its P(optimal) is reproduced at twice the
#: verification cutoff. Tolerance is loose because agreement is either exact
#: (a confined state: identical to ~1e-10) or gross (a state pressed against
#: the ceiling, where the cutoff changes the simulated dynamics, not just the
#: readout -- one n=12 seed moved from E=2.73 to E=13.70). There is no middle
#: ground to calibrate against, so anything but near-exact agreement is a
#: truncation artifact.
CUTOFF_TOL = 1e-6


def validate_cutoff(sizes, factor: int = 2, force: bool = False) -> None:
    """Re-evaluate every stored seed at `factor` x the verification cutoff.

    A converged, confined state gives identical numbers at any cutoff above it
    -- checked here to 1e-6, and observed to agree to ten decimal places. A
    state that pushes population toward the ceiling does not, because the
    cutoff bounds the simulated Hilbert space rather than merely the readout.
    So agreement is what licenses the claim "this seed solved the problem"
    rather than "this seed solved a truncated caricature of it".

    Cheap enough to apply to everything already computed: the cells store their
    converged parameters, so this is one forward pass per seed (~0.5 s at n=12)
    against the ~2000 s that produced it. Results are written back into each
    cell as `p_optimal_hi`, `energy_hi` and `cutoff_ok`.
    """
    import ecd_vqe_sandia as base
    import ecd_vqe_sandia_jax as sj

    for size in sizes:
        for path in sorted(OUT_DIR.glob(f"multi_n{size}_*.npz")):
            d = dict(np.load(path, allow_pickle=False))
            if "cutoff_ok" in d and not force:
                print(f"[cache] {path.name}", flush=True)
                continue
            if "params" not in d:
                print(f"[skip]  {path.name}: no stored parameters", flush=True)
                continue

            base.set_problem(PROBLEMS[size])
            depth = int(d["depth"])
            f0, f1 = sj.resolve_fock(None)
            hi = (f0 * factor, f1 * factor)
            h_hi, iw_hi = sj.build_cost_diagonal_jax(hi, base.WIRES, 5.0)
            qn_hi = sj.get_qnode(hi, base.WIRES)
            tq, tm0, tm1 = base.TARGET

            params = np.asarray(d["params"])
            mask_hi = np.asarray(iw_hi).astype(float)
            p_hi, e_hi, c_hi = [], [], []
            for row in params:
                probs = np.abs(np.asarray(qn_hi(row, depth, base.WIRES))) ** 2
                p_hi.append(float(probs[tq * hi[0] * hi[1] + tm0 * hi[1] + tm1]))
                e_hi.append(float(np.dot(probs, np.asarray(h_hi))))
                # Confinement re-measured in the larger space: a state that only
                # looked confined because the old cutoff hid where it went shows
                # up here, and is the mechanism behind a failed cutoff check.
                c_hi.append(float(np.sum(probs * mask_hi)))
            p_hi, e_hi, c_hi = np.array(p_hi), np.array(e_hi), np.array(c_hi)
            p_lo = np.asarray(d["p_optimal"])
            ok = np.abs(p_hi - p_lo) < CUTOFF_TOL

            d.update({"p_optimal_hi": p_hi, "energy_hi": e_hi,
                      "confinement_hi": c_hi, "cutoff_ok": ok,
                      "cutoff_factor": factor})
            # `savez_compressed` appends `.npz` unless the name already ends
            # in it, so the temp name has to carry the extension itself or the
            # rename below chases a file that was never written under that name.
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(tmp, **d)
            tmp.replace(path)
            solved = p_lo >= SUCCESS
            print(f"[valid] {path.name}: {int(ok.sum())}/{ok.size} reproduce at "
                  f"{hi}; of the {int(solved.sum())} solved, "
                  f"{int((solved & ~ok).sum())} fail the check", flush=True)


def report_multistart(sizes) -> None:
    """Success rate per (depth, budget), the number the study actually needs."""
    for size in sizes:
        cells = sorted(OUT_DIR.glob(f"multi_n{size}_*.npz"))
        if not cells:
            continue
        chunks = [dict(np.load(c, allow_pickle=False)) for c in cells]
        # Stitch seed-chunks of the same (depth, budget) back into one cell, so
        # a 50-seed run split across two jobs reports as 50 seeds rather than
        # two unrelated rows. Seeds are deduplicated because a re-run with a
        # different --seed-start can overlap an existing chunk.
        merged: dict[tuple[int, int], dict] = {}
        for c in chunks:
            key = (int(c["depth"]), int(c["n_iters"]))
            m = merged.setdefault(key, {"depth": key[0], "n_iters": key[1],
                                        "seed": [], "p_optimal": [],
                                        "energy": [], "conv_0p01": []})
            for field in ("seed", "p_optimal", "energy", "conv_0p01"):
                m[field].extend(np.asarray(c[field]).ravel().tolist())
            # Seeds whose P(optimal) does not survive a doubled cutoff are
            # counted separately rather than dropped: silently discarding them
            # would make an unvalidated cell look identical to a clean one.
            if "cutoff_ok" in c:
                bad = (~np.asarray(c["cutoff_ok"])) & \
                      (np.asarray(c["p_optimal"]) >= SUCCESS)
                m["n_unsafe"] = m.get("n_unsafe", 0) + int(bad.sum())
        rows = []
        for key, m in merged.items():
            seen, keep = set(), []
            for i, sd in enumerate(m["seed"]):
                if sd not in seen:
                    seen.add(sd)
                    keep.append(i)
            idx = np.array(keep, dtype=int)
            rows.append({
                "depth": key[0], "n_iters": key[1],
                "seed": np.array(m["seed"])[idx],
                "p_optimal": np.array(m["p_optimal"])[idx],
                "energy": np.array(m["energy"])[idx],
                "truncation": float(np.mean(np.array(m["conv_0p01"])[idx]) / key[1]),
                "n_unsafe": m.get("n_unsafe", 0),
            })
        print(f"\n=== n={size} random starts ({PROBLEMS[size]}) ===")
        print(f"{'depth':>6}{'iters':>8}{'seeds':>7}{'solved':>8}{'rate':>8}"
              f"{'bestP':>8}{'medP':>8}{'medE':>10}{'trunc':>8}  budget")
        for r in sorted(rows, key=lambda r: (int(r["depth"]), int(r["n_iters"]))):
            p = np.asarray(r["p_optimal"])
            e = np.asarray(r["energy"])
            n = p.size
            ok = int((p >= SUCCESS).sum())
            unsafe = int(r.get("n_unsafe", 0))
            trunc = float(r["truncation"])
            # >0.9 means the median run was still improving at the very end:
            # the budget, not the depth, is what stopped it.
            note = "BINDING" if trunc > 0.9 else "adequate" if trunc < 0.75 else "marginal"
            flag = "" if unsafe == 0 else f"  [{unsafe} fail cutoff check]"
            print(f"{int(r['depth']):6d}{int(r['n_iters']):8d}{n:7d}{ok:8d}"
                  f"{ok / n:8.0%}{p.max():8.4f}{np.median(p):8.4f}"
                  f"{np.median(e):10.3f}{trunc:8.2f}  {note}{flag}")
        best = max(rows, key=lambda r: ((np.asarray(r["p_optimal"]) >= SUCCESS).mean(),
                                        -int(r["depth"]), -int(r["n_iters"])))
        bp = np.asarray(best["p_optimal"])
        if (bp >= SUCCESS).any():
            print(f"  --> best rate: depth {int(best['depth'])} "
                  f"({2 * int(best['depth'])} ECD gates) at {int(best['n_iters'])} "
                  f"iterations, {int((bp >= SUCCESS).sum())}/{bp.size}")
        else:
            print("  --> no cell produced a single success at any depth or budget")


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
    ap.add_argument("--stage", choices=("a", "b", "d", "ab", "bd", "abd"),
                    default="a",
                    help="a=deterministic depth grid, b=random-start grid, "
                         "d=DV arm at the same budgets (keeps the comparison "
                         "optimizer-matched)")
    ap.add_argument("--validate", action="store_true",
                    help="re-evaluate every stored seed at twice the "
                         "verification cutoff and record whether its P(optimal) "
                         "survives; run once after new cells land")
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
    ap.add_argument("--seeds", type=int, default=8,
                    help="random starts per stage-B cell. This grid trades "
                         "seeds for coverage: 8 detects a large shift in the "
                         "rate (5%% -> 50%%) but cannot estimate a small one -- "
                         "at a true 5%%, 8 draws come back empty 66%% of the "
                         "time. Re-run the winning cell with 50 to quote a rate")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.8],
                    help="stage A initial displacements. Each beta is a full "
                         "optimization, so the production default (0.6 0.8 1.0) "
                         "is 3x the cost; 0.8 alone is the grid default and the "
                         "winning cell can be re-run with all three")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                    help="first random seed for stage B. Lets a 50-seed cell be "
                         "split across jobs (--seeds 25 --seed-start 0, then "
                         "--seeds 25 --seed-start 25); --report stitches the "
                         "chunks back into one row")
    ap.add_argument("--layers", type=int, nargs="+", default=list(DV_LAYERS),
                    help=f"stage D layer counts (default {list(DV_LAYERS)}, "
                         "matching vqe_resource_comparison)")
    ap.add_argument("--device", default="auto", choices=backend.VALID)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.validate:
        backend.select(args.device)
        validate_cutoff(args.sizes, force=args.force)
        report_multistart(args.sizes)
        return

    if args.report:
        report(args.sizes)
        report_multistart(args.sizes)
        report_dv(args.sizes)
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
            depths = [args.depth] if args.depth is not None else args.depths
            for it in sorted(args.iters):
                for d in sorted(depths):
                    sfx = "" if args.seed_start == 0 else f"_from{args.seed_start}"
                    done = (OUT_DIR / f"multi_n{args.size}_d{d}_it{it}"
                            f"_s{args.seeds}{sfx}.npz").exists()
                    m = _estimate_minutes(args.size, d, it, args.seeds)
                    total += 0 if done else m
                    print(f"  B  d={d:2d} it={it:6d} x {args.seeds} seeds "
                          f"~{m:6.1f} min{'  [cached]' if done else ''}")
        if "d" in args.stage:
            for it in sorted(args.iters):
                for n_layers in sorted(args.layers):
                    done = (OUT_DIR / f"dv_n{args.size}_L{n_layers}_it{it}"
                            f"_s{args.seeds}.npz").exists()
                    m = (DV_MIN_PER_KILOITER.get(args.size, 0.3)
                         * (it / 1000) * args.seeds)
                    total += 0 if done else m
                    print(f"  D  L={n_layers:2d} it={it:6d} x {args.seeds} seeds "
                          f"~{m:6.1f} min{'  [cached]' if done else ''}")
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
        depths = [args.depth] if args.depth is not None else args.depths
        print(f"\n=== stage B: random starts, n={args.size}, "
              f"{args.seeds} seeds/cell ===", flush=True)
        stage_b(args.size, depths, args.iters, args.seeds, workers, args.force,
                args.seed_start)
        report_multistart([args.size])
    if "d" in args.stage:
        print(f"\n=== stage D: DV arm, n={args.size}, "
              f"{args.seeds} seeds/cell ===", flush=True)
        stage_dv(args.size, args.layers, args.iters, args.seeds, workers,
                 args.force)
        report_dv([args.size])
    print("\nall done", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
