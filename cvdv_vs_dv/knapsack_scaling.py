# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""
Design the knapsack instances used by the VQE **scaling** study, at 4, 8, 12
and 16 binary variables.

Why a search rather than four hand-written instances: the study's claim is
about how the CV-DV advantage grows with problem size, so the four instances
have to be comparable to each other. Picking them by hand would let unrelated
differences -- how tightly the optimum packs the knapsack, how large the
optimal Fock states are, how big the energy gap is -- ride along with the size
and contaminate the trend. :func:`design_instance` fixes those properties
across sizes and searches for values/weights that realize them.

**The bit layout.** ``sandia.ecd_vqe_sandia`` puts ``n_bits_m0`` item variables
on the primary qumode m0 (as the binary digits of its Fock number), one item
variable on the qubit, and the slack variable's ``n_bits_m1``-bit expansion on
m1. So an instance carries ``n_bits_m0 + 1 + n_bits_m1`` binary variables, and
a register size of exactly 4/8/12/16 needs the two bit counts to differ -- the
original symmetric layout only reaches the odd sizes 2b+1.

**Why the optimum is forced to low Fock numbers.** The Fock number of m0 *is*
the binary integer whose bits are the item variables, so an optimum that
selects high-index items needs a highly excited mode -- large ECD displacements,
more of the state pushed toward the truncation, and a harder optimization that
has nothing to do with problem size. Two things prevent that:

* the search requires the optimal m0 Fock number and slack to be small
  (:data:`MAX_TARGET_FOCK`, :data:`MAX_TARGET_SLACK`), and
* :func:`_relabel_to_low_bits` permutes the item indices so the selected items
  occupy the lowest bit positions.

The permutation is free: which wire carries which item is an encoding choice,
and the QUBO is identical up to relabelling, so the DV baseline solves exactly
the same problem (both the hardware-efficient ansatz and QAOA on a fully
connected graph are invariant under it). It is stated in the notebook rather
than left implicit, because it *is* a choice made in CV-DV's favour -- it is
the choice a practitioner would make, but it is not free of assumptions.

Run this module to print the instance dicts:

.. code-block:: bash

    uv run python -m cvdv_vs_dv.knapsack_scaling

The printed dicts are what ``sandia.ecd_vqe_sandia._PROBLEMS`` carries; this
module is kept so those constants can be re-derived rather than trusted.
"""

from __future__ import annotations

import itertools

import numpy as np

# Register sizes the scaling study reports.
SCALING_SIZES = (4, 8, 12, 16)

# Ceilings on the optimum's Fock numbers. The window on m0 is 2**n_bits_m0
# (up to 256 at 16 variables), so these are far below it by construction. The
# ceiling is what keeps the *photon* cost of the optimum flat across sizes:
# an ECD gate has to drive the mode to Fock n, so a target that grew with the
# register would confound problem size with drive amplitude.
MAX_TARGET_FOCK = 3
MAX_TARGET_SLACK = 1


def bit_split(n_vars: int) -> tuple[int, int]:
    """Item bits on m0 and slack bits on m1, for an ``n_vars``-variable instance.

    ``n_bits_m0 + 1 + n_bits_m1 == n_vars``, split so that m0 carries at least
    as many bits as m1 -- the natural knapsack shape, more items than capacity
    bits. The one exception is ``n_vars = 4``: the split (2, 1) leaves a slack
    capacity of 2**1 - 1 = 1, i.e. a knapsack that holds a single unit-weight
    item, which is too degenerate to be the small end of a trend. (1, 2) gives
    2 items against a capacity of 3 instead.
    """
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")
    if n_vars == 4:
        return 1, 2
    rest = n_vars - 1
    return rest - rest // 2, rest // 2


def _cost(x, y, values, weights, max_weight, l_val) -> float:
    """The QUBO cost, byte for byte what ``ecd_vqe_sandia.knapsack_cost`` computes."""
    value = sum(v * xi for v, xi in zip(values, x, strict=True))
    weight = sum(w * xi for w, xi in zip(weights, x, strict=True))
    slack = sum(2**b * yb for b, yb in enumerate(y))
    return -value + l_val * (max_weight - weight - slack) ** 2


def brute_force_exhaustive(values, weights, max_weight, l_val, n_bits_m1) -> dict:
    """Exhaustive minimum of the QUBO over all item *and* slack assignments.

    The reference implementation. :func:`brute_force` computes the same thing
    without enumerating the slack, and ``test_knapsack_scaling`` holds the two
    against each other.
    """
    n_items = len(values)
    best, best_z, n_best, second = np.inf, None, 0, np.inf
    for x in itertools.product((0, 1), repeat=n_items):
        for y in itertools.product((0, 1), repeat=n_bits_m1):
            c = _cost(x, y, values, weights, max_weight, l_val)
            if c < best - 1e-9:
                second, best, best_z, n_best = best, c, (x, y), 1
            elif abs(c - best) <= 1e-9:
                n_best += 1
            elif c < second - 1e-9:
                second = c
    x, y = best_z
    return {
        "best_cost": best,
        "second_cost": second,
        "gap": second - best,
        "degeneracy": n_best,
        "x": list(x),
        "y": list(y),
    }


def _subset_table(n_items: int) -> np.ndarray:
    """``(2**n_items, n_items)`` matrix of every item selection, LSB first."""
    idx = np.arange(2**n_items)
    return ((idx[:, None] >> np.arange(n_items)[None, :]) & 1).astype(np.int64)


def brute_force(values, weights, max_weight, l_val, n_bits_m1, table=None) -> dict:
    """Minimum of the QUBO, enumerating item selections only.

    The slack never has to be enumerated, because for a *fixed* item selection
    the optimal slack is forced and unique:

    * a packing that fits (``weight <= max_weight``) is best served by
      ``slack = max_weight - weight``, which zeroes the penalty -- and the slack
      bits cover exactly ``0..max_weight``, so that value is always available;
    * an overweight packing already has a negative residual, which any positive
      slack only makes more negative, so ``slack = 0`` is best.

    So the degeneracy of the full ``(x, y)`` problem equals the degeneracy over
    selections, and this is ``2**n_bits_m1`` times cheaper -- which is what
    makes a search over hundreds of thousands of candidates affordable.
    """
    n_items = len(values)
    table = _subset_table(n_items) if table is None else table
    value = table @ np.asarray(values, dtype=np.int64)
    weight = table @ np.asarray(weights, dtype=np.int64)

    fits = weight <= max_weight
    slack = np.where(fits, max_weight - weight, 0)
    residual = max_weight - weight - slack
    cost = -value + l_val * residual**2

    best = float(cost.min())
    hits = np.flatnonzero(cost <= best + 1e-9)
    rest = cost[cost > best + 1e-9]
    second = float(rest.min()) if rest.size else float("inf")
    x = table[hits[0]].tolist()
    y = [(int(slack[hits[0]]) >> b) & 1 for b in range(n_bits_m1)]
    return {
        "best_cost": best,
        "second_cost": second,
        "gap": second - best,
        "degeneracy": int(hits.size),
        "x": x,
        "y": y,
    }


def _relabel_to_low_bits(values, weights, selected, n_bits_m0):
    """Reorder items so the selected ones take the lowest bit positions.

    The last index is the qubit's item; the first ``n_bits_m0`` are m0's bits in
    increasing significance. Selected m0 items are moved to the low bits, so the
    optimal m0 Fock number is ``2**k - 1`` for ``k`` selected m0 items -- the
    smallest it can be for that count.
    """
    m0_sel = [i for i in range(n_bits_m0) if selected[i]]
    m0_unsel = [i for i in range(n_bits_m0) if not selected[i]]
    order = m0_sel + m0_unsel + [n_bits_m0]  # qubit item keeps the last slot
    return (
        [values[i] for i in order],
        [weights[i] for i in order],
        [selected[i] for i in order],
    )


def _score(info: dict) -> float:
    """Rank admissible candidates. Larger is better.

    The energy gap dominates -- a well-separated optimum is what makes an
    instance solvable at modest depth -- with mild preferences for a tightly
    packed knapsack (small slack) and small coefficients.
    """
    n_distinct = len(set(zip(info["values"], info["weights"], strict=True)))
    return (
        10.0 * info["gap"]
        - 0.5 * info["target"][2]
        - 0.05 * max(info["values"])
        - 0.05 * max(info["weights"])
        # Distinct items. Duplicates are legal and the search finds them at 4
        # variables (where every admissible instance selects both items, so the
        # gap term alone would pick two copies of the best one), but they read
        # as more contrived than the instance is. Weighted above the gap term's
        # spread so it decides ties rather than merely nudging them.
        - 15.0 * (info["n_items"] - n_distinct)
    )


# Constraints the search tries to satisfy, dropped one at a time from the end
# when no candidate satisfies them all. ``design_instance`` records which set it
# actually used, so an instance that had to give something up says so rather
# than looking like the others.
_RELAXATIONS = (
    ("binding", "qubit_item", "m0_item", "excited_slack"),
    ("qubit_item", "m0_item", "excited_slack"),
    ("qubit_item", "m0_item"),
    ("m0_item",),
    (),
)


def _search(
    n_vars: int, seed: int, n_candidates: int, max_value: int, required: tuple,
    split: tuple[int, int] | None = None,
) -> dict | None:
    """One pass of the instance search under a fixed constraint set.

    ``None`` if nothing admissible turned up. See :func:`design_instance`.
    """
    n_bits_m0, n_bits_m1 = bit_split(n_vars) if split is None else split
    if n_bits_m0 + 1 + n_bits_m1 != n_vars:
        raise ValueError(f"split {split} does not carry {n_vars} variables")
    n_items = n_bits_m0 + 1
    max_weight = 2**n_bits_m1 - 1  # the largest capacity the slack bits cover
    rng = np.random.default_rng(seed)
    table = _subset_table(n_items)

    best_info, best_score = None, -np.inf
    for _ in range(n_candidates):
        weights = rng.integers(1, max(2, max_weight // 2 + 1), size=n_items).tolist()
        values = rng.integers(1, max_value + 1, size=n_items).tolist()
        l_val = int(max(values)) + 1  # penalty above any value, the standard choice

        bf = brute_force(values, weights, max_weight, l_val, n_bits_m1, table)
        # A unique optimum is never relaxed: degenerate optima split the target
        # probability, which would make P(optimal) mean something different at
        # one size than at another.
        if bf["degeneracy"] != 1:
            continue
        sel = bf["x"]
        packed = sum(w * s for w, s in zip(weights, sel, strict=True))
        slack = max_weight - packed
        # The qubit's item and at least one m0 item selected, and a non-zero
        # slack, so that all three wires are actually exercised -- an instance
        # leaving a wire in the vacuum tests less of the ansatz.
        if "qubit_item" in required and not sel[-1]:
            continue
        if "m0_item" in required and not any(sel[:n_bits_m0]):
            continue
        if "excited_slack" in required and slack < 1:
            continue
        if slack > MAX_TARGET_SLACK:
            continue
        # Binding: some unselected item is excluded by the capacity rather than
        # by being worthless, i.e. this is a knapsack and not "take everything".
        if "binding" in required and all(
            packed + w <= max_weight for w, s in zip(weights, sel, strict=True) if not s
        ):
            continue

        values, weights, sel = _relabel_to_low_bits(values, weights, sel, n_bits_m0)
        m0_fock = sum(2**b for b in range(n_bits_m0) if sel[b])
        if m0_fock > MAX_TARGET_FOCK:
            continue

        info = {
            "n_items": n_items,
            "values": [int(v) for v in values],
            "weights": [int(w) for w in weights],
            "max_weight": int(max_weight),
            "l_val": int(l_val),
            "n_bits_m0": n_bits_m0,
            "n_bits_m1": n_bits_m1,
            "h_opt": float(bf["best_cost"]),
            "target": (int(sel[-1]), int(m0_fock), int(slack)),
            "gap": float(bf["gap"]),
            "degeneracy": int(bf["degeneracy"]),
            "n_vars": n_vars,
            "packed_weight": int(packed),
            "constraints": list(required),
        }
        s = _score(info)
        if s > best_score:
            best_info, best_score = info, s

    if best_info is None:
        return None

    # The relabelling must not have changed which assignment is optimal.
    check = brute_force(
        best_info["values"], best_info["weights"], best_info["max_weight"],
        best_info["l_val"], best_info["n_bits_m1"],
    )
    assert check["degeneracy"] == 1
    assert abs(check["best_cost"] - best_info["h_opt"]) < 1e-9
    assert best_info["target"][1] == sum(
        2**b for b in range(best_info["n_bits_m0"]) if check["x"][b]
    )
    return best_info


def design_instance(
    n_vars: int, seed: int = 0, n_candidates: int = 200_000, max_value: int = 9,
    split: tuple[int, int] | None = None,
) -> dict:
    """Search for a knapsack instance on ``n_vars`` binary variables.

    Returns a dict in ``ecd_vqe_sandia._PROBLEMS`` form plus the diagnostics the
    notebook reports (gap, degeneracy, optimal packing, and the constraint set
    that was met).

    Admissibility is the constraint set described on :data:`_RELAXATIONS`. The
    full set is not satisfiable at every size -- at 4 variables the layout has
    only 2 items against a capacity of 3, so a binding capacity and a non-zero
    slack cannot both hold -- so the search walks down the ladder and reports
    where it stopped instead of failing or silently lowering the bar.

    Deterministic given ``seed``.
    """
    for required in _RELAXATIONS:
        info = _search(n_vars, seed, n_candidates, max_value, required, split)
        if info is not None:
            return info
    raise RuntimeError(f"no admissible instance found for n_vars={n_vars}")


def simulation_cutoffs(n_bits_m0: int, n_bits_m1: int) -> tuple[int, int]:
    """Per-mode Fock cutoffs: the encoding window with headroom above it.

    The confinement penalty needs levels above the window to act on, but the
    headroom that matters is a *number of levels*, not a ratio -- and the cost
    of the simulation is the product of the two cutoffs, so a fixed ratio would
    be ruinous at 16 variables (a window of 256 on m0 at 8x headroom is a 2048-
    level mode). Small windows get 8x, large ones get 2x, which is still 128
    free levels above the m0 window at 16 variables.
    """
    def one(bits):
        window = 2**bits
        return window * (8 if window <= 16 else 2)

    return one(n_bits_m0), one(n_bits_m1)


def _format(name: str, info: dict) -> str:
    f0, f1 = simulation_cutoffs(info["n_bits_m0"], info["n_bits_m1"])
    return (
        f'    "{name}": dict(\n'
        f'        n_items={info["n_items"]}, values={info["values"]}, '
        f'weights={info["weights"]},\n'
        f'        max_weight={info["max_weight"]}, l_val={info["l_val"]},\n'
        f'        n_bits_m0={info["n_bits_m0"]}, n_bits_m1={info["n_bits_m1"]},\n'
        f'        h_opt={info["h_opt"]}, target={info["target"]}, n_depth=TBD,\n'
        f'        max_fock=({f0}, {f1}),\n'
        f'    ),'
    )


def main():  # pragma: no cover - reproducibility entry point
    for n in SCALING_SIZES:
        info = design_instance(n)
        print(_format(f"knapsack_n{n}", info))
        print(
            f"    # gap={info['gap']:.1f}  packed={info['packed_weight']}"
            f"/{info['max_weight']}  target={info['target']}  "
            f"n_vars={info['n_vars']}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
