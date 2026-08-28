# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""
Precision scaling for continuous optimization: what it costs each encoding to
locate a minimum to *d* decimal places.

`continuous_qaoa_comparison.ipynb` compares CV and DV at two fixed resolutions
on minima that sit at round numbers. That hides the axis where the two
encodings differ most sharply, because a round minimum is representable on both
sides. This module supplies the missing axis: a ladder of quadratics
``f(x) = (x - x*)^2`` whose minima are successive truncations of pi
(3.1, 3.14, 3.142, 3.1416, ...), so the *only* thing varying is how many
decimal places the answer needs.

**The asymmetry being measured.** A qumode's position is a genuine continuous
observable: ``<x>`` is not quantized, so *any* real ``x*`` is representable, and
the circuit that reaches it is the same circuit at every precision -- one
qumode, one Displacement + Squeezing + Rotation per layer, independent of ``d``.
A qubit register instead pins ``x`` to a lattice of ``2**n_q`` points. The
encoded cost function only exists on that lattice, so its minimum is the
lattice point nearest ``x*`` and the answer carries a **systematic offset** of
up to half a grid spacing. No amount of optimization removes it; only more
qubits do, and each qubit added rebuilds a wider circuit
(``O(n_q**d)`` Pauli terms for a degree-``d`` polynomial, plus an
``O(n_q**2)`` QFT).

**The other side of the ledger, which this module also computes.** That
argument is about the *mean*. If what is wanted is a state actually localized
to within ``eps`` -- variance, not just centre -- the qumode pays
``<n> ~ 1/(4 eps^2)`` photons where the register pays ``log2(L/eps)`` qubits, and
the qubit side wins that asymptotically. Both are reported, because quoting
either alone misstates the comparison. See :func:`photon_cost_for_width` and
:func:`qubits_for_width`.

Everything here is closed-form or a small exact computation: the grid offset is
a property of the encoding, not of an optimizer, so it is calculated rather
than searched for. The notebook then confirms with real QAOA runs that the
optimizer does reach the floor this module predicts.
"""

from __future__ import annotations

import math

import numpy as np

from .continuous_dv import DEFAULT_DOMAIN, position_grid

#: Decimal places the ladder runs over.
PRECISION_LADDER = (1, 2, 3, 4, 5)


def target(n_decimals: int) -> float:
    """``pi`` truncated to ``n_decimals`` decimal places.

    Successive truncations of one constant, rather than unrelated numbers, so
    that the number of decimals is the only variable across the ladder. The
    constant itself is irrelevant -- what matters is that the target is not a
    dyadic rational, so it never sits exactly on a power-of-two grid.
    """
    scale = 10**n_decimals
    return math.floor(math.pi * scale) / scale


def tolerance(n_decimals: int) -> float:
    """The error that counts as "correct to ``n_decimals`` places"."""
    return 0.5 * 10.0**(-n_decimals)


def quadratic(x_star: float) -> tuple[float, float, float]:
    """Ascending coefficients of ``(x - x_star)**2``."""
    return (x_star**2, -2.0 * x_star, 1.0)


# ---------------------------------------------------------------------------
# The DV grid floor
# ---------------------------------------------------------------------------


def grid_spacing(n_qubits: int, domain=DEFAULT_DOMAIN) -> float:
    """Distance between adjacent representable positions."""
    return (domain[1] - domain[0]) / (1 << n_qubits)


def grid_offset(x_star: float, n_qubits: int, domain=DEFAULT_DOMAIN) -> float:
    """Distance from ``x_star`` to the nearest representable position.

    This is the DV side's error floor for this target: the encoded cost
    function's minimum *is* the nearest grid point, so no optimizer can do
    better. Computed from the lattice directly rather than from the ``2**n``
    grid array, so it stays exact and cheap at 25 qubits.
    """
    delta = grid_spacing(n_qubits, domain)
    frac = (x_star - domain[0]) / delta
    return float(abs(frac - round(frac)) * delta)


def qubits_guaranteed(tol: float, domain=DEFAULT_DOMAIN) -> int:
    """Fewest qubits whose *half-spacing* is within ``tol``.

    The target-independent requirement: at this width every point of the domain
    is within ``tol`` of a grid point, so the encoding meets the tolerance
    whatever ``x*`` happens to be. This is the number to quote -- see
    :func:`qubits_achieved` for why the alternative is a lottery.
    """
    return math.ceil(math.log2((domain[1] - domain[0]) / (2.0 * tol)))


def qubits_achieved(
    x_star: float, tol: float, domain=DEFAULT_DOMAIN, max_qubits: int = 40
) -> int | None:
    """Fewest qubits whose grid happens to land within ``tol`` of *this* target.

    Can be smaller than :func:`qubits_guaranteed` when the target falls close
    to a grid point by luck, and is **not monotonic** in ``n_qubits`` -- a width
    that happens to straddle the target does worse than a narrower one that
    happens to bracket it. Reported for completeness, never as the headline: a
    resource number that depends on the arithmetic coincidences of one constant
    does not generalize to the next problem.
    """
    for n in range(1, max_qubits + 1):
        if grid_offset(x_star, n, domain) <= tol:
            return n
    return None


def grid_floor_table(x_star: float, n_qubits_range, domain=DEFAULT_DOMAIN) -> dict:
    """Achieved offset and half-spacing bound over a range of register widths."""
    ns = list(n_qubits_range)
    return {
        "n_qubits": np.array(ns),
        "offset": np.array([grid_offset(x_star, n, domain) for n in ns]),
        "half_spacing": np.array([0.5 * grid_spacing(n, domain) for n in ns]),
    }


def verify_grid_floor(x_star: float, n_qubits: int, domain=DEFAULT_DOMAIN) -> dict:
    """Check the closed-form offset against the explicit grid.

    :func:`grid_offset` does modular arithmetic where the rest of the module
    builds the actual ``2**n`` array; this holds the two against each other so
    a sign or an off-by-one in the lattice convention cannot go unnoticed.
    """
    grid = position_grid(n_qubits, domain)
    explicit = float(np.abs(grid - x_star).min())
    return {
        "explicit": explicit,
        "closed_form": grid_offset(x_star, n_qubits, domain),
        "argmin_x": float(grid[int(np.abs(grid - x_star).argmin())]),
    }


# ---------------------------------------------------------------------------
# The CV side of the same question: is x* representable at all?
# ---------------------------------------------------------------------------


def squeezed_coherent(x_star: float, squeeze: float, cutoff: int) -> np.ndarray:
    """Fock amplitudes of a squeezed vacuum displaced to position ``x_star``.

    The CV counterpart of "the nearest grid point": the most localized state
    this encoding can put at ``x_star``. Unlike a grid point it is not drawn
    from a discrete set -- the displacement is a continuous parameter, so the
    state can be centred on *any* real number, which is the whole asymmetry
    this module measures.

    Built by exponentiating the squeeze and displacement generators on the
    truncated Fock space rather than by a closed-form amplitude formula: at
    cutoff 512 the closed form's Hermite polynomials overflow, and the matrix
    exponential is both exact and cheap enough here.
    """
    from scipy.linalg import expm

    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    adag = a.conj().T
    # S(r) = exp[r/2 (a^2 - a^dag^2)] narrows x; D(alpha) = exp[alpha a^dag - alpha* a]
    # with a real alpha shifts <x> by sqrt(2) alpha.
    s_gen = 0.5 * squeeze * (a @ a - adag @ adag)
    alpha = x_star / np.sqrt(2.0)
    d_gen = alpha * (adag - a)
    psi = np.zeros(cutoff, dtype=complex)
    psi[0] = 1.0
    psi = expm(d_gen) @ (expm(s_gen) @ psi)
    return psi / np.linalg.norm(psi)


def cv_representation_error(x_star: float, squeeze: float, cutoff: int) -> dict:
    """How far the best CV state at ``x_star`` actually sits from ``x_star``.

    The CV analogue of :func:`grid_offset`, and the number that decides whether
    "a qumode can represent any real number" survives contact with a finite
    Fock truncation. It is not identically zero -- truncating the ladder clips
    the tail of a displaced squeezed state and pulls its mean in -- so it is
    measured rather than asserted, and the notebook reports it against the DV
    grid offset at the same precision.
    """
    from hyqbench_benchmarks.cv_qaoa import position_operators

    psi = squeezed_coherent(x_star, squeeze, cutoff)
    x, x2 = position_operators(cutoff)
    mean_x = float(np.real(np.vdot(psi, x @ psi)))
    mean_x2 = float(np.real(np.vdot(psi, x2 @ psi)))
    return {
        "mean_x": mean_x,
        "offset": abs(mean_x - x_star),
        "var_x": mean_x2 - mean_x**2,
        "mean_photon": mean_photon(psi),
        "leakage": fock_leakage(psi),
        "cutoff": cutoff,
    }


# ---------------------------------------------------------------------------
# Reading the answer off the position distribution
# ---------------------------------------------------------------------------


def mass_within(positions, probs, x_star: float, tol: float) -> float:
    """Probability mass inside ``[x* - tol, x* + tol]``.

    The distributional success metric: not "is the mean close" but "how much of
    the distribution is actually on the answer", which is what a single shot
    samples. Works for either encoding -- pass the grid and its probabilities
    for DV, or the x-quadrature points and their probabilities for CV.
    """
    positions = np.asarray(positions)
    probs = np.asarray(probs)
    inside = np.abs(positions - x_star) <= tol
    return float(probs[inside].sum() / probs.sum())


def dv_mass_is_structurally_zero(x_star: float, n_qubits: int, tol: float,
                                 domain=DEFAULT_DOMAIN) -> bool:
    """True when *no* representable position lies within ``tol`` of ``x_star``.

    When this holds, the qubit register's probability of landing within
    tolerance is exactly zero for **every** state it can prepare -- not small,
    not optimizer-dependent, zero -- because the support of any state is the
    grid and none of the grid is inside the window. It is the sharpest form of
    the discretization limit, and it needs no optimizer to establish.
    """
    return grid_offset(x_star, n_qubits, domain) > tol


# ---------------------------------------------------------------------------
# The width criterion -- the axis where the qubit register wins
# ---------------------------------------------------------------------------


def photon_cost_for_width(sigma: float) -> float:
    """Mean photon number of a squeezed vacuum of position width ``sigma``.

    A squeezed vacuum with ``Var(x) = e^(-2r)/2 = sigma^2`` carries
    ``<n> = sinh^2(r)``, so localizing to ``sigma`` costs ``O(1/sigma^2)``
    photons -- the energetic price of CV precision, and the reason the CV side
    cannot simply be run at a huge cutoff.
    """
    r = -0.5 * math.log(2.0 * sigma**2)
    return float(math.sinh(r) ** 2)


def cutoff_for_width(sigma: float) -> int:
    """Fock cutoff needed to hold a state of position width ``sigma``.

    From the truncation-limited resolution ``sigma ~ 2/sqrt(N_fock)`` that
    `continuous_qaoa_comparison.ipynb` section 2 uses, inverted.
    """
    return math.ceil((2.0 / sigma) ** 2)


def qubits_for_width(sigma: float, domain=DEFAULT_DOMAIN) -> int:
    """Qubits needed for a grid spacing of ``sigma``."""
    return math.ceil(math.log2((domain[1] - domain[0]) / sigma))


# ---------------------------------------------------------------------------
# CV diagnostics
# ---------------------------------------------------------------------------


def mean_photon(psi) -> float:
    """``<n>`` of a Fock-basis statevector -- the CV side's resource meter.

    The CV circuit does not change with precision, so photon number is where a
    harder CV target actually shows up in the cost.
    """
    probs = np.abs(np.asarray(psi).ravel()) ** 2
    probs = probs / probs.sum()
    return float(probs @ np.arange(probs.size))


def fock_leakage(psi, fraction: float = 0.125) -> float:
    """Probability in the top ``fraction`` of the Fock ladder.

    If this is not tiny the truncation is shaping the answer and the run has to
    be repeated at a larger cutoff -- the same check ``cv_qaoa.state_moments``
    reports, restated here so the precision sweep can assert on it.
    """
    probs = np.abs(np.asarray(psi).ravel()) ** 2
    probs = probs / probs.sum()
    k = max(2, int(probs.size * fraction))
    return float(probs[-k:].sum())


def ladder(precisions=PRECISION_LADDER, domain=DEFAULT_DOMAIN) -> list[dict]:
    """The precision ladder, with each rung's target, tolerance and DV cost."""
    rows = []
    for d in precisions:
        x_star = target(d)
        tol = tolerance(d)
        rows.append(
            {
                "decimals": d,
                "x_star": x_star,
                "tol": tol,
                "coeffs": quadratic(x_star),
                "n_qubits_guaranteed": qubits_guaranteed(tol, domain),
                "n_qubits_achieved": qubits_achieved(x_star, tol, domain),
            }
        )
    return rows
