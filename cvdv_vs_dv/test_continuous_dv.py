# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Tests for the qubit-only continuous-optimization baseline.

Two claims carry the comparison and are pinned here:

1. The Pauli expansion is *exact* -- the DV circuit really implements ``f(x)``
   on the grid, so a resource count taken from it is a count for the right
   unitary.
2. The term count grows as ``sum_k C(n_qubits, k)``, which is the ``O(n_q^d)``
   scaling the whole CV-versus-DV argument rests on.

`test_free_particle_spreads_correctly` is the one that would otherwise fail
silently: an incorrectly centred momentum grid still produces a unitary circuit
that conserves probability, it just simulates a different kinetic term.
"""

from math import comb

import numpy as np
import pytest

from cvdv_vs_dv import continuous_dv as dv

QUADRATIC = (9.0, -6.0, 1.0)  # (x - 3)^2
CUBIC = (2.8, -2.2, 0.1, 0.08)  # (x - 2)^2 + 0.08 (x - 2)^3


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits", [3, 5, 7])
@pytest.mark.parametrize("coeffs", [QUADRATIC, CUBIC, (0.0, 1.0)])
def test_pauli_expansion_is_exact(n_qubits, coeffs):
    """The Pauli-Z expansion must reproduce ``f(x_j)`` on every grid point."""
    op = dv.cost_operator(coeffs, n_qubits)
    reconstructed = np.real(np.diag(op.to_matrix()))
    expected = dv.poly_diagonal(coeffs, n_qubits)
    assert np.allclose(reconstructed, expected, atol=1e-9)


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_term_count_matches_binomial_sum(n_qubits, degree):
    """Non-identity terms must be exactly ``sum_{k=1..d} C(n_qubits, k)``.

    This is the ``O(n_q^d)`` blow-up the CV side avoids, so it is asserted
    rather than assumed: a regression that silently truncated small
    coefficients would understate the DV cost and flatter the comparison.
    """
    coeffs = [0.0] * degree + [1.0]  # a pure x^degree term
    counts = dv.term_counts(coeffs, n_qubits)
    assert counts["n_nonidentity"] == sum(comb(n_qubits, k) for k in range(1, degree + 1))
    assert counts["max_weight"] == degree


def test_cost_terms_all_commute():
    """All cost terms are diagonal, so Lie-Trotter synthesis is exact.

    This is what makes the DV baseline strong rather than a strawman: unlike
    the JCH study, the qubit side pays *no* Trotter error here at any degree.
    """
    op = dv.cost_operator(CUBIC, 5)
    for i in range(len(op)):
        for j in range(i + 1, len(op)):
            assert op.paulis[i].commutes(op.paulis[j])


def test_momentum_grid_is_centred():
    """Half the momenta must be negative -- an uncentred grid is a different model."""
    p = dv.momentum_grid(6)
    assert p.min() < 0 < p.max()
    assert np.isclose(p[0], 0.0)
    assert (p < 0).sum() == p.size // 2


def test_gaussian_initial_state_has_the_requested_variance():
    """The matched-start option must actually deliver the variance it advertises.

    The CV and DV sides are only comparable if they begin equally spread; an
    earlier version compared a uniform DV start (variance 12) against a CV
    squeezed vacuum (variance 3.7) and the mismatch was worth a factor of ~300
    in the reported localization.
    """
    n_qubits, target_var = 8, 3.695
    amps = dv.gaussian_amplitudes(n_qubits, target_var)
    probs = amps**2
    x = dv.position_grid(n_qubits)
    mean = probs @ x
    var = probs @ (x**2) - mean**2
    assert var == pytest.approx(target_var, rel=0.05)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def test_free_particle_spreads_correctly():
    """With no cost layer, the mixer alone must reproduce free-particle spreading.

    A minimum-uncertainty Gaussian of position variance ``s`` has momentum
    variance ``1/(4s)``, so after evolving for time ``gamma`` under ``p^2/2``
    its position variance is ``s + gamma^2 / (4 s)``. This is the check that
    catches a mis-centred or mis-scaled momentum grid, which is otherwise
    invisible: the circuit stays unitary either way.
    """
    n_qubits, var0 = 9, 1.0
    for gamma in (0.5, 1.0, 2.0):
        probs = dv.run_state(
            [0.0], [0.0], [gamma], n_qubits, initial=("gaussian", var0)
        )
        moments = dv.state_moments(probs, [0.0], n_qubits)
        expected = var0 + gamma**2 / (4 * var0)
        assert moments["var_x"] == pytest.approx(expected, rel=0.05)
        assert moments["edge_mass"] < 1e-3


def test_dv_finds_the_quadratic_minimum():
    """The DV baseline must solve the same problem the CV side solves.

    Load-bearing: if the two stacks did not agree on the answer, comparing
    their resource counts would be meaningless -- the same role
    `test_jch_dv_matches_cvdv_trajectory` plays in the JCH study.
    """
    result = dv.run_dv_qaoa(
        QUADRATIC, depth=5, n_qubits=8, maxiter=250,
        objective="localization", initial=("gaussian", 3.695),
    )
    assert result["problem"]["x_star"] == pytest.approx(3.0, abs=1e-9)
    assert result["x_error"] < 0.1
    assert result["edge_mass"] < 1e-2


def test_finer_grid_resolves_better():
    """Localization must improve with grid resolution, since it is grid-limited.

    Measured DV localization sits at roughly one grid cell, so this is the
    counterpart of the CV side's ``Var ~ 1 / Fock cutoff``: DV buys precision
    with qubits (logarithmically) where CV buys it with photons.
    """
    best = {}
    for n_qubits in (5, 8):
        result = dv.run_dv_qaoa(
            QUADRATIC, depth=5, n_qubits=n_qubits, maxiter=250,
            objective="localization", initial=("gaussian", 3.695),
        )
        best[n_qubits] = result["localization"]
    assert best[8] < best[5]


def test_energy_objective_rejected_for_unbounded_polynomial():
    """Same well-posedness guard as the CV side, so neither can run away."""
    with pytest.raises(ValueError, match="ill-posed"):
        dv.run_dv_qaoa(CUBIC, depth=1, n_qubits=5, objective="energy")


def test_summarize_starts_reports_a_distribution():
    """The summary must expose spread, not just a best-case number."""
    fake = [
        {"x_error": 0.01, "localization": 0.02, "energy": -1.0},
        {"x_error": 0.90, "localization": 1.50, "energy": 0.5},
        {"x_error": 0.02, "localization": 0.03, "energy": -0.9},
    ]
    summary = dv.summarize_starts(fake, tol=0.05)
    assert summary["n_starts"] == 3
    assert summary["n_converged"] == 2
    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["x_error_median"] == pytest.approx(0.02)
    assert summary["x_error_best"] == pytest.approx(0.01)
