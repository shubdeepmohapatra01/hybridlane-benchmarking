# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Correctness tests for the pure-DV baselines.

These guard the claim the whole study rests on: the DV baseline solves the
*same* problem as the CV-DV benchmark. If the encoding or the QUBO drifts, the
resource comparison is meaningless, so these run in the fast test path.
"""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from cvdv_vs_dv import boson_encoding as be
from cvdv_vs_dv import knapsack_dv as kd
from cvdv_vs_dv.jch_dv import JchDvLayout, jch_dv_terms, run_jch_dv

CUTOFFS = [2, 4, 8]


# ---------------------------------------------------------------------------
# Boson encodings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", be.ENCODINGS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_ladder_action_on_encoded_levels(encoding, cutoff):
    """`a|k> = sqrt(k)|k-1>` and `n|k> = k|k>` on the encoded subspace."""
    ops = be.encode_ops(cutoff, encoding)
    idx = be.basis_index_map(cutoff, encoding)
    dim = 1 << ops["n_qubits"]

    for k in range(cutoff):
        ket = np.zeros(dim, dtype=complex)
        ket[idx[k]] = 1.0
        assert np.isclose((ops["n"] @ ket)[idx[k]], k)
        if k > 0:
            assert np.isclose((ops["a"] @ ket)[idx[k - 1]], np.sqrt(k))


@pytest.mark.parametrize("encoding", be.ENCODINGS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_pauli_ops_match_matrices_on_encoded_subspace(encoding, cutoff):
    """`SparsePauliOp` constructions agree with the reference matrices.

    Only *on the encoded subspace* — for unary the operators are deliberately
    extended differently outside it (see `boson_encoding`'s module docstring).
    """
    nq = be.n_qubits_per_mode(cutoff, encoding)
    qubits = list(range(nq))
    idx = be.basis_index_map(cutoff, encoding)
    ref = be.encode_ops(cutoff, encoding)

    a = be.annihilation_pauli(cutoff, encoding, qubits, nq).to_matrix()
    n = be.number_pauli(cutoff, encoding, qubits, nq).to_matrix()
    assert np.allclose(a[np.ix_(idx, idx)], ref["a"][np.ix_(idx, idx)])
    assert np.allclose(n[np.ix_(idx, idx)], ref["n"][np.ix_(idx, idx)])


@pytest.mark.parametrize("encoding", be.ENCODINGS)
def test_unary_stays_local(encoding):
    """Unary `a + a^dag` must stay at the analytic 2*(cutoff-1) term count.

    A regression to matrix-based decomposition would silently balloon this to
    896 terms at cutoff 8 and inflate every unary gate count in the study.
    """
    cutoff = 8
    nq = be.n_qubits_per_mode(cutoff, encoding)
    op = be.hermitian(be.annihilation_pauli(cutoff, encoding, list(range(nq)), nq))
    expected = {"unary": 2 * (cutoff - 1), "binary": 12, "gray": 12}[encoding]
    assert len(op) == expected


@pytest.mark.parametrize("encoding", be.ENCODINGS)
def test_pad_places_operators_on_the_right_qubits(encoding):
    """A mode's operator must act as identity on every other mode's qubits."""
    cutoff = 4
    nq = be.n_qubits_per_mode(cutoff, encoding)
    total = 2 * nq
    n1 = be.number_pauli(cutoff, encoding, list(range(nq, total)), total)
    expected = np.kron(
        be.encode_ops(cutoff, encoding)["n"], np.eye(1 << nq)
    )  # qiskit is little-endian: mode 1 is the *high* tensor factor
    idx = be.basis_index_map(cutoff, encoding)
    keep = np.array([hi * (1 << nq) + lo for hi in idx for lo in idx])
    assert np.allclose(n1.to_matrix()[np.ix_(keep, keep)], expected[np.ix_(keep, keep)])


# ---------------------------------------------------------------------------
# JCH DV circuit
# ---------------------------------------------------------------------------


def test_jch_terms_commute_with_total_excitation():
    """Every JCH term conserves `N = sum_i n_i + sum_i sigma^+sigma^-_i`.

    This is the same Trotter-error-independent invariant that
    `hyqbench_benchmarks/test_jch_simulation.py` checks on the CV-DV side.
    """
    layout = JchDvLayout(n_sites=3, cutoff=4, encoding="binary")
    nqb = layout.n_qubits
    number = SparsePauliOp.sum(
        [be.number_pauli(4, "binary", layout.mode_qubits(i), nqb) for i in range(3)]
        + [be.tls_excitation_pauli(layout.tls_qubit(i), nqb) for i in range(3)]
    ).simplify(1e-12)

    for name, term in jch_dv_terms(layout, 1.0, 1.0, 1.0, 1.0):
        commutator = (term @ number - number @ term).simplify(1e-9)
        assert np.allclose(commutator.coeffs, 0), f"{name} does not conserve N"


@pytest.mark.parametrize("encoding", ["binary", "gray"])
def test_exact_synthesis_conserves_excitation_number(encoding):
    """With exact per-term synthesis, the DV trajectory conserves `N` exactly.

    Same invariant `hyqbench_benchmarks/test_jch_simulation.py` checks on the
    CV-DV side. It holds here because each Trotter *term group* commutes with
    `N` (see `test_jch_terms_commute_with_total_excitation`), so exponentiating
    each group exactly cannot move population between excitation sectors.
    """
    result = run_jch_dv(
        n_sites=2,
        cutoff=4,
        encoding=encoding,
        tau=0.1,
        n_steps=5,
        synthesis="exact",
        initial_photons=2,
    )
    assert np.allclose(result["total_excitation"], 2.0, atol=1e-9)


@pytest.mark.parametrize("encoding", ["binary", "gray"])
def test_lie_trotter_breaks_excitation_number(encoding):
    """Pauli-Trotterized synthesis *violates* excitation-number conservation.

    This is a real qualitative cost of the DV approach, not a numerical
    artifact, and it is worth pinning down. The hopping and Jaynes-Cummings
    groups conserve `N` as a whole, but the individual Pauli strings they
    decompose into do not, so splitting them (`LieTrotter`) leaks population
    across excitation sectors. The CV-DV circuit never pays this: a
    `Beamsplitter` or `JaynesCummings` gate is a single symmetry-preserving
    native operation, so `N` is exact there *regardless* of step size.

    The violation shrinks as `reps` grows, at proportionally more gates.
    """
    drifts = []
    for reps in (1, 4):
        result = run_jch_dv(
            n_sites=2,
            cutoff=4,
            encoding=encoding,
            tau=0.1,
            n_steps=5,
            synthesis="lie_trotter",
            reps=reps,
            initial_photons=2,
        )
        drifts.append(np.abs(result["total_excitation"] - 2.0).max())

    assert drifts[0] > 1e-6, "expected LieTrotter to break N conservation"
    assert drifts[1] < drifts[0], "more reps should reduce the violation"


def test_unary_lie_trotter_preserves_excitation_number():
    """Unary is the exception: Lie-Trotter does *not* break `N` there.

    In one-hot encoding `a = sum_k sqrt(k+1) sigma^+_k sigma^-_{k+1}`, so the
    hopping and Jaynes-Cummings terms decompose into XX+YY-type strings that
    each conserve Hamming weight -- which *is* the excitation number in this
    encoding. Every individual Pauli string commutes with `N`, so splitting
    them cannot leak population between excitation sectors.

    Binary and Gray have no such property (see the test above), which makes
    this a genuine three-way tradeoff rather than a CV-DV-versus-DV one: unary
    buys the symmetry back by spending `cutoff` qubits per mode instead of
    `log2(cutoff)`.
    """
    result = run_jch_dv(
        n_sites=2,
        cutoff=4,
        encoding="unary",
        tau=0.1,
        n_steps=10,
        synthesis="lie_trotter",
        reps=1,
        initial_photons=2,
    )
    assert np.allclose(result["total_excitation"], 2.0, atol=1e-9)


@pytest.mark.parametrize("encoding", be.ENCODINGS)
def test_fountain_matches_chain(encoding):
    """The cheaper CNOT structure must be the *same circuit*, not an approximation.

    `cx_structure` only changes how each `exp(-i t P)` is realized as a CNOT
    ladder, so chain and fountain implement identical unitaries and the
    trajectories must agree to machine precision. This is what licenses the
    notebook to report the fountain counts as an optimization rather than as a
    second approximation on top of Lie-Trotter -- roughly half the CNOTs for
    exactly the same answer. If a future Qiskit changed that, every "optimized"
    column in the study would silently become a different physics run.
    """
    common = {
        "n_sites": 2,
        "cutoff": 4,
        "encoding": encoding,
        "omega_c": 2.0,
        "omega_tls": 2.0,
        "kappa": 1.0,
        "eta": 0.5,
        "tau": 0.1,
        "n_steps": 8,
        "initial_photons": 2,
    }
    chain = run_jch_dv(**common, cx_structure="chain")
    fountain = run_jch_dv(**common, cx_structure="fountain")
    assert np.allclose(chain["photons"], fountain["photons"], atol=1e-12)
    assert np.allclose(chain["total_excitation"], fountain["total_excitation"], atol=1e-12)


@pytest.mark.parametrize("encoding", be.ENCODINGS)
def test_analytic_cost_is_an_upper_bound(encoding):
    """`pauli_evolution_cost` must never under-count the transpiled circuit.

    The cutoff sweep in `jch_resource_comparison.ipynb` uses the analytic sum
    where transpiling is impractical (cutoff 64 is ~2.5M CNOTs), so what has to
    hold is the *direction* of the error: the sum is blind to cancellation
    between neighbouring CNOT ladders, so it over-counts, and the notebook
    presents it as an upper bound. A regression that made it under-count would
    turn that row into an understatement of the DV cost -- the one error this
    study cannot afford, since it argues the DV cost is high.

    The magnitude is deliberately not pinned: it ranges from under 1% to 2x
    depending on cutoff and encoding, and the notebook measures it live rather
    than quoting a remembered number.
    """
    from cvdv_vs_dv.jch_dv import build_jch_dv_trotter
    from cvdv_vs_dv.resources import count_dv, jch_step_cost_analytic

    layout = JchDvLayout(n_sites=2, cutoff=4, encoding=encoding)
    analytic = jch_step_cost_analytic(jch_dv_terms(layout, 2.0, 2.0, 1.0, 0.5))
    qc, _ = build_jch_dv_trotter(
        n_sites=2,
        cutoff=4,
        encoding=encoding,
        omega_c=2.0,
        omega_tls=2.0,
        kappa=1.0,
        eta=0.5,
        n_steps=1,
    )
    assert analytic["n_two_qubit"] >= count_dv(qc)["n_two_qubit"]


def test_fountain_is_cheaper_than_chain():
    """The optimization has to actually pay, or the extra column is noise.

    Pinned loosely (a 25% floor against a measured ~57%) so a transpiler
    version bump does not fail the suite for a few percent, while a regression
    that made the two structures equivalent would still be caught.
    """
    from cvdv_vs_dv.jch_dv import build_jch_dv_trotter
    from cvdv_vs_dv.resources import count_dv

    counts = {}
    for structure in ("chain", "fountain"):
        qc, _ = build_jch_dv_trotter(
            n_sites=3, cutoff=4, encoding="binary", n_steps=1, cx_structure=structure
        )
        counts[structure] = count_dv(qc)["n_two_qubit"]
    assert counts["fountain"] < 0.75 * counts["chain"]


@pytest.mark.slow
def test_jch_dv_matches_cvdv_trajectory():
    """DV and CV-DV must produce the same `<n_i>(t)` for the same Hamiltonian.

    This is the load-bearing check of the JCH study: only if the two stacks
    agree on the physics is a resource comparison between them meaningful.
    `synthesis="exact"` is used so both carry *identical* Trotter error --
    any residual difference would be an encoding bug, not a splitting artifact.
    """
    pennylane = pytest.importorskip("pennylane")
    import hybridlane as hqml
    from hyqbench_benchmarks.jch_simulation import jch_evolve

    pennylane.decomposition.enable_graph()
    n_sites, cutoff, tau, n_steps = 2, 4, 0.1, 6
    omega, kappa, eta = 2.0, 1.0, 0.5

    dv = run_jch_dv(
        n_sites=n_sites,
        cutoff=cutoff,
        encoding="binary",
        omega_c=omega,
        omega_tls=omega,
        kappa=kappa,
        eta=eta,
        tau=tau,
        n_steps=n_steps,
        synthesis="exact",
        initial_photons=2,
    )

    modes = [f"m{i}" for i in range(n_sites)]
    qubits = [f"q{i}" for i in range(n_sites)]
    dev = pennylane.device("default.hybrid", fock_level=cutoff)

    @pennylane.qnode(dev)
    def circuit(k):
        hqml.FockState(2, wires=["q0", "m0"])
        for m in modes[1:]:
            hqml.Rotation(0.0, wires=m)
        for q in qubits:
            pennylane.RZ(0.0, wires=q)
        # sandia/hyqbench convention: omega_tls * sigma^+sigma^- == -(omega_q/2) Z
        jch_evolve(kappa, omega, -omega, eta, tau, k, modes, qubits)
        return [hqml.expval(hqml.N(m)) for m in modes]

    cvdv = np.array([np.real(np.array(circuit(k))) for k in range(n_steps + 1)])
    assert np.allclose(cvdv, dv["photons"], atol=1e-8)


# ---------------------------------------------------------------------------
# Knapsack QUBO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem", ["knapsack3", "knapsack4a", "knapsack4b"])
def test_qubo_matches_sandia_cost(problem):
    """The DV QUBO/Ising reproduces `sandia.ecd_vqe_sandia.knapsack_cost`."""
    kd.verify_against_sandia(problem)


def test_knapsack4b_optimum():
    """knapsack4b's documented optimum, reached by brute force."""
    bf = kd.brute_force("knapsack4b")
    assert bf["best_cost"] == -9.0
    assert bf["degeneracy"] == 1
    # x = [1,0,0,1] (items 0 and 3), slack = 1 -> y = [1,0,0]
    assert bf["optimal_assignments"][0].tolist() == [1, 0, 0, 1, 1, 0, 0]


def test_qaoa_layer_has_published_cnot_count():
    """One QAOA layer on knapsack4b costs 42 CNOTs, matching arXiv:2501.11735."""
    from cvdv_vs_dv.resources import count_dv

    qubo, offset = kd.qubo_matrix("knapsack4b")
    ising, _ = kd.qubo_to_ising(qubo, offset)
    n_zz = sum(1 for p in ising.paulis if str(p).count("Z") == 2)
    assert n_zz == 21  # 21 ZZ couplings -> 2 CNOTs each
    counts = count_dv(kd.qaoa_ansatz(ising, p=1), transpile_circuit=False)
    assert counts["gate_counts"]["rzz"] == 21


def test_hardware_efficient_parameter_count():
    ansatz = kd.hardware_efficient_ansatz(7, n_layers=3)
    assert ansatz.num_parameters == 2 * 7 * 4


def test_golden_angle_init_is_deterministic_and_nonzero():
    a = kd.golden_angle_init(56)
    assert np.array_equal(a, kd.golden_angle_init(56))
    assert np.all(np.abs(a) > 1e-6)
