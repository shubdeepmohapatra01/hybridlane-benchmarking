# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Pure-DV (qubit-only) Trotterized Jaynes-Cummings-Hubbard simulation.

This is the baseline that `hyqbench_benchmarks.jch_simulation`'s CV-DV circuit
is compared against. It simulates the *same* Hamiltonian, with the *same*
first-order Trotter term ordering, so any difference in resource count is
attributable to the encoding rather than to a different algorithm.

Hamiltonian (HyQBench arXiv:2603.04398 section 5.6 convention, which this
module uses directly since the DV side has no sign ambiguity)::

    H = omega_c   * sum_i n_i
      + omega_tls * sum_i sigma^+_i sigma^-_i
      + kappa     * sum_i (a^dag_{i+1} a_i + a^dag_i a_{i+1})
      + eta       * sum_i (a_i sigma^+_i + a^dag_i sigma^-_i)

Note that `hyqbench_benchmarks.jch_simulation` writes the TLS term as
`(omega_q / 2) Z_i` with ground state `|0> -> Z = +1`. Since
`omega_tls sigma^+ sigma^- = (omega_tls / 2)(I - Z)`, the two agree only for
`omega_q = -omega_tls`. See `jch_cvdv_demo.ipynb`.

Register layout (little-endian, Qiskit convention): mode `i` occupies qubits
`[i*nq, (i+1)*nq)`, and TLS `i` occupies qubit `n_sites*nq + i`.

Synthesis
---------
Each Trotter term is emitted as a `PauliEvolutionGate`. Two synthesis modes:

- ``"lie_trotter"`` (default) — the standard, cheapest choice. Terms *within*
  a group that do not commute (hopping, Jaynes-Cummings) get an additional
  layer of Trotter error on top of the outer first-order splitting. `reps`
  controls it. This is the cost a real DV implementation would pay, so it is
  the headline number.
- ``"exact"`` — each term group exponentiated as a matrix and synthesized as a
  unitary. Matches the CV-DV circuit's Trotter error exactly (useful for
  isolating "encoding cost" from "extra Trotter cost"), but the synthesis is
  exponential in the group's qubit count, so it is only usable for
  binary/Gray at small cutoff.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import LieTrotter, MatrixExponential

from .boson_encoding import (
    annihilation_pauli,
    basis_index_map,
    hermitian,
    n_qubits_per_mode,
    number_pauli,
    tls_excitation_pauli,
)

# HyQBench section 5.6 parameters for the 3-site benchmark.
HYQBENCH_JCH = {
    "n_sites": 3,
    "cutoff": 8,
    "omega_c": 4 * np.pi,
    "omega_tls": 4 * np.pi,
    "kappa": 1.0,
    "eta": 0.5,
    "tau": 0.1,
    "n_steps": 50,
    "initial_photons": 2,
}


class JchDvLayout:
    """Qubit register layout for an `n_sites` JCH chain under one encoding."""

    def __init__(self, n_sites: int, cutoff: int, encoding: str):
        self.n_sites = n_sites
        self.cutoff = cutoff
        self.encoding = encoding
        self.nq_mode = n_qubits_per_mode(cutoff, encoding)
        self.n_qubits = n_sites * self.nq_mode + n_sites

    def mode_qubits(self, site: int) -> list[int]:
        base = site * self.nq_mode
        return list(range(base, base + self.nq_mode))

    def tls_qubit(self, site: int) -> int:
        return self.n_sites * self.nq_mode + site

    def __repr__(self) -> str:
        return (
            f"JchDvLayout(n_sites={self.n_sites}, cutoff={self.cutoff}, "
            f"encoding={self.encoding!r}, nq_mode={self.nq_mode}, "
            f"n_qubits={self.n_qubits})"
        )


def jch_dv_terms(layout: JchDvLayout, omega_c, omega_tls, kappa, eta) -> list:
    """The four JCH term groups, in Trotter order, as `(name, SparsePauliOp)`.

    Ordering mirrors `hyqbench_benchmarks.jch_simulation.jch_trotter_step`:
    brick-wall hopping (even bonds, then odd bonds), cavity energy, TLS energy,
    Jaynes-Cummings coupling. Coefficients are the Hamiltonian coefficients;
    `PauliEvolutionGate(op, time=tau)` then implements `exp(-i tau op)`.
    """
    n, c, enc, nqb = layout.n_sites, layout.cutoff, layout.encoding, layout.n_qubits
    a = [annihilation_pauli(c, enc, layout.mode_qubits(i), nqb) for i in range(n)]
    terms = []

    # 1. Photon hopping, brick-wall: even bonds first, then odd bonds.
    for parity, name in ((0, "hop_even"), (1, "hop_odd")):
        bonds = [kappa * hermitian(a[i].adjoint() @ a[i + 1]) for i in range(parity, n - 1, 2)]
        if bonds:
            terms.append((name, sum(bonds[1:], bonds[0]).simplify(1e-12)))

    # 2. Local cavity energy (all-Z, mutually commuting: Trotter-exact).
    cav = [omega_c * number_pauli(c, enc, layout.mode_qubits(i), nqb) for i in range(n)]
    terms.append(("cavity", sum(cav[1:], cav[0]).simplify(1e-12)))

    # 3. Local TLS energy (all-Z, mutually commuting: Trotter-exact).
    tls = [omega_tls * tls_excitation_pauli(layout.tls_qubit(i), nqb) for i in range(n)]
    terms.append(("tls", sum(tls[1:], tls[0]).simplify(1e-12)))

    # 4. Jaynes-Cummings coupling, eta * (a_i sigma^+_i + h.c.).
    from .boson_encoding import _sigma_plus  # local: internal helper

    jc = [eta * hermitian(a[i] @ _sigma_plus(layout.tls_qubit(i), nqb)) for i in range(n)]
    terms.append(("jaynes_cummings", sum(jc[1:], jc[0]).simplify(1e-12)))

    return terms


def build_jch_dv_trotter(
    n_sites: int = 3,
    cutoff: int = 8,
    encoding: str = "binary",
    omega_c: float = HYQBENCH_JCH["omega_c"],
    omega_tls: float = HYQBENCH_JCH["omega_tls"],
    kappa: float = 1.0,
    eta: float = 0.5,
    tau: float = 0.1,
    n_steps: int = 1,
    synthesis: str = "lie_trotter",
    reps: int = 1,
    initial_photons: int | None = None,
    cx_structure: str = "chain",
) -> tuple[QuantumCircuit, JchDvLayout]:
    """Build the pure-DV JCH Trotter circuit.

    If `initial_photons` is given, mode 0 is prepared in that Fock level with
    X gates on the encoded bitstring; otherwise the circuit starts from
    all-zeros and the caller prepares the state.

    `cx_structure` selects the CNOT ladder Qiskit uses to realize each
    `exp(-i t P)`: ``"chain"`` walks the support next-neighbour to next-neighbour,
    ``"fountain"`` fans every leg into one target. Both realize the *same*
    unitary, so this is a pure compilation choice with no effect on the physics
    — but the fountain shape shares CNOTs between the many overlapping Pauli
    strings of a JCH term group far better, and `optimization_level=3` cancels
    roughly half of them (see `test_fountain_matches_chain`). ``"chain"`` is the
    default so the unoptimized baseline stays the reported default.

    Returns `(circuit, layout)`.
    """
    layout = JchDvLayout(n_sites, cutoff, encoding)
    qc = QuantumCircuit(layout.n_qubits, name=f"jch_{encoding}_c{cutoff}")

    if initial_photons is not None:
        prepare_fock(qc, layout, site=0, level=initial_photons)

    terms = jch_dv_terms(layout, omega_c, omega_tls, kappa, eta)
    synth = (
        MatrixExponential()
        if synthesis == "exact"
        else LieTrotter(reps=reps, cx_structure=cx_structure)
    )
    for _ in range(n_steps):
        for _name, op in terms:
            qc.append(PauliEvolutionGate(op, time=tau, synthesis=synth), range(layout.n_qubits))
    return qc, layout


def prepare_fock(qc: QuantumCircuit, layout: JchDvLayout, site: int, level: int) -> None:
    """Set mode `site` to Fock level `level` with X gates (from all-zeros)."""
    if not 0 <= level < layout.cutoff:
        raise ValueError(f"level {level} outside cutoff {layout.cutoff}")
    index = int(basis_index_map(layout.cutoff, layout.encoding)[level])
    for bit, qubit in enumerate(layout.mode_qubits(site)):
        if (index >> bit) & 1:
            qc.x(qubit)


def _propagator(hamiltonian: np.ndarray, tau: float) -> np.ndarray:
    """`exp(-i tau H)` for Hermitian `H`, via eigendecomposition."""
    evals, evecs = np.linalg.eigh(hamiltonian)
    return (evecs * np.exp(-1j * tau * evals)) @ evecs.conj().T


def run_jch_dv(
    n_sites: int = 3,
    cutoff: int = 8,
    encoding: str = "binary",
    omega_c: float = HYQBENCH_JCH["omega_c"],
    omega_tls: float = HYQBENCH_JCH["omega_tls"],
    kappa: float = 1.0,
    eta: float = 0.5,
    tau: float = 0.1,
    n_steps: int = 50,
    synthesis: str = "lie_trotter",
    reps: int = 1,
    initial_photons: int = 2,
    cx_structure: str = "chain",
) -> dict:
    """Evolve step by step, recording `<n_i>` and `<sigma^+sigma^->_i` per step.

    Unlike the CV-DV `jch_trajectory` (which re-simulates from scratch at each
    step), this advances a single `Statevector` one Trotter step at a time —
    same trajectory, `n_steps` times less work.

    Returns a dict with `times`, `photons` (shape `(n_steps+1, n_sites)`),
    `tls` (same shape), `total_excitation`, and `layout`.
    """
    layout = JchDvLayout(n_sites, cutoff, encoding)
    nqb = layout.n_qubits

    n_obs = [number_pauli(cutoff, encoding, layout.mode_qubits(i), nqb) for i in range(n_sites)]
    e_obs = [tls_excitation_pauli(layout.tls_qubit(i), nqb) for i in range(n_sites)]

    prep = QuantumCircuit(nqb)
    prepare_fock(prep, layout, site=0, level=initial_photons)
    psi = np.asarray(Statevector.from_int(0, 2**nqb).evolve(prep).data)

    if synthesis == "exact":
        # Going through Qiskit's gate machinery here is unusably slow: it
        # re-runs a dense 2^n matrix exponential every time a gate definition is
        # expanded. Since each term group is Hermitian, diagonalize once and
        # build exp(-i tau H) from the eigendecomposition, then apply the
        # per-term propagators as plain matrix-vector products.
        propagators = [
            _propagator(op.to_matrix(), tau)
            for _, op in jch_dv_terms(layout, omega_c, omega_tls, kappa, eta)
        ]

        def advance(vec):
            for u in propagators:
                vec = u @ vec
            return vec
    else:
        step, _ = build_jch_dv_trotter(
            n_sites=n_sites,
            cutoff=cutoff,
            encoding=encoding,
            omega_c=omega_c,
            omega_tls=omega_tls,
            kappa=kappa,
            eta=eta,
            tau=tau,
            n_steps=1,
            synthesis=synthesis,
            reps=reps,
            cx_structure=cx_structure,
        )

        def advance(vec):
            return np.asarray(Statevector(vec).evolve(step).data)

    n_mats = [o.to_matrix() for o in n_obs]
    e_mats = [o.to_matrix() for o in e_obs]

    photons, tls = [], []
    for _ in range(n_steps + 1):
        photons.append([float(np.real(np.vdot(psi, m @ psi))) for m in n_mats])
        tls.append([float(np.real(np.vdot(psi, m @ psi))) for m in e_mats])
        psi = advance(psi)

    photons = np.array(photons)
    tls = np.array(tls)
    return {
        "times": np.arange(n_steps + 1) * tau,
        "photons": photons,
        "tls": tls,
        "total_excitation": photons.sum(axis=1) + tls.sum(axis=1),
        "layout": layout,
    }


# ---------------------------------------------------------------------------
# Parallel sweeps
# ---------------------------------------------------------------------------
#
# The sweeps these notebooks run (trajectories at several step sizes, gate
# counts at several cutoffs) are independent jobs, and running them serially
# wastes most of a multi-core machine. These helpers live at module level
# rather than in the notebooks on purpose: macOS spawns fresh interpreters for
# worker processes, and a function defined in a notebook cell cannot be pickled
# across that boundary, so a `ProcessPoolExecutor` over notebook-local closures
# fails at submit time.


def _run_jch_dv_job(kwargs: dict) -> dict:
    """Worker: run one trajectory and drop the unpicklable layout object."""
    result = run_jch_dv(**kwargs)
    result.pop("layout", None)
    return result


def sweep_trajectories(jobs: list[dict], max_workers: int | None = None) -> list[dict]:
    """Run `run_jch_dv` over a list of kwarg dicts, in parallel.

    Results come back in the same order as `jobs`. `max_workers` defaults to
    `min(len(jobs), cpu_count() - 2)`, leaving headroom so the machine stays
    usable.
    """
    return _parallel_map(_run_jch_dv_job, jobs, max_workers)


def _count_jch_dv_job(kwargs: dict) -> dict:
    """Worker: build one Trotter step and return its transpiled resource counts."""
    from .resources import count_dv

    qc, layout = build_jch_dv_trotter(**{**kwargs, "n_steps": kwargs.get("n_steps", 1)})
    counts = count_dv(qc)
    counts.pop("transpiled", None)  # a QuantumCircuit is large to ship back
    counts["nq_mode"] = layout.nq_mode
    return counts


def sweep_resource_counts(jobs: list[dict], max_workers: int | None = None) -> list[dict]:
    """Build and transpile JCH circuits over a list of kwarg dicts, in parallel."""
    return _parallel_map(_count_jch_dv_job, jobs, max_workers)


def _parallel_map(fn, jobs: list, max_workers: int | None):
    """`executor.map` with a sensible worker count, falling back to serial.

    Falls back when there is only one job (process startup would dominate) or
    when a pool cannot be created, so callers never have to branch.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    if len(jobs) <= 1:
        return [fn(job) for job in jobs]
    if max_workers is None:
        max_workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    # A pool of one buys nothing and, on GPU, actively breaks: the executor
    # forks on Linux and a forked CUDA context is unusable, so the pool dies
    # with BrokenProcessPool. Callers pass max_workers=1 on GPU precisely
    # because there is one card to share.
    if max_workers <= 1:
        return [fn(job) for job in jobs]
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(fn, jobs))
    except (OSError, RuntimeError):
        return [fn(job) for job in jobs]
