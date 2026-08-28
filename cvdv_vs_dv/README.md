<!--
SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
SPDX-License-Identifier: BSD-2-Clause
-->
# CV-DV vs. pure-DV: quantifying the advantage

This folder answers one question for two benchmarks: **how much does a hybrid
CV-DV device actually save over a qubit-only device, on a problem where the
CV-DV encoding is natural?**

The repo already had both CV-DV benchmarks working
(`hyqbench_benchmarks/jch_simulation.py`, `sandia/ecd_vqe_sandia*.py`) and the
HyQBench paper (arXiv:2603.04398) asserts an advantage, but the pure-DV
baseline it compares against is not reproducible from this repository. This
folder builds that baseline from scratch, verifies it computes the *same
answer*, and turns the assertion into measured numbers.

## The two studies

| | benchmark | CV-DV side | DV baseline |
|---|---|---|---|
| **1** | Jaynes-Cummings-Hubbard time evolution, 3 cavities + 3 TLS | 3 qumodes + 3 qubits | boson-to-qubit encodings (binary / Gray / unary) + Pauli Trotterization |
| **2** | Binary knapsack `knapsack4b`, 7 binary variables | 1 qubit + 2 qumodes, ECD-VQE | hardware-efficient VQE and QAOA on 7 qubits |

## Notebooks

Run them in this order; each comparison notebook assumes its demo.

1. **`jch_cvdv_demo.ipynb`** — the JCH benchmark on `hybridlane`. HyQBench's
   3-site parameters, 50 Trotter steps, photon-occupation dynamics, and the
   conservation check. Validated against an independent QuTiP integration and
   cross-checked on two backends.
2. **`jch_resource_comparison.ipynb`** — the pure-DV baseline for the same
   simulation, in six sections that build on each other: the CV-DV circuit (1),
   the three qubit-only encodings (2), correctness against QuTiP (3),
   conservation of the excitation number (4), resources per step / per
   simulation / per cutoff (5), and one table over every setting (6).
   Correctness comes before cost on purpose. **Sections 5 and 6 report only the
   optimized compilation on both sides**; section 7 is an appendix holding the
   unoptimized baseline, the per-pass evidence, and the analytic cutoff-64
   estimate — nothing in 1-6 depends on reading it.
3. **`ecd_vqe_cvdv_demo.ipynb`** — ECD-VQE on `knapsack4b`, loading the
   converged parameters from `sandia/`. Decoding, solution distribution, and
   the hardware displacement budget.
4. **`vqe_resource_comparison.ipynb`** — the same QUBO solved with qubit-only
   ansaetze, at matched parameter count and grown to convergence.

## Modules

| file | contents |
|---|---|
| `boson_encoding.py` | Fock-level to qubit-state maps and `SparsePauliOp` ladder/number operators for the binary, Gray and unary encodings |
| `jch_dv.py` | pure-DV JCH Trotter circuit builder and statevector runner, with `cx_structure` selecting the baseline or optimized compilation |
| `knapsack_dv.py` | knapsack QUBO to Ising, hardware-efficient and QAOA ansaetze, deterministic VQE runner |
| `resources.py` | one resource-counting interface across the PennyLane and Qiskit stacks, `optimize_cvdv` for the CV-DV optimization passes, and `html_table` for rendering the notebook tables |
| `test_cvdv_vs_dv.py` | correctness tests (see below) |

Run the tests with:

```bash
uv run pytest cvdv_vs_dv/ -c pytest.toml -m "not slow"   # fast path
uv run pytest cvdv_vs_dv/ -c pytest.toml -m slow          # cross-stack check
```

## Things worth knowing before you edit this

**The DV baseline must keep matching the CV-DV physics.** This is the load-bearing
assumption of the whole study, and it is what
`test_jch_dv_matches_cvdv_trajectory` (marked `slow`) and
`test_qubo_matches_sandia_cost` protect. The knapsack QUBO is not re-derived
here — `knapsack_dv.qubo_matrix` reads the problem constants and
`knapsack_cost` straight out of `sandia.ecd_vqe_sandia`, and
`verify_against_sandia` asserts agreement over all `2^7` assignments. Keep it
that way; a re-derived copy would drift.

**The TLS sign convention.** `hyqbench_benchmarks.jch_simulation` writes the TLS
term as `(omega_q/2) Z` with ground state `|0> -> Z=+1`; HyQBench writes it as
`omega_tls sigma^+sigma^-`. Since `omega_tls sigma^+sigma^- = (omega_tls/2)(I - Z)`,
matching them needs **`omega_q = -omega_tls`**. The wrong sign raises no error
and still conserves excitation number — it silently detunes the cavity from the
TLS. `jch_cvdv_demo.ipynb` demonstrates both against the exact answer. (The
repo README already records that HyQBench's own docstring and code disagree by
a factor of two on this same term; this is the same class of bug one level up.)

**Do not decompose unary operators from their matrix.** The one-hot subspace
spans only `cutoff` of `2**cutoff` basis states, so `SparsePauliOp.from_operator`
faithfully encodes "acts as zero everywhere else" — a projector, costing 896
Pauli terms for `a + a^dag` at cutoff 8 instead of the correct 14. Unary
operators are built analytically instead; `test_unary_stays_local` pins the term
count so a regression cannot silently inflate every unary gate count in the
study.

**`qml.specs` returns a `CircuitSpecs` object, not a dict**, and it exposes no
trainable-parameter count for these circuits. `resources.count_cvdv` therefore
takes `n_parameters` explicitly — pass it when the count matters.

**One optimizer run does not characterize a variational ansatz here, and
reporting one will mislead you.** From a single deterministic start the qubit
ansatz appears to solve `knapsack4b` exactly with 2 layers and 42 parameters,
beating CV-DV on every axis. From 40 random starts, that same ansatz reaches
the optimum **48% of the time**; it needs ~12 layers (182 parameters) for 98%.
The first version of `vqe_resource_comparison.ipynb` was built on single runs
and reported the opposite conclusion. Always report the distribution over
random starts. `knapsack_dv.sweep_vqe` accepts a `seed` per job for exactly
this.

**Sweeps are parallel; the workers must live in the modules.** `sweep_vqe`,
`sweep_trajectories` and `sweep_resource_counts` map independent jobs over
processes. The worker functions are module-level on purpose: macOS spawns
fresh interpreters for workers, so a function defined in a notebook cell cannot
be pickled across that boundary and a `ProcessPoolExecutor` over notebook-local
closures fails at submit time.

**QAOA needs its own initialization.** It gets `qaoa_ramp_init` (annealing-style
linear ramp, `gamma` up / `beta` down, scaled by the mean cost coefficient), not
the golden-angle schedule the other ansaetze use. QAOA's landscape is
structured and scattered angles make it look worse than it is — which would
strawman the baseline this study criticizes.

**Do not add wire-registration padding to circuits you are counting.**
`hybridlane` raises `TypeCheckError` if a wire is never touched, which tempts
you to add `Rotation(0.0)` / `RZ(0.0)` no-ops. Those are legitimate at step 0 of
a trajectory, but inside a resource-counting circuit they inflate the reported
gate count (11 -> 17 for one JCH Trotter step).

**`cx_structure="fountain"` is worth roughly half the DV CNOT count, and it is
free.** Qiskit's default `"chain"` walks each Pauli string's support
next-neighbour to next-neighbour; `"fountain"` fans every leg into one target.
Both realize the *same unitary* — `test_fountain_matches_chain` asserts the
trajectories agree to 1e-12 — but fountain shares CNOTs between the hundreds of
overlapping strings in a JCH term group far better, and `optimization_level=3`
then cancels about half of them: binary at cutoff 8 goes 4,623 → 1,979 CNOTs
per step. Every DV number in the first version of this study was a chain count,
i.e. an over-estimate by that factor.

Report the optimized number in the main line and keep the baseline in the
section-7 appendix. The one deliberate exception is Table 5.3, the cross-check
against the published HyQBench figure: that figure is itself an unoptimized
count, so the like-for-like comparison needs the chain row and the table says
so. Never let the optimized number be described as a different approximation —
it is the same unitary.

**Do not add `single_qubit_fusion` to `optimize_cvdv`.** PennyLane's optimizers
find *nothing* on a Trotterized JCH circuit — 550 gates in, 550 out — and that
is structural, not a missing pass: consecutive gates are exponentials of
distinct non-commuting terms, so there are no inverse pairs for
`cancel_inverses` and no same-axis neighbours for `merge_rotations`. The
temptation is then to reach for `single_qubit_fusion`, which *inflates* the
count to 850 because it is a normalizer rather than a reducer: it rewrites each
`RZ` into a generic `Rot` (`RZ·RY·RZ`) with nothing on that wire to fuse with.
Table 5.6 of `jch_resource_comparison.ipynb` shows all of this measured, so the
zero is demonstrated rather than asserted.

**`pauli_evolution_cost` is an upper bound, not a ±10% estimate.** It sums
`2*(w-1)` CNOTs per weight-`w` string, blind to cancellation between
neighbouring ladders, so it over-counts the transpiled *chain* result by
+0% to +102% across cutoffs 2-32 (worst at cutoff 2 for binary/Gray and at 16
for unary) — and by a further ~2x against fountain. The notebook measures the
over-count live rather than quoting a remembered figure, and
`test_analytic_cost_is_an_upper_bound` pins the *direction* only. An earlier
version of this README claimed 7-12% agreement; that was fitted on cutoffs 4
and 8 alone and does not hold across the sweep.

**Exact synthesis needs the fast path.** Going through Qiskit's
`PauliEvolutionGate` with `MatrixExponential` re-runs a dense `2^n` matrix
exponential every time a gate definition is expanded — 16 s per step at 12
qubits, so ~13 minutes for one 50-step trajectory. `jch_dv.run_jch_dv`
diagonalizes each Hermitian term group once and applies the propagators as
matrix-vector products instead.

## Findings

Measured for the 3-site JCH benchmark at Fock cutoff 8 and for `knapsack4b`;
the notebooks carry the full tables.

- **Width.** JCH: 3 qumodes + 3 qubits vs. 12 qubits (binary/Gray) or 27
  (unary). Knapsack: 1 qubit + 2 qumodes carries 7 binary variables, vs. 7
  qubits for every DV method.
- **Gate count, at each side's best stock compilation.** One JCH Trotter step
  is 11 native CV-DV gates versus 1,979 CNOTs for the cheapest DV encoding at
  cutoff 8 (binary; Gray 1,681, unary 2,786). The DV baseline independently
  reproduces the "9 qubits, 393 CNOT, 265 U3" figure quoted in the HyQBench
  paper at cutoff 4 — this repo measures 9 qubits, 374 CNOT, 274 U through a
  different code path with no `mat2qubit`.
- **Optimization is asymmetric, and that is itself a result.** The DV side
  gives back a factor of ~2 to a single stock Qiskit flag (see below); the
  CV-DV side gives back *nothing*, because it never manufactures the redundancy
  the encoding creates. Report both columns; the optimized DV column is the
  fair comparison and CV-DV still wins it by two orders of magnitude.
- **Scaling.** The CV-DV gate count is *flat* in the Fock cutoff — a qumode
  holds all levels natively. Every DV encoding grows, as ~`d^2.6` (binary),
  `d^2.5` (Gray), `d^2.0` (unary) fitted on the *optimized* curve between
  cutoffs 16 and 32. Optimization moves the DV curves down by a factor; it does
  not bend them, which is why the advantage is not a constant factor to be
  engineered away.
- **Symmetry, and it is a three-way tradeoff.** The CV-DV circuit conserves the
  total excitation number exactly at any step size, because each Hamiltonian
  term is one symmetry-preserving native gate. Under Lie-Trotter synthesis:

  | | qubits/mode (cutoff 8) | max \|N - 2\| at r=1 |
  |---|---|---|
  | CV-DV qumode | native | 1e-14 (exact) |
  | DV unary | 8 | 2e-12 (exact) |
  | DV binary | 3 | 1.2e-4 (broken) |
  | DV Gray | 3 | 7.4e-4 (broken) |

  **Unary preserves it; binary and Gray do not.** In one-hot,
  `a = sum_k sqrt(k+1) sigma^+_k sigma^-_{k+1}`, so hopping and Jaynes-Cummings
  decompose into XX+YY strings that each conserve Hamming weight -- which *is*
  the excitation number there -- so splitting them cannot leak between sectors.
  Binary/Gray strings have no such property. So unary buys the symmetry back by
  spending `cutoff` qubits per mode instead of `log2(cutoff)`; the qumode gets
  it for free at native width. Do not state this as "DV breaks the symmetry" --
  that was an early extrapolation from binary/Gray alone and it is wrong for
  unary.

### Study 2 (knapsack VQE) is a much weaker result than study 1

Stated plainly, because it would be easy to present this as a CV-DV win and it
mostly is not one:

- **Width is the only uncontested advantage**: 7 binary variables on 3 wires
  versus 7 qubits. It is also the claim that should scale.
- **A hardware-efficient qubit ansatz is competitive or better on parameters
  and gates**, and reaches the exact optimum (−9) where CV-DV plateaus at
  −8.904. Which side wins on parameter count depends on whether you compare
  against the shallowest qubit ansatz that *can* find the optimum (DV wins,
  42 vs 56) or the shallowest that finds it *reliably* (CV-DV wins, 56 vs ~98).
  Quoting only one of those is how this comparison goes wrong.
- **Much of the published CV-DV advantage rests on QAOA being the baseline.**
  QAOA genuinely struggles here — the QUBO is fully connected, 42 CNOTs per
  layer, P(optimal) flatlines around a few percent even at p = 20 with a proper
  annealing-ramp init. But a hardware-efficient ansatz on the same 7 qubits
  does far better for far fewer gates, so comparing only against QAOA overstates
  the advantage.
- **Depth 7 is close to minimal for CV-DV, not conservative.** Re-optimizing at
  every depth shows a sharp knee: P(optimal) is 0.43 at depth 5 and 0.97 at
  depth 6. Depth 6 (48 parameters, 12 ECD gates) is the honest minimum; depth 8
  buys almost nothing.
- **Both landscapes are rugged.** Every local minimum of the DV problem is a
  computational basis state (energies −9, −5, −4, −2, −1, 0), and the CV-DV
  ansatz also fails from some random starts. The deterministic golden-angle
  initialization is doing real work on the CV-DV side.
- Validation worth keeping: synthesizing one ECD gate onto a 3-qubit
  binary-encoded mode costs **93 CNOTs**, so a depth-5 ansatz's 10 ECD gates
  come to 930 — an exact match to the figure quoted in arXiv:2501.11735.

## Caveats, stated plainly

- DV gate counts assume **all-to-all connectivity**, matching Bosonic Qiskit's
  model. A restricted coupling map would add SWAP overhead, so these numbers
  are a lower bound on the real DV cost.
- The `n_two_qubit` column counts **different physical operations** on the two
  sides: transpiled CNOTs for DV, native hybrid entangling gates (ECD,
  Jaynes-Cummings, beamsplitter) for CV-DV. Do not read it as a ratio. Each
  comparison notebook has a section that puts both on one axis by synthesizing
  a native CV-DV gate onto encoded qubits — that is an estimate, not a measured
  hardware cost.
- Everything is exact statevector simulation: no shot noise, no hardware noise.
- In the VQE study the CV-DV parameters came from 8000 JAX/Adam steps while the
  DV runs use scipy BFGS. Both use the same deterministic golden-angle
  initialization and no random restarts, and the DV layer sweep shows the trend
  continuing past the matched point — but the optimizers do differ.
