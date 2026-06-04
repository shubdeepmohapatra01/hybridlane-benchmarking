# Template Examples

This folder contains runnable Jupyter notebooks that demonstrate the built-in
circuit templates in `hybridlane.templates`. Each notebook is self-contained and
can be run from this directory after installing hybridlane with the `[bq]` extra:

```bash
pip install hybridlane[bq]
```

---

## Notebooks

### `state_transfer_test.ipynb` — CV↔DV State Transfer

Demonstrates the `StateTransferCVtoDV` and `StateTransferDVtoCV` templates, which
implement the non-Abelian QSP protocol for transferring quantum states between a
qumode (CV register) and n qubits (DV register).

**Covers:**
- Circuit drawing at the top-level and decomposed views
- Vacuum state transfer: CV → DV expectation values and measurement distribution
- Fock state inputs (|0⟩, |1⟩, |2⟩)
- Compilation to Sandia Qscout native gates and Jaqal IR export

**Templates used:** `StateTransferCVtoDV`

---

### `ecd_vqe_test.ipynb` — ECD VQE for Binary Knapsack

Demonstrates the `ECDLayer` template — a variational ansatz layer built from
Echoed Conditional Displacements (ECD) and qubit rotations — applied to solve
a binary knapsack problem via VQE on a hybrid qubit + 2 qumode system.

**Wire convention:** `(qubit, qumode_primary, qumode_auxiliary)`
- Qubit encodes one binary item variable
- Primary qumode (`m0`, Fock cutoff 8) encodes three binary item variables
- Auxiliary qumode (`m1`, Fock cutoff 8) encodes slack variables for the weight constraint

**Covers:**
- Circuit drawing for depth-1 and depth-5 ansatz
- QUBO Hamiltonian construction for the knapsack problem
- Statevector simulation via Bosonic Qiskit
- BFGS optimisation with finite-difference gradients (5 random restarts)
- Convergence plot and probability distribution analysis
- Full binary measurement distribution over all 128 basis states

**Templates used:** `ECDLayer`, `random_ecd_params`

---

## Dependencies

| Package | Purpose |
|---|---|
| `hybridlane[bq]` | Core library + Bosonic Qiskit simulation device |
| `pennylane` | Quantum circuit framework |
| `scipy` | Classical optimisation (`minimize`, `approx_fprime`) |
| `matplotlib` | Plotting |
| `hybridlane[qscout]` | Optional — required only for the Qscout compilation cells |
