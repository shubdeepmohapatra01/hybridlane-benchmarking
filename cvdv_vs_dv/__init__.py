# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""CV-DV vs. pure-DV resource comparison studies.

This package builds the *pure discrete-variable* baselines that the CV-DV
benchmarks in `hyqbench_benchmarks/` and `sandia/` are compared against, so
that the "CV-DV is cheaper" claim becomes a measured table of qubit / gate /
depth / parameter counts rather than an assertion.

Two studies:

- **JCH Hamiltonian simulation** — `boson_encoding.py` + `jch_dv.py` map each
  bosonic mode onto qubits (binary / Gray / unary), Trotterize the same
  Hamiltonian with the same term ordering as
  `hyqbench_benchmarks.jch_simulation`, and count transpiled resources.
- **Binary knapsack VQE** — `knapsack_dv.py` solves the *same* QUBO as
  `sandia.ecd_vqe_sandia`'s `knapsack4b` instance with a hardware-efficient
  qubit ansatz and with QAOA.

`resources.py` provides a single resource-counting interface for both the
PennyLane/hybridlane and the Qiskit sides.

The notebooks in this folder are the study itself; the modules here exist so
that the notebooks stay readable and the numbers stay reproducible.
"""
