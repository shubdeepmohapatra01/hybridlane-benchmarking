# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Boson-to-qubit encodings for the pure-DV baselines.

A CV-DV device carries a bosonic mode natively. A qubit-only device must
truncate the mode to `cutoff` Fock levels and then *encode* those levels into
qubit basis states. Three standard encodings, spanning the qubit-count vs.
operator-locality tradeoff (Sawaya et al., npj Quantum Information 6, 49
(2020), "Resource-efficient digital quantum simulation of d-level systems"):

- ``"binary"`` — `|k> -> |bin(k)>`, `ceil(log2(cutoff))` qubits per mode.
  Fewest qubits; ladder operators become dense sums of high-weight Pauli
  strings.
- ``"gray"`` — `|k> -> |gray(k)>`, same qubit count. Adjacent Fock levels
  differ in a single bit, and the ladder operators only connect adjacent
  levels, so `a` decomposes into lower-weight Pauli strings than plain binary.
- ``"unary"`` — `|k> -> |0...010...0>` (one-hot), `cutoff` qubits per mode.
  Most qubits, but `a` becomes a sum of nearest-neighbour two-qubit terms, so
  circuits are shallow.

Operators are returned as Qiskit `SparsePauliOp`s on a full multi-mode
register, ready to hand to `PauliEvolutionGate`.

How they are built, and why it differs by encoding
--------------------------------------------------
For binary/Gray the qubit space is (for power-of-two `cutoff`) *exactly* the
truncated mode space, so the operator is built by permuting the Fock-basis
matrix and calling `SparsePauliOp.from_operator` — correct by construction,
with no hand-derived Pauli algebra to get wrong.

For unary that route is wrong in a way that silently inflates every gate
count: the one-hot states span only `cutoff` of the `2**cutoff` qubit basis
states, and `from_operator` would faithfully encode "acts as zero everywhere
else" — a projector onto the one-hot subspace, costing 896 Pauli terms for
`a + a^dag` at cutoff 8 instead of the correct 14. Outside the one-hot
subspace the operator is unconstrained, and the standard choice is the local
form `a = sum_k sqrt(k+1) sigma^+_k sigma^-_{k+1}`, which conserves Hamming
weight and therefore never leaves the encoded subspace. Unary operators are
built from that expression directly. `test_boson_encoding.py` checks both
routes agree *on the encoded subspace*.

Qubit ordering follows Qiskit's little-endian convention throughout: bit 0 of
the encoded integer is qubit 0 of the mode's register.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from qiskit.quantum_info import SparsePauliOp

ENCODINGS = ("binary", "gray", "unary")


# ---------------------------------------------------------------------------
# Level <-> qubit-basis-state mapping
# ---------------------------------------------------------------------------


def n_qubits_per_mode(cutoff: int, encoding: str) -> int:
    """Number of qubits needed to hold one mode truncated to `cutoff` levels."""
    _validate(cutoff, encoding)
    if encoding == "unary":
        return cutoff
    return max(1, math.ceil(math.log2(cutoff)))


def basis_index_map(cutoff: int, encoding: str) -> np.ndarray:
    """Map Fock level `k` to its qubit computational-basis index.

    Returns an integer array `m` of length `cutoff` where `m[k]` is the index
    (in the `2**n_qubits`-dimensional qubit space) representing `|k>`.

    >>> basis_index_map(4, "gray").tolist()
    [0, 1, 3, 2]
    >>> basis_index_map(3, "unary").tolist()
    [1, 2, 4]
    """
    _validate(cutoff, encoding)
    levels = np.arange(cutoff, dtype=np.int64)
    if encoding == "binary":
        return levels
    if encoding == "gray":
        return levels ^ (levels >> 1)
    return (1 << levels).astype(np.int64)  # unary / one-hot


def embed(matrix: np.ndarray, cutoff: int, encoding: str) -> np.ndarray:
    """Embed a `cutoff x cutoff` Fock-basis operator into the qubit space.

    The operator acts as zero outside the encoded levels. For binary/Gray at
    power-of-two `cutoff` there is nothing outside, so this is exact. For unary
    it is only used as a *reference* in tests — see the module docstring.
    """
    nq = n_qubits_per_mode(cutoff, encoding)
    idx = basis_index_map(cutoff, encoding)
    out = np.zeros((1 << nq, 1 << nq), dtype=complex)
    out[np.ix_(idx, idx)] = matrix
    return out


@functools.cache
def encode_ops(cutoff: int, encoding: str) -> dict:
    """Truncated ladder/number operators for one mode, as dense matrices.

    Returns ``{"a", "adag", "n", "n_qubits", "index_map"}``. The matrices act
    as zero outside the encoded levels; they are the reference used to check
    the `SparsePauliOp` constructions below.

    >>> ops = encode_ops(4, "binary")
    >>> ops["n_qubits"]
    2
    >>> bool(np.allclose(np.diag(ops["n"]).real, [0, 1, 2, 3]))
    True
    """
    _validate(cutoff, encoding)
    a_fock = np.diag(np.sqrt(np.arange(1, cutoff, dtype=float)), 1)
    # Build n from the Fock diagonal rather than adag @ a: the latter loses the
    # top level, since a|cutoff-1> leaves the window and truncates to 0.
    n_fock = np.diag(np.arange(cutoff, dtype=float))
    a = embed(a_fock, cutoff, encoding)
    return {
        "a": a,
        "adag": a.conj().T,
        "n": embed(n_fock, cutoff, encoding),
        "n_qubits": n_qubits_per_mode(cutoff, encoding),
        "index_map": basis_index_map(cutoff, encoding),
    }


# ---------------------------------------------------------------------------
# Pauli operators on a full register
# ---------------------------------------------------------------------------


def annihilation_pauli(
    cutoff: int, encoding: str, qubits: list[int], total_qubits: int
) -> SparsePauliOp:
    """`a` for one mode, as a (non-Hermitian) `SparsePauliOp` on the register.

    `qubits` lists the mode's qubits in little-endian order (for unary,
    `qubits[k]` holds Fock level `k`).
    """
    _validate(cutoff, encoding)
    if len(qubits) != n_qubits_per_mode(cutoff, encoding):
        raise ValueError(
            f"{encoding} encoding of cutoff {cutoff} needs "
            f"{n_qubits_per_mode(cutoff, encoding)} qubits, got {len(qubits)}"
        )

    if encoding == "unary":
        # a = sum_k sqrt(k+1) |k><k+1| = sum_k sqrt(k+1) sigma^+_k sigma^-_{k+1}
        terms = []
        for k in range(cutoff - 1):
            terms.append(
                math.sqrt(k + 1)
                * _sigma_plus(qubits[k], total_qubits)
                @ _sigma_minus(qubits[k + 1], total_qubits)
            )
        return sum(terms[1:], terms[0]).simplify(1e-12)

    local = SparsePauliOp.from_operator(encode_ops(cutoff, encoding)["a"])
    return _pad(local, qubits, total_qubits)


def number_pauli(cutoff: int, encoding: str, qubits: list[int], total_qubits: int) -> SparsePauliOp:
    """The photon-number operator `n` for one mode, as a `SparsePauliOp`."""
    _validate(cutoff, encoding)
    if encoding == "unary":
        # n = sum_k k |1><1|_k = sum_k k (I - Z_k)/2
        terms = [
            k * (_identity(total_qubits) - _single(qubits[k], "Z", total_qubits)) * 0.5
            for k in range(1, cutoff)
        ]
        return sum(terms[1:], terms[0]).simplify(1e-12)
    local = SparsePauliOp.from_operator(encode_ops(cutoff, encoding)["n"])
    return _pad(local, qubits, total_qubits)


def tls_excitation_pauli(qubit: int, total_qubits: int) -> SparsePauliOp:
    """`sigma^+ sigma^- = |e><e| = (I - Z)/2` for a two-level system."""
    return ((_identity(total_qubits) - _single(qubit, "Z", total_qubits)) * 0.5).simplify(1e-12)


def hermitian(op: SparsePauliOp) -> SparsePauliOp:
    """`op + op^dag`, the combination the JCH hopping and JC terms take."""
    return (op + op.adjoint()).simplify(1e-12)


# ---------------------------------------------------------------------------
# SparsePauliOp helpers
# ---------------------------------------------------------------------------


def _identity(total_qubits: int) -> SparsePauliOp:
    return SparsePauliOp("I" * total_qubits, [1.0])


def _single(qubit: int, pauli: str, total_qubits: int) -> SparsePauliOp:
    label = ["I"] * total_qubits
    label[qubit] = pauli
    # Qiskit Pauli labels are most-significant-qubit first.
    return SparsePauliOp("".join(reversed(label)), [1.0])


def _sigma_plus(qubit: int, total_qubits: int) -> SparsePauliOp:
    """`|1><0| = (X - iY)/2`."""
    return SparsePauliOp.sum(
        [
            0.5 * _single(qubit, "X", total_qubits),
            -0.5j * _single(qubit, "Y", total_qubits),
        ]
    )


def _sigma_minus(qubit: int, total_qubits: int) -> SparsePauliOp:
    """`|0><1| = (X + iY)/2`."""
    return _sigma_plus(qubit, total_qubits).adjoint()


def _pad(op: SparsePauliOp, qubits: list[int], total_qubits: int) -> SparsePauliOp:
    """Widen a local `SparsePauliOp` onto `qubits` of a `total_qubits` register."""
    labels, coeffs = [], []
    for pauli, coeff in zip(op.paulis, op.coeffs, strict=True):
        local = str(pauli)[::-1]  # to little-endian
        full = ["I"] * total_qubits
        for local_idx, wire in enumerate(qubits):
            full[wire] = local[local_idx]
        labels.append("".join(reversed(full)))
        coeffs.append(coeff)
    return SparsePauliOp(labels, np.asarray(coeffs)).simplify(1e-12)


def _validate(cutoff: int, encoding: str) -> None:
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    if cutoff < 2:
        raise ValueError(f"cutoff must be >= 2, got {cutoff}")
