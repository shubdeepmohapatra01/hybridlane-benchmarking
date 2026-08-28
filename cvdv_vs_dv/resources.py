# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Unified resource counting across the CV-DV (PennyLane) and DV (Qiskit) stacks.

The whole point of these studies is a like-for-like table, so both sides are
reduced to the same dict of fields:

``n_qubits``, ``n_qumodes``, ``gate_counts``, ``n_gates``, ``n_two_qubit``,
``depth``, ``n_parameters``.

What "two-qubit gate" means differs by stack, and pretending otherwise would
be the easiest way to produce a misleading table:

- On the DV side it is literal CNOTs after transpiling to ``{cx, u}``, which is
  the basis set behind the "393 CNOT, 265 U3" figure quoted in the HyQBench
  paper (arXiv:2603.04398 section 2).
- On the CV-DV side it is the count of *native hybrid entangling gates*
  (`ECD`, `JaynesCummings`, `ConditionalDisplacement`, `Beamsplitter`, ...) —
  each a single hardware pulse on a circuit-QED or trapped-ion device, not a
  CNOT. The two columns are therefore counts of different physical things, and
  the tables label them as such. `cvdv_gate_cost_in_dv` in this module is what
  puts them on one axis: it synthesizes a native CV-DV gate onto encoded
  qubits and reports how many CNOTs that costs.

All-to-all connectivity is assumed on the DV side, matching Bosonic Qiskit's
all-to-all model (HyQBench section 4). A restricted coupling map would only
inflate the DV counts, so the comparison is conservative.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

#: Native CV-DV gates that entangle two or more wires. Used to separate
#: "entangling native operations" from single-wire rotations on the CV-DV side.
CVDV_ENTANGLING_GATES = frozenset(
    {
        "Beamsplitter",
        "TwoModeSqueezing",
        "TwoModeSum",
        "ModeSwap",
        "ConditionalRotation",
        "ConditionalDisplacement",
        "ConditionalXDisplacement",
        "ConditionalYDisplacement",
        "ConditionalSqueezing",
        "ConditionalParity",
        "SelectiveQubitRotation",
        "JaynesCummings",
        "AntiJaynesCummings",
        "Rabi",
        "EchoedConditionalDisplacement",
        "ConditionalBeamsplitter",
        "ConditionalTwoModeSqueezing",
        "ConditionalTwoModeSum",
        "CNOT",
        "CZ",
        "IsingZZ",
    }
)

DEFAULT_BASIS = ("cx", "u")


# ---------------------------------------------------------------------------
# DV / Qiskit
# ---------------------------------------------------------------------------


def count_dv(
    circuit: QuantumCircuit,
    basis_gates: tuple[str, ...] = DEFAULT_BASIS,
    optimization_level: int = 3,
    transpile_circuit: bool = True,
    seed: int = 0,
) -> dict:
    """Transpile to `basis_gates` and count qubits, gates, CNOTs, and depth."""
    qc = circuit
    if transpile_circuit:
        qc = transpile(
            circuit,
            basis_gates=list(basis_gates),
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
    counts = {k: int(v) for k, v in qc.count_ops().items()}
    return {
        "n_qubits": qc.num_qubits,
        "n_qumodes": 0,
        "gate_counts": counts,
        "n_gates": int(sum(counts.values())),
        "n_two_qubit": int(counts.get("cx", 0) + counts.get("cz", 0) + counts.get("rzz", 0)),
        "depth": int(qc.depth()),
        "n_parameters": int(circuit.num_parameters),
        "transpiled": qc if transpile_circuit else None,
    }


def cvdv_gate_cost_in_dv(
    matrix: np.ndarray,
    basis_gates: tuple[str, ...] = DEFAULT_BASIS,
    optimization_level: int = 3,
) -> dict:
    """CNOT cost of one native CV-DV gate, once its mode(s) are qubit-encoded.

    `matrix` is the gate's action on the encoded qubit space (see
    `boson_encoding.embed`). Synthesis uses Qiskit's default unitary synthesis
    (Quantum Shannon decomposition), which is what "an ECD gate costs ~93
    CNOTs" style figures come from. Exponential in qubit count — keep the
    encoded register small.
    """
    n_qubits = int(np.log2(matrix.shape[0]))
    qc = QuantumCircuit(n_qubits)
    qc.append(UnitaryGate(matrix), range(n_qubits))
    return count_dv(qc, basis_gates=basis_gates, optimization_level=optimization_level)


# ---------------------------------------------------------------------------
# CV-DV / PennyLane + hybridlane
# ---------------------------------------------------------------------------


def optimize_cvdv(qnode):
    """Apply PennyLane's built-in peephole optimizations to a hybridlane QNode.

    `cancel_inverses` removes adjacent gate/adjoint pairs and `merge_rotations`
    fuses adjacent rotations about the same axis on the same wire — the
    circuit-level counterpart of what Qiskit's `optimization_level=3` does for
    the DV baseline, so both sides are reported at their compiler's best.

    Two omissions are deliberate. `single_qubit_fusion` is *not* included: on
    these circuits it rewrites each `RZ` into a generic `Rot` (an `RZ RY RZ`
    triple) and inflates the JCH count from 550 gates to 850, because it is a
    normalizer rather than a reducer and there is nothing on those wires to fuse
    with. And `hybridlane` ships no optimization transforms of its own
    (`hqml.transforms` exposes only `from_pennylane`), so PennyLane's stack is
    the whole of the CV-DV-side optimizer.

    Expect this to find nothing on a Trotterized JCH circuit, which is the
    point rather than a disappointment: consecutive gates there are
    non-commuting exponentials of distinct Hamiltonian terms, so there is no
    redundancy to remove. See `jch_resource_comparison.ipynb` section 5.6.
    """
    import pennylane as qml

    return qml.transforms.merge_rotations(qml.transforms.cancel_inverses(qnode))


def count_cvdv(
    qnode,
    *args,
    level: str = "device",
    n_parameters: int | None = None,
    optimize: bool = False,
    **kwargs,
) -> dict:
    """Resource count for a hybridlane QNode.

    Uses `qml.specs(qnode, level=...)` for gate counts and depth, and
    `hybridlane.type_check` to split the wires into qubits vs. qumodes.
    `level="device"` counts *native* gates after decomposition — the right
    level for "what does the hardware actually run".

    `qml.specs` does not expose a trainable-parameter count for these circuits
    (the ansatz parameters are baked into the tape as plain floats), so pass
    `n_parameters` explicitly when the count matters for a comparison table.

    `optimize=True` routes the circuit through `optimize_cvdv` first, which is
    how the notebooks report an optimized CV-DV column beside the raw one.
    """
    import pennylane as qml

    import hybridlane as hqml

    if optimize:
        qnode = optimize_cvdv(qnode)

    specs = qml.specs(qnode, level=level)(*args, **kwargs)
    resources = specs.resources
    counts = {str(k): int(v) for k, v in resources.gate_types.items()}

    wire_types = hqml.type_check(qnode)(*args, **kwargs).wire_types
    n_qumodes = sum(1 for t in wire_types.values() if isinstance(t, hqml.wires.Qumode))
    n_qubits = len(wire_types) - n_qumodes

    return {
        "n_qubits": n_qubits,
        "n_qumodes": n_qumodes,
        "gate_counts": counts,
        "n_gates": int(sum(counts.values())),
        "n_two_qubit": int(sum(v for k, v in counts.items() if k in CVDV_ENTANGLING_GATES)),
        "depth": int(resources.depth),
        "n_parameters": int(n_parameters) if n_parameters is not None else None,
        "specs": specs,
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def compare_table(rows: dict[str, dict], columns: list[str] | None = None) -> str:
    """Render `{label: resource_dict}` as a GitHub-flavoured markdown table."""
    if columns is None:
        columns = [
            "n_qubits",
            "n_qumodes",
            "n_gates",
            "n_two_qubit",
            "depth",
            "n_parameters",
        ]
    header = "| " + " | ".join(["method", *columns]) + " |"
    rule = "|" + "|".join(["---"] * (len(columns) + 1)) + "|"
    lines = [header, rule]
    for label, row in rows.items():
        cells = [_fmt(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value in (None, ""):
        return "-"
    return str(value)


#: Grey that stays visible on both light and dark notebook themes. The tables
#: below set no background and no text colour, so they inherit the theme's ink
#: and only the rules are drawn — which is what keeps them readable in either.
_RULE = "rgba(128,128,128,0.45)"
_RULE_FAINT = "rgba(128,128,128,0.22)"


def fmt_cell(value, spec: str | None = None) -> str:
    """Format one table cell: thousands separators for ints, `.3g`/`.2e` for floats."""
    if value is None or value == "":
        return "&ndash;"
    if spec is not None:
        return format(value, spec)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(value)


def html_table(
    headers: list[str],
    rows: list[list],
    caption: str | None = None,
    note: str | None = None,
    formats: list[str | None] | None = None,
    emphasize: set[int] | None = None,
    align: str = "right",
) -> object:
    """Render a table as styled HTML for a notebook.

    The first column is the row label (left aligned); the rest are `align`ed.
    `emphasize` is a set of row indices drawn in bold with an accent rule — used
    to make the CV-DV row stand out from the DV baselines it is compared to.
    Returns an `IPython.display.HTML` so a cell can just end with the call.
    """
    from IPython.display import HTML

    emphasize = emphasize or set()
    formats = formats or [None] * len(headers)

    def cells(row, tag, styles):
        return "".join(
            f'<{tag} style="{styles(j)}">{fmt_cell(v, formats[j]) if tag == "td" else v}</{tag}>'
            for j, v in enumerate(row)
        )

    def head_style(j):
        return (
            f"padding:0.4em 0.9em;text-align:{'left' if j == 0 else align};"
            f"border-bottom:2px solid {_RULE};font-weight:600;white-space:nowrap;"
        )

    parts = [
        '<div style="margin:0.6em 0 1.2em 0;overflow-x:auto;">',
    ]
    if caption:
        parts.append(
            f'<div style="font-weight:600;margin-bottom:0.35em;font-size:0.95em;">{caption}</div>'
        )
    parts.append(
        '<table style="border-collapse:collapse;font-variant-numeric:tabular-nums;'
        'font-size:0.92em;line-height:1.5;">'
    )
    parts.append(f"<thead><tr>{cells(headers, 'th', head_style)}</tr></thead><tbody>")
    for i, row in enumerate(rows):
        bold = "font-weight:600;" if i in emphasize else ""
        rule = _RULE if i in emphasize else _RULE_FAINT

        def body_style(j, bold=bold, rule=rule):
            return (
                f"padding:0.35em 0.9em;text-align:{'left' if j == 0 else align};"
                f"border-bottom:1px solid {rule};{bold}white-space:nowrap;"
            )

        parts.append(f"<tr>{cells(row, 'td', body_style)}</tr>")
    parts.append("</tbody></table>")
    if note:
        parts.append(
            '<div style="font-size:0.85em;opacity:0.75;margin-top:0.5em;'
            f'max-width:52em;">{note}</div>'
        )
    parts.append("</div>")
    return HTML("".join(parts))


# ---------------------------------------------------------------------------
# Analytic Pauli-evolution cost
# ---------------------------------------------------------------------------


def pauli_evolution_cost(op) -> dict:
    """Exact gate cost of `exp(-i t H)` for a `SparsePauliOp`, without transpiling.

    Standard synthesis of one term `exp(-i theta P)` is a CNOT ladder onto one
    target, an `Rz`, and the reversed ladder: `2*(w - 1)` CNOTs for a weight-`w`
    Pauli string, plus basis-change rotations on the X/Y legs. Summing that over
    terms approximates what Qiskit's transpiler produces from a
    `PauliEvolutionGate`, in microseconds instead of minutes.

    This matters because the term count grows steeply with the Fock cutoff — one
    JCH hopping bond is 32 Pauli terms at cutoff 4 but 12,800 at cutoff 32 — and
    transpiling that at `optimization_level=3` dominates the runtime of a cutoff
    sweep.

    **It is an upper bound, and a loose one at the extremes.** The sum is blind
    to cancellation between neighbouring ladders, so it over-counts the
    transpiled `cx_structure="chain"` result by 8-27% across cutoffs 4-32, and
    by 2x in the small cases where the transpiler finds almost everything (JCH
    binary at cutoff 2, unary at 16). It models the *baseline* compilation only:
    against `cx_structure="fountain"` the gap is a further factor of ~2.
    `test_analytic_cost_is_an_upper_bound` pins that direction so the estimate
    can never silently start under-counting, which is the failure that would
    matter.

    Identity terms are skipped: they are a global phase, not a gate.
    """
    n_cx = 0
    n_rz = 0
    n_basis = 0
    for pauli in op.paulis:
        label = str(pauli)
        support = [c for c in label if c != "I"]
        weight = len(support)
        if weight == 0:
            continue
        n_cx += 2 * (weight - 1)
        n_rz += 1
        # X legs need an H on each side; Y legs an Sdg-H pair on each side.
        n_basis += 2 * sum(1 for c in support if c in "XY")
    return {
        "n_terms": len(op),
        "n_two_qubit": n_cx,
        "n_rz": n_rz,
        "n_basis_change": n_basis,
        "n_gates": n_cx + n_rz + n_basis,
    }


def jch_step_cost_analytic(terms) -> dict:
    """Total analytic cost of one Trotter step, given `jch_dv_terms` output."""
    total = {"n_terms": 0, "n_two_qubit": 0, "n_rz": 0, "n_basis_change": 0, "n_gates": 0}
    for _name, op in terms:
        for key, value in pauli_evolution_cost(op).items():
            total[key] += value
    return total


def qsd_cnot_count(n_qubits: int) -> int:
    """CNOTs to synthesize a *generic* n-qubit unitary (Quantum Shannon decomp).

    `(23/48) 4^n - (3/2) 2^n + 4/3` (Shende, Bullock & Markov 2006). Used to
    quote the cost of `synthesis="exact"` without actually running the
    transpiler: at 12 qubits this is ~8 million CNOTs, so transpiling it is not
    merely slow but pointless. Exact synthesis is a simulation reference for
    *accuracy*, not a compilable circuit, and the resource tables say so rather
    than trying to compute it.
    """
    return round((23 / 48) * 4**n_qubits - 1.5 * 2**n_qubits + 4 / 3)
