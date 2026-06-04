# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

from collections.abc import Sequence
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.decomposition.resources import adjoint_resource_rep
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hqml

from ..ops import ConditionalDisplacement, Hybrid


class StateTransferCVtoDV(Operation, Hybrid):
    r"""CV-to-DV state transfer using the non-abelian protocol.

    Transfers a qumode state to n qubits using alternating V_j and W_j gates
    followed by a basis transformation.

    V_j implements a sigma_y controlled position displacement.

    W_j implements a sigma_x controlled momentum displacement.

    The ``wires`` attribute is ``(q0, q1, ..., q_{n-1}, qumode)``.

    **Details**:

    * Number of wires: variable (n_qubits + 1 qumode)
    * Wire arguments: ``[qubit_0, qubit_1, ..., qubit_{n-1}, qumode]``
    * Number of parameters: 2

    Args:
        n_qubits: Number of qubits for the DV register.
        lmbda: Coupling strength parameter (default 0.29).
        wires: Wire labels for the qubits and qumode.
        id: Custom label for the gate.
    """

    num_wires = None  # variable: n_qubits + 1 qumode
    num_params = 2
    grad_method = None
    resource_keys = set()

    def __init__(
        self,
        n_qubits: int,
        lmbda: TensorLike = 0.29,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(n_qubits, lmbda, wires=wires, id=id)

    @property
    def type_signature(self):
        n = int(self.parameters[0])
        return tuple([hqml.sa.Qubit()] * n + [hqml.sa.Qumode()])

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires = None,
        **hyperparameters: dict[str, Any],
    ) -> Sequence[Operation]:
        n_qubits_param, lmbda = params
        n = int(n_qubits_param)
        ops = []

        qubit_wires = list(wires[:n])
        m = wires[n]  # qumode wire

        # V_j and W_j gates for j = 1 to n
        for j in range(1, n + 1):
            qb = qubit_wires[n - j]  # reversed index matching c2qa convention

            # --- V_j: sigma_y controlled position displacement ---
            # S†(qb) → H(qb) → CD(a, pi/2, [qb, m]) → H(qb) → S(qb)
            a_vj = np.pi / (2 ** (j + 1) * float(lmbda) * np.sqrt(2))
            ops.append(qml.adjoint(qml.S)(qb))
            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(a_vj, np.pi / 2, [qb, m]))
            ops.append(qml.H(qb))
            ops.append(qml.S(qb))

            # --- W_j: sigma_x controlled momentum displacement ---
            # H(qb) → CD(b, phi, [qb, m]) → H(qb)
            b_wj = float(lmbda) * 2 ** (j - 1) / np.sqrt(2)
            if j == n:
                phi_wj = 0.0  # positive real
            else:
                phi_wj = np.pi  # negative real

            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(b_wj, phi_wj, [qb, m]))
            ops.append(qml.H(qb))

        # --- Basis transformation ---
        for i in range(n):
            qb_i = qubit_wires[i]
            ops.append(qml.H(qb_i))
            if i == n - 1:  # MSB
                ops.append(qml.X(qb_i))
                ops.append(qml.Z(qb_i))
            elif i == 0:  # LSB
                ops.append(qml.Z(qb_i))
            else:  # Middle qubits
                ops.append(qml.X(qb_i))

        return ops

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals,
            base_label=base_label or "CV→DV",
            cache=cache,
        )


# ---------------------------------------------------------------------------
# Decomposition registration
# ---------------------------------------------------------------------------

@qml.register_resources(
    {
        qml.H: 1,
        qml.S: 1,
        adjoint_resource_rep(qml.S): 1,
        ConditionalDisplacement: 1,
        qml.X: 1,
        qml.Z: 1,
    }
)
def _state_transfer_decomp(*params, wires, **_):
    n_qubits_param, lmbda = params
    n = int(n_qubits_param)

    qubit_wires = list(wires[:n])
    m = wires[n]

    for j in range(1, n + 1):
        qb = qubit_wires[n - j]

        # V_j
        a_vj = np.pi / (2 ** (j + 1) * float(lmbda) * np.sqrt(2))
        qml.adjoint(qml.S)(qb)
        qml.H(qb)
        ConditionalDisplacement(a_vj, np.pi / 2, [qb, m])
        qml.H(qb)
        qml.S(qb)

        # W_j
        b_wj = float(lmbda) * 2 ** (j - 1) / np.sqrt(2)
        phi_wj = 0.0 if j == n else np.pi
        qml.H(qb)
        ConditionalDisplacement(b_wj, phi_wj, [qb, m])
        qml.H(qb)

    # Basis transformation
    for i in range(n):
        qb_i = qubit_wires[i]
        qml.H(qb_i)
        if i == n - 1:
            qml.X(qb_i)
            qml.Z(qb_i)
        elif i == 0:
            qml.Z(qb_i)
        else:
            qml.X(qb_i)


qml.add_decomps(StateTransferCVtoDV, _state_transfer_decomp)


# ---------------------------------------------------------------------------
# DV-to-CV state transfer (inverse of CV-to-DV)
# ---------------------------------------------------------------------------

class StateTransferDVtoCV(Operation, Hybrid):
    r"""DV-to-CV state transfer using the non-abelian protocol.

    Transfers an n-qubit state to a qumode. This is the inverse (adjoint) of
    :class:`StateTransferCVtoDV`.

    The decomposition applies the reverse basis transformation followed by
    the adjoint V_j and W_j gates in reverse order:

    .. math::

        U_{D/A} = \prod_{j=1}^{n} V_j^\dagger W_j^\dagger \cdot B^\dagger

    where :math:`B` is the basis transformation and :math:`V_j^\dagger`,
    :math:`W_j^\dagger` are the adjoints of the forward gates (CD phase
    shifted by :math:`\pi`).

    The ``wires`` attribute is ``(q0, q1, ..., q_{n-1}, qumode)``.

    **Details**:

    * Number of wires: variable (n_qubits + 1 qumode)
    * Wire arguments: ``[qubit_0, qubit_1, ..., qubit_{n-1}, qumode]``
    * Number of parameters: 2

    Args:
        n_qubits: Number of qubits for the DV register.
        lmbda: Coupling strength parameter (default 0.29).
        wires: Wire labels for the qubits and qumode.
        id: Custom label for the gate.
    """

    num_wires = None
    num_params = 2
    grad_method = None
    resource_keys = set()

    def __init__(
        self,
        n_qubits: int,
        lmbda: TensorLike = 0.29,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(n_qubits, lmbda, wires=wires, id=id)

    @property
    def type_signature(self):
        n = int(self.parameters[0])
        return tuple([hqml.sa.Qubit()] * n + [hqml.sa.Qumode()])

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires = None,
        **hyperparameters,
    ) -> Sequence[Operation]:
        n_qubits_param, lmbda = params
        n = int(n_qubits_param)
        ops = []

        qubit_wires = list(wires[:n])
        m = wires[n]  # qumode wire

        # --- Reverse basis transformation (adjoint of forward basis) ---
        # Forward was: H, then [X,Z | Z | X] per qubit
        # Adjoint reverses order: [Z,X | Z | X], then H
        for i in range(n):
            qb_i = qubit_wires[i]
            if i == n - 1:  # MSB
                ops.append(qml.Z(qb_i))
                ops.append(qml.X(qb_i))
            elif i == 0:  # LSB
                ops.append(qml.Z(qb_i))
            else:  # Middle qubits
                ops.append(qml.X(qb_i))
            ops.append(qml.H(qb_i))

        # --- Adjoint W_j and V_j gates, j = n down to 1 ---
        # CD(a, phi)† = CD(a, phi + pi), so each CD phase is shifted by pi
        for j in range(n, 0, -1):
            qb = qubit_wires[n - j]

            # W_j†: H, CD(b, phi + pi), H
            b_wj = float(lmbda) * 2 ** (j - 1) / np.sqrt(2)
            if j == n:
                phi_wj_adj = np.pi  # forward was 0, adjoint adds pi
            else:
                phi_wj_adj = 0.0  # forward was pi, adjoint adds pi -> 2pi = 0

            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(b_wj, phi_wj_adj, [qb, m]))
            ops.append(qml.H(qb))

            # V_j†: S†, H, CD(a, 3pi/2), H, S
            a_vj = np.pi / (2 ** (j + 1) * float(lmbda) * np.sqrt(2))
            ops.append(qml.adjoint(qml.S)(qb))
            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(a_vj, 3 * np.pi / 2, [qb, m]))
            ops.append(qml.H(qb))
            ops.append(qml.S(qb))

        return ops

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals,
            base_label=base_label or "DV→CV",
            cache=cache,
        )


@qml.register_resources(
    {
        qml.H: 1,
        qml.S: 1,
        adjoint_resource_rep(qml.S): 1,
        ConditionalDisplacement: 1,
        qml.X: 1,
        qml.Z: 1,
    }
)
def _state_transfer_dv_to_cv_decomp(*params, wires, **_):
    n_qubits_param, lmbda = params
    n = int(n_qubits_param)

    qubit_wires = list(wires[:n])
    m = wires[n]

    # Reverse basis transformation
    for i in range(n):
        qb_i = qubit_wires[i]
        if i == n - 1:
            qml.Z(qb_i)
            qml.X(qb_i)
        elif i == 0:
            qml.Z(qb_i)
        else:
            qml.X(qb_i)
        qml.H(qb_i)

    # Adjoint W_j† then V_j†, j = n down to 1
    for j in range(n, 0, -1):
        qb = qubit_wires[n - j]

        # W_j†
        b_wj = float(lmbda) * 2 ** (j - 1) / np.sqrt(2)
        phi_wj_adj = np.pi if j == n else 0.0
        qml.H(qb)
        ConditionalDisplacement(b_wj, phi_wj_adj, [qb, m])
        qml.H(qb)

        # V_j†
        a_vj = np.pi / (2 ** (j + 1) * float(lmbda) * np.sqrt(2))
        qml.adjoint(qml.S)(qb)
        qml.H(qb)
        ConditionalDisplacement(a_vj, 3 * np.pi / 2, [qb, m])
        qml.H(qb)
        qml.S(qb)


qml.add_decomps(StateTransferDVtoCV, _state_transfer_dv_to_cv_decomp)
