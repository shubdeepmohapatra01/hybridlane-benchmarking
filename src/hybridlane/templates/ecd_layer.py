# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

from collections.abc import Sequence
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hqml

from ..ops import Hybrid


class ECDLayer(Operation, Hybrid):
    r"""One layer of the ECD VQE ansatz on 1 qubit and 2 qumodes.

    Implements the circuit block:

    .. math::

        L = \mathrm{ECD}_0(\beta_1)\; R(\theta_1, \phi_1)\;
            \mathrm{ECD}_1(\beta_2)\; R(\theta_2, \phi_2)

    where :math:`R(\theta, \phi) = e^{-i(\theta/2)(\cos\phi\, X + \sin\phi\, Y)}`
    is a qubit rotation decomposed as :math:`RZ(\phi)\,RX(\theta)\,RZ(-\phi)`,
    and :math:`\mathrm{ECD}_k(\beta)` is the Echoed Conditional Displacement on
    qumode :math:`k` with complex amplitude :math:`\beta = |\beta|e^{i\varphi}`.

    **Wire convention**: ``wires = (qubit, qumode_primary, qumode_auxiliary)``.
    The primary qumode (:math:`m_0`) encodes the main quantum register; the
    auxiliary qumode (:math:`m_1`) holds ancillary or slack degrees of freedom.

    **Details**:

    * Number of wires: 3
    * Wire arguments: ``[qubit, qumode_primary, qumode_auxiliary]``
    * Number of parameters: 8

    Args:
        beta1_mag (TensorLike): Magnitude :math:`|\beta_1|` of the displacement
            amplitude for the primary qumode.
        beta1_arg (TensorLike): Phase :math:`\varphi_1` of the displacement
            amplitude for the primary qumode.
        theta1 (TensorLike): Qubit rotation angle :math:`\theta_1`.
        phi1 (TensorLike): Qubit rotation phase :math:`\phi_1`.
        beta2_mag (TensorLike): Magnitude :math:`|\beta_2|` of the displacement
            amplitude for the auxiliary qumode.
        beta2_arg (TensorLike): Phase :math:`\varphi_2` of the displacement
            amplitude for the auxiliary qumode.
        theta2 (TensorLike): Qubit rotation angle :math:`\theta_2`.
        phi2 (TensorLike): Qubit rotation phase :math:`\phi_2`.
        wires (WiresLike): Wire labels ``(qubit, qumode_primary, qumode_auxiliary)``.
        id (str | None): Optional custom label for the gate.
    """

    num_wires = 3
    num_params = 8
    grad_method = None
    resource_keys = set()

    type_signature = (hqml.sa.Qubit(), hqml.sa.Qumode(), hqml.sa.Qumode())

    def __init__(
        self,
        beta1_mag: TensorLike,
        beta1_arg: TensorLike,
        theta1: TensorLike,
        phi1: TensorLike,
        beta2_mag: TensorLike,
        beta2_arg: TensorLike,
        theta2: TensorLike,
        phi2: TensorLike,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(
            beta1_mag, beta1_arg, theta1, phi1,
            beta2_mag, beta2_arg, theta2, phi2,
            wires=wires, id=id,
        )

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires = None,
        **hyperparameters: dict[str, Any],
    ) -> Sequence[Operation]:
        beta1_mag, beta1_arg, theta1, phi1, beta2_mag, beta2_arg, theta2, phi2 = params
        q  = wires[0]   # qubit
        m0 = wires[1]   # primary qumode
        m1 = wires[2]   # auxiliary qumode

        return [
            # R(theta1, phi1) on qubit, entangled with primary qumode
            qml.RZ(-phi1, q),
            qml.RX(theta1, q),
            qml.RZ(phi1, q),
            hqml.ECD(beta1_mag, beta1_arg, [q, m0]),
            # R(theta2, phi2) on qubit, entangled with auxiliary qumode
            qml.RZ(-phi2, q),
            qml.RX(theta2, q),
            qml.RZ(phi2, q),
            hqml.ECD(beta2_mag, beta2_arg, [q, m1]),
        ]

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "ECD-L", cache=cache
        )


# ---------------------------------------------------------------------------
# Decomposition registration
# ---------------------------------------------------------------------------

@qml.register_resources(
    {
        qml.RZ: 4,
        qml.RX: 2,
        hqml.ECD: 2,
    }
)
def _ecd_layer_decomp(
    beta1_mag, beta1_arg, theta1, phi1,
    beta2_mag, beta2_arg, theta2, phi2,
    wires, **_
):
    q  = wires[0]
    m0 = wires[1]
    m1 = wires[2]

    qml.RZ(-phi1, q)
    qml.RX(theta1, q)
    qml.RZ(phi1, q)
    hqml.ECD(beta1_mag, beta1_arg, [q, m0])

    qml.RZ(-phi2, q)
    qml.RX(theta2, q)
    qml.RZ(phi2, q)
    hqml.ECD(beta2_mag, beta2_arg, [q, m1])


qml.add_decomps(ECDLayer, _ecd_layer_decomp)


# ---------------------------------------------------------------------------
# Parameter helper
# ---------------------------------------------------------------------------

def random_ecd_params(ndepth: int, rng=None) -> np.ndarray:
    """Return a flat array of random initial parameters for ``ndepth`` ECD layers.

    Parameter order per layer (8 values):
    ``beta1_mag, beta1_arg, theta1, phi1, beta2_mag, beta2_arg, theta2, phi2``

    Args:
        ndepth: Number of :class:`ECDLayer` layers.
        rng: Optional :class:`numpy.random.Generator` for reproducibility.

    Returns:
        1-D array of shape ``(8 * ndepth,)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    beta_mag = rng.uniform(0.0, 3.0,        size=(ndepth, 2))
    beta_arg = rng.uniform(0.0, 2 * np.pi,  size=(ndepth, 2))
    theta    = rng.uniform(0.0, np.pi,       size=(ndepth, 2))
    phi      = rng.uniform(0.0, 2 * np.pi,  size=(ndepth, 2))

    layers = np.stack(
        [beta_mag[:, 0], beta_arg[:, 0], theta[:, 0], phi[:, 0],
         beta_mag[:, 1], beta_arg[:, 1], theta[:, 1], phi[:, 1]],
        axis=1,
    )
    return layers.flatten()
