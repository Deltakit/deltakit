from dataclasses import dataclass
from typing import Callable

from deltakit_circuit import measurement_noise_profile
from deltakit_circuit.gates import (I, OneQubitCliffordGate,
                                    OneQubitMeasurementGate, OneQubitResetGate)
from deltakit_circuit.noise_channels import Depolarise1, Depolarise2
from deltakit_explorer.qpu._noise._noise_parameters import (
    NoiseParameters, _idle_noise_from_t1_t2)


@dataclass
class PhysicalNoise(NoiseParameters):
    r"""Gets gate noise parameters given physical gate data

    Parameters
    ----------
    t1: float
        :math:`T_1` time (relaxation from :math:`\ket{1}` to :math:`\ket{0}`), seconds.
    t2: float
        :math:`T_2` time (dephasing), seconds.
    p_1_qubit_gate_error: float
        Probability of a flip while doing a 1-qubit gate.
    p_2_qubit_gate_error: float
        Probability of a flip while doing a 2-qubit gate.
    p_reset_error: float
        Probability of a flip while doing a reset.
    p_meas_qubit_error: float
        Probability of incorrect measurement.
    p_readout_flip: float
        Probability of a flip while measuring a qubit.

    Returns
    -------
    noise_model: NoiseParameters
        A `NoiseParameters` object representing the noise model of a QPU.

    Examples
    --------
    >>> from deltakit.circuit import Qubit
    >>> from deltakit.explorer import qpu
    >>> noise_model = qpu.PhysicalNoise(
    ...     t1=20e-6,
    ...     t2=30e-6,
    ...     p_readout_flip=0.01,
    ...     p_1_qubit_gate_error=0.001,
    ...     p_2_qubit_gate_error=0.01,
    ...     p_reset_error=0.01,
    ...     p_meas_qubit_error=0.01,
    ... )

    """

    name = "PhysicalNoise"

    t1: float = 0
    t2: float = 0
    p_1_qubit_gate_error: float = 0
    p_2_qubit_gate_error: float = 0
    p_reset_error: float = 0
    p_meas_qubit_error: float = 0
    p_readout_flip: float = 0

    def __post_init__(self):
        idle_noise = _idle_noise_from_t1_t2(self.t1, self.t2)

        def _gate_noise(noise_context):
            noise_ops = []
            for gate in noise_context.gate_layer.gates:
                if isinstance(gate, I):
                    continue
                if len(gate.qubits) == 2:
                    noise_ops.append(
                        Depolarise2(*gate.qubits, self.p_2_qubit_gate_error)  # type: ignore[call-arg]
                    )
                elif isinstance(gate, OneQubitCliffordGate):
                    noise_ops.append(
                        Depolarise1(*gate.qubits, self.p_1_qubit_gate_error)
                    )
            return noise_ops

        gate_noise = [_gate_noise]

        reset_noise: list[Callable] = [
            lambda noise_context: Depolarise1.generator_from_prob(self.p_reset_error)(
                noise_context.gate_layer_qubits(OneQubitResetGate)
            )
        ]
        measurement_noise: list[Callable] = [
            lambda noise_context: Depolarise1.generator_from_prob(
                self.p_meas_qubit_error
            )(noise_context.gate_layer_qubits(OneQubitMeasurementGate))
        ]
        measurement_flip_noise = measurement_noise_profile(self.p_readout_flip)

        self.gate_noise = gate_noise
        self.idle_noise = idle_noise
        self.reset_noise = reset_noise
        self.measurement_noise_after = measurement_noise
        self.measurement_flip = measurement_flip_noise
