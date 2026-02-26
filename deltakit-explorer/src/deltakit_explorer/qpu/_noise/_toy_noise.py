# (c) Copyright Riverlane 2020-2025.
"""
This module includes implementation on a toy noise model.
Toy noise is a default choice of noise parameters to explore codes
scalability and to compare against other works. This a de-facto
experimental standard.
"""

from dataclasses import dataclass

from deltakit_circuit import measurement_noise_profile
from deltakit_circuit.gates import (
    TWO_QUBIT_GATES,
    OneQubitCliffordGate,
    OneQubitMeasurementGate,
    OneQubitResetGate,
)
from deltakit_circuit.noise_channels import Depolarise1, Depolarise2

from deltakit_explorer.qpu._noise._noise_parameters import NoiseParameters


@dataclass(kw_only=True)
class ToyNoise(NoiseParameters):
    """
    A noise model specified only by two parameters.

    One parameter gives the two-qubit operation error rate, which is assumed to be the
    dominant error. Therefore, all other noise channels (excluding the probability of
    obtaining an incorrect measurement result, which is specified separately) have an
    error rate a factor of ten lower. Idle noise is therefore not time dependent. All
    these noise channels are depolarising channels -- the noise channels occurring
    after a two-qubit gate are two-qubit depolarising channels and all others are
    one-qubit depolarising channels.

    The other parameter gives the probability of obtaining an incorrect measurement
    result.

    Parameters
    ----------
    p: float
        Depolarising error rate of two-qubit gates.
        One-qubit operations, including idle noise, have a depolarising error rate of
        0.1 * p. By default, this is 0.
    p_measurement_flip: Optional[float]
        The probability of obtaining an incorrect measurement result. By default, this
        has the same value as p.
    """

    name = "toy_noise"

    p: float = 0
    p_measurement_flip: float | None = None

    def __post_init__(self):
        self.p_measurement_flip = (
            self.p if self.p_measurement_flip is None else self.p_measurement_flip
        )

        depolarise1_generator = Depolarise1.generator_from_prob(self.p / 10)
        depolarise2_generator = Depolarise2.generator_from_prob(self.p)

        self.gate_noise.append(
            lambda noise_context: depolarise1_generator(
                noise_context.gate_layer_qubits(OneQubitCliffordGate)
            )
        )

        self.gate_noise.append(
            lambda noise_context: depolarise2_generator(
                noise_context.gate_layer_qubits(tuple(TWO_QUBIT_GATES))
            )
        )

        self.idle_noise = lambda qubit, _t=0.0: Depolarise1(
            qubit=qubit, probability=self.p / 10
        )

        # Measurement reset gates are not included here
        self.reset_noise.append(
            lambda noise_context: depolarise1_generator(
                noise_context.gate_layer_qubits(OneQubitResetGate)
            )
        )

        # Measurement reset gates are included here
        self.measurement_noise_after.append(
            lambda noise_context: depolarise1_generator(
                noise_context.gate_layer_qubits(OneQubitMeasurementGate)
            )
        )

        self.measurement_flip = measurement_noise_profile(self.p_measurement_flip)

    def __str__(self) -> str:
        if self.p_measurement_flip != self.p:
            return f"{self.name}_{self.p:.0e}_{self.p_measurement_flip:.0e}"
        return f"{self.name}_{self.p:.0e}"
