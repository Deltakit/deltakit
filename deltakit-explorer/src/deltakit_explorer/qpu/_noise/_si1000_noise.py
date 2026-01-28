# (c) Copyright Riverlane 2020-2025.
"""
This module includes implementation of a SI1000 noise model.
Toy noise is a default choice of noise parameters to explore codes
scalability and to compare against other works. This a de-facto
experimental standard.
"""

from dataclasses import dataclass

from deltakit_circuit import after_reset_flip_probability, measurement_noise_profile
from deltakit_circuit.gates import TWO_QUBIT_GATES, OneQubitCliffordGate
from deltakit_circuit.noise_channels import Depolarise1, Depolarise2

from deltakit_explorer.qpu._noise._leakage_noise_profiles import (
    idle_qubit_relaxation_noise_profile,
    one_qubit_clifford_gate_relaxation_noise_profile,
    qubit_reset_leakage_noise_profile,
    resonator_idle_qubit_relaxation_noise_profile,
    two_qubit_gate_leakage_noise_profile,
    two_qubit_gate_relaxation_noise_profile,
)
from deltakit_explorer.qpu._noise._noise_parameters import NoiseParameters


@dataclass(kw_only=True)
class SI1000Noise(NoiseParameters):
    """
    Superconducting inspired noise model from: https://arxiv.org/abs/2108.10457.
    This noise model assumes that after every measurement there is a reset.
    Includes an optional parameter pL for leakage implemented as described
    in the LCD paper: https://arxiv.org/abs/2411.10343.

    Parameters
    ----------
    p: float
        Physical error rate.
    pL: float
        Informs leakage related probabilities.
    """

    name = "SI1000_noise"

    p: float = 0
    pL: float = 0

    def __post_init__(self):
        depolarise1_generator = Depolarise1.generator_from_prob(self.p / 10)
        depolarise2_generator = Depolarise2.generator_from_prob(self.p)

        self.gate_noise.append(
            lambda noise_context: depolarise1_generator(
                noise_context.gate_layer_qubits(
                    (*tuple(TWO_QUBIT_GATES), OneQubitCliffordGate), gate_qubit_count=1
                )
            )
        )

        self.gate_noise.append(
            lambda noise_context: depolarise2_generator(
                noise_context.gate_layer_qubits(
                    tuple(TWO_QUBIT_GATES), gate_qubit_count=2
                )
            )
        )

        self.idle_noise = lambda qubit, _t=0.0: Depolarise1(
            qubit=qubit, probability=self.p / 10
        )

        self.reset_noise = after_reset_flip_probability(2 * self.p)

        resonator_leakage = 4 * self.p if self.pL > 0 else 0
        self.gate_noise.append(
            resonator_idle_qubit_relaxation_noise_profile(2 * self.p, resonator_leakage)
        )

        if self.pL > 0:
            self.gate_noise.append(two_qubit_gate_leakage_noise_profile(self.pL))
            self.reset_noise.append(qubit_reset_leakage_noise_profile(self.pL))
            self.gate_noise.append(two_qubit_gate_relaxation_noise_profile(self.pL))
            self.gate_noise.append(
                one_qubit_clifford_gate_relaxation_noise_profile(self.p / 5)
            )
            self.gate_noise.append(idle_qubit_relaxation_noise_profile(self.p / 5))

        self.measurement_flip = measurement_noise_profile(5 * self.p)

    def __str__(self) -> str:
        if self.pL:
            return f"{self.name}_{self.p:.0e}_{self.pL:.0e}"
        return f"{self.name}_{self.p:.0e}"
