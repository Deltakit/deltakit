# (c) Copyright Riverlane 2020-2025.
"""
This module includes implementations of phenomenological
noise. `PhenomenologicalNoise` adds noise to `I` gates.
`ToyPhenomenologicalNoise` specifies this noise
to be Depolarise1, and adds measurement flip noise.
"""

from collections.abc import Callable
from dataclasses import dataclass

from deltakit_circuit import Qubit, measurement_noise_profile
from deltakit_circuit.gates import I
from deltakit_circuit.noise_channels import Depolarise1, OneQubitNoiseChannel

from deltakit_explorer.qpu._noise._noise_parameters import NoiseParameters


@dataclass(kw_only=True)
class PhenomenologicalNoise(NoiseParameters):
    """
    Class for capturing phenomenological noise.

    Parameters
    ----------
    phenomenological_noise : Optional[Callable[[Qubit], OneQubitNoiseChannel]]
        Phenomenological noise which depends on qubit only. By default,
        no noise.
    """

    phenomenological_noise: Callable[[Qubit], OneQubitNoiseChannel] | None = None

    def __post_init__(self):
        if self.phenomenological_noise is not None:
            self.gate_noise.append(
                lambda noise_context: [
                    self.phenomenological_noise(qubit)
                    for qubit in noise_context.gate_layer_qubits(I)
                ]
            )


@dataclass(kw_only=True)
class ToyPhenomenologicalNoise(PhenomenologicalNoise):
    """
    Class for capturing a phenomenological noise model with simple input parameters.

    Parameters
    ----------
    p : float, optional
        The depolarising error rate for phenomenological noise. By default, 0.0.
    p_measurement_flip: Optional[float]
        The probability of obtaining an incorrect measurement result. By default, this
        has the same value as p.
    """

    p: float = 0.0
    p_measurement_flip: float | None = None

    def __post_init__(self):
        if self.p_measurement_flip is None:
            self.p_measurement_flip = self.p
        self.measurement_flip = measurement_noise_profile(self.p_measurement_flip)
        self.phenomenological_noise = lambda qubit: Depolarise1(qubit, self.p)
        super().__post_init__()
