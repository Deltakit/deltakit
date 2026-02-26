# (c) Copyright Riverlane 2020-2025.
"""
* `NoiseParameters` treats the quantum system in terms of one- and two-qubit
    noise, which comes together with gate application, measurement, reset,
    or intervals of idleness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
from deltakit_circuit import NoiseProfile, Qubit
from deltakit_circuit.gates import _MeasurementGate
from deltakit_circuit.noise_channels import (
    Depolarise1,
    NoiseChannel,
    OneQubitNoiseChannel,
    PauliChannel1,
)


def _idle_noise_from_t1_t2(
    t1: float, t2: float
) -> Callable[[Qubit, float], OneQubitNoiseChannel]:
    """
    Return a function that calculates idle noise as a function of T1 and T2
    times. See `Ghosh et al. <https://arxiv.org/abs/1210.5799>`_ for the
    derivation, with the final result in equation (10).

    Parameters
    ----------
    t1 : float
        T1 time in s.
    t2 : float
        T2 time in s.

    Returns
    -------
    Callable[[Qubit, float], OneQubitNoiseChannel]
        Function which takes a qubit and a time in s as inputs and returns
        a one-qubit noise channel.
    """
    if t1 <= 0.0:
        msg = "Relaxation time `t1` must be positive."
        raise ValueError(msg)
    if t2 <= 0.0:
        msg = "Dephasing time `t2` must be positive."
        raise ValueError(msg)
    if t2 >= 2 * t1:
        msg = "Dephasing time `t2` must be less than twice relaxation time `t1`."
        raise ValueError(msg)

    def _get_idle_noise(qubit: Qubit, t: float) -> OneQubitNoiseChannel:
        if t1 == t2:
            return Depolarise1(qubit, 0.75 * (1.0 - np.exp(-t / t1)))
        return PauliChannel1(
            qubit,
            0.25 * (1.0 - np.exp(-t / t1)),
            0.25 * (1.0 - np.exp(-t / t1)),
            0.5 * (1.0 - np.exp(-t / t2)) - 0.25 * (1.0 - np.exp(-t / t1)),
        )

    return _get_idle_noise


@dataclass(kw_only=True)
class NoiseParameters:
    """
    Dataclass to capture a noise model for a quantum computer.

    Parameters
    ----------
    gate_noise : Dict[Type[Gate], Callable[[Gate], NoiseChannel]]
        Noise to be added after applying a gate. This is specified through a
        dictionary with gate classes as keys and functions which take a gate and
        return a noise channel as values. The noise can thus be different for each
        gate class and can be dependent on the qubit(s) involved in the gate. By
        default, there is no noise.

    idle_noise : Optional[Callable[[Qubit, float], NoiseChannel]]
        Noise to be added to an idle qubit. This can be dependent on the qubit and the
        time for which it is idle. By default, there is no noise.

    reset_noise : Optional[Callable[[OneQubitResetGate], NoiseChannel]]
        Noise to be added after resetting a qubit. This can be dependent on the
        reset gate applied and the qubit involved. By default, there is no noise.

    measurement_noise_after : Optional[Callable[[OneQubitMeasurementGate], NoiseChannel]]
        Noise to be added after measuring a qubit. This can be dependent on the
        measurement gate applied and the qubit involved. By default, there is no noise.

    measurement_flip : Optional[Callable[[_MeasurementGate], _MeasurementGate]]
        Mapping from a measurement gate to a noisy measurement gate, capturing adding
        error associated with obtaining an incorrect measurement result. By default,
        there is no noise.

    measurement_noise_before : Optional[Callable[[OneQubitMeasurementGate], NoiseChannel]]
        Noise which will be applied before a measurement gate. By default, there is no noise.

    Raises
    ------
    ValueError
        If any key in gate_noise is not a one-qubit or two-qubit gate.
    """

    name = "noise_model"

    gate_noise: list[NoiseProfile] = field(default_factory=list)
    idle_noise: Callable[[Qubit, float], NoiseChannel] | None = None
    reset_noise: list[NoiseProfile] = field(default_factory=list)
    measurement_noise_after: list[NoiseProfile] = field(default_factory=list)
    measurement_noise_before: list[NoiseProfile] = field(default_factory=list)

    measurement_flip: dict[
        type[_MeasurementGate], Callable[[_MeasurementGate], _MeasurementGate]
    ] = field(default_factory=dict)

    def as_noise_profile_after_gate(self) -> list[NoiseProfile]:
        """
        Returns the noise profiles encapsulated by this object as a single generator
        of noise profiles. Omits the noise before measurement.
        """
        return list(
            chain(
                self.gate_noise,
                self.reset_noise,
                self.measurement_noise_after,
            )
        )
