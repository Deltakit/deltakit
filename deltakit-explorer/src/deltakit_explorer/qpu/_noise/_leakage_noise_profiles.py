# (c) Copyright Riverlane 2020-2025.
"""
This module includes noise profiles to add leakage to noise models.
Defines noise profiles for leakage and relaxation contexts parameterised
on leakage (pL) and relaxation (pR) probabilities parameters.
"""

from deltakit_circuit import NoiseProfile
from deltakit_circuit.gates import (
    TWO_QUBIT_GATES,
    Gate,
    OneQubitCliffordGate,
    OneQubitMeasurementGate,
    OneQubitResetGate,
)
from deltakit_circuit.noise_channels import Depolarise1, Leakage, Relax


def two_qubit_gate_leakage_noise_profile(pL: float) -> NoiseProfile:
    """
    Noise profile for leakage channels caused by
    two qubit gates.

    Parameters
    ----------
    pL: float
        Leakage channel probability parameter.

    Returns
    -------
    leakage_two_qubit_gate : NoiseProfile
        Leakage noise profile for two qubit gates.
    """

    def leakage_two_qubit_gate(noise_context):
        return [
            Leakage(qubit, pL)
            for qubit in noise_context.gate_layer_qubits(
                tuple(TWO_QUBIT_GATES), gate_qubit_count=2
            )
        ]


    return leakage_two_qubit_gate


def qubit_reset_leakage_noise_profile(pL: float) -> NoiseProfile:
    """
    Noise profile for leakage channels caused by
    reset gates.

    Parameters
    ----------
    pL: float
        Leakage channel probability parameter.

    Returns
    -------
    reset_leakage : NoiseProfile
        Leakage noise profile for reset gates.
    """

    def reset_leakage(noise_context):
        return [
            Leakage(qubit, pL)
            for qubit in noise_context.gate_layer_qubits(OneQubitResetGate)
        ]

    return reset_leakage


def two_qubit_gate_relaxation_noise_profile(pR: float) -> NoiseProfile:
    """
    Noise profile for relaxation channels caused by
    two qubit gates.

    Parameters
    ----------
    pR: float
        Relaxation channel probability parameter.

    Returns
    -------
    two_qubit_gate_relax : NoiseProfile
        Relaxation noise profile for two qubit gates.
    """

    def two_qubit_gate_relax(noise_context):
        return [
            Relax(qubit, pR)
            for qubit in noise_context.gate_layer_qubits(
                tuple(TWO_QUBIT_GATES), gate_qubit_count=2
            )
        ]

    return two_qubit_gate_relax


def one_qubit_clifford_gate_relaxation_noise_profile(pR: float) -> NoiseProfile:
    """
    Noise profile for relaxation channels caused by
    one qubit clifford gates.

    Parameters
    ----------
    pR: float
        Relaxation channel probability parameter.

    Returns
    -------
    one_qubit_relax : NoiseProfile
        Relaxation noise profile for ne qubit clifford gates.
    """

    def one_qubit_relax(noise_context):
        return [
            Relax(qubit, pR)
            for qubit in noise_context.gate_layer_qubits(
                (*tuple(TWO_QUBIT_GATES), OneQubitCliffordGate), gate_qubit_count=1
            )
        ]

    return one_qubit_relax


def idle_qubit_relaxation_noise_profile(pR: float) -> NoiseProfile:
    """
    Noise profile for relaxation channels on idle qubits.

    Parameters
    ----------
    pR: float
        Relaxation channel probability parameter.

    Returns
    -------
    idle_relax : NoiseProfile
        Relaxation noise profile for idle qubits.
    """

    def idle_relax(noise_context):
        idle_qubits = noise_context.circuit.qubits - set(
            noise_context.gate_layer_qubits(Gate)
        )
        return [Relax(qubit, pR) for qubit in idle_qubits]

    return idle_relax


def resonator_idle_qubit_relaxation_noise_profile(p: float, pR: float) -> NoiseProfile:
    """
    Noise profile for channels on resonator idle qubits. Adds
    relaxation and one qubit depolarising noise.

    Parameters
    ----------
    p: float
        One qubit depolarising channel probability parameter.
    pR: float
        Relaxation channel probability parameter.

    Returns
    -------
    resonator_idle : NoiseProfile
        Relaxation noise profile for idle qubits during active
        resonator operations.
    """
    depolarise1_generator_resonator_idle = Depolarise1.generator_from_prob(p)

    def resonator_idle(noise_context):
        noise_channels = []
        if resonator_qubits := set(
            noise_context.gate_layer_qubits(OneQubitMeasurementGate)
            + noise_context.gate_layer_qubits(OneQubitResetGate)
        ):
            idle_qubits = noise_context.circuit.qubits - resonator_qubits
            relaxations = [Relax(q, pR) for q in idle_qubits] if pR > 0 else []
            noise_channels = (
                depolarise1_generator_resonator_idle(idle_qubits) + relaxations
            )
        return noise_channels

    return resonator_idle
