# (c) Copyright Riverlane 2020-2025.
"""Factory functions for noise profiles"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, no_type_check

from deltakit_circuit._qubit_identifiers import Qubit, T
from deltakit_circuit.gates import Gate, OneQubitCliffordGate, TwoOperandGate, _Gate
from deltakit_circuit.gates._measurement_gates import (
    MPP,
    MRX,
    MRY,
    MRZ,
    MX,
    MY,
    MZ,
    _MeasurementGate,
)
from deltakit_circuit.gates._reset_gates import RX, RY, RZ
from deltakit_circuit.noise_channels import (
    Depolarise1,
    Depolarise2,
    PauliXError,
    PauliZError,
    _NoiseChannel,
)

if TYPE_CHECKING:  # pragma: no cover
    from deltakit_circuit._circuit import Circuit
    from deltakit_circuit._gate_layer import GateLayer


@dataclass
class NoiseContext:
    """Circuit data passed by deltakit_circuit.Circuit.apply_gate_noise to the
    noise generation functions. In guaranteeing that this object is
    passed through to noise generation functions, users have maximal
    access to the circuit's state and are able to formulate noise
    profiles for circuits that are not yet in scope"""

    circuit: Circuit
    gate_layer: GateLayer

    def gate_layer_qubits(
        self,
        gate_t: type[Gate] | tuple[type[Gate], ...] | None,
        gate_qubit_count: int | None = None,
    ) -> Sequence[Qubit]:
        """Returns all the qubits that belong to gates of type gate_t in
        self.gate_layer's gates with optional gate qubit count
        selector. If gate_t and gate_qubit_count are both None,
        all the qubits operated upon in self.gate_layer are returned.

        Parameters:
        -----------
        gate_t : Type[Gate] | Tuple[Type[Gate], ...] | None
            The gate type over which noise should be applied. If None, all
            the qubits operated upon in this gate_layer are returned.
        gate_qubit_count : Optional[int] = None
            Selector for gates which act on this amount of qubits.
            If None, gates may act on an arbitrary amount of qubits.

        Examples:
        ---------

        >>> import deltakit_circuit as sp
        >>> from deltakit_circuit._noise_factory import NoiseContext
        >>> gate_layer = sp.GateLayer(
        ...     [sp.gates.H(0), sp.gates.CX(1, 2), sp.gates.MX(4)]
        ... )
        >>> deltakit_circuit_circuit = sp.Circuit(gate_layer)
        >>> noise_context = NoiseContext(deltakit_circuit_circuit, gate_layer)
        >>> list(noise_context.gate_layer_qubits(sp.gates.H))
        [Qubit(0)]
        >>> list(noise_context.gate_layer_qubits((sp.gates.H, sp.gates.CX)))
        [Qubit(0), Qubit(1), Qubit(2)]
        >>> list(noise_context.gate_layer_qubits(None))
        [Qubit(0), Qubit(1), Qubit(2), Qubit(4)]
        """
        return tuple(
            chain.from_iterable(
                gate.qubits
                for gate in self.gate_layer.gates
                if (gate_t is None or isinstance(gate, gate_t))
                and (gate_qubit_count is None or gate_qubit_count == len(gate.qubits))
            )
        )


def noise_profile_with_inverted_noise(
    target_gate_t: type[Gate] | None = None,
    target_noise_generator: NoiseChannelGen | None = None,
    inverse_noise_generator: NoiseChannelGen | None = None,
) -> Callable[[NoiseContext], Sequence[_NoiseChannel]]:
    """Generate a noise profile for a given gate, with additional
    noise across all other qubits in the circuit that are not
    operated on by the given gate in a given gate layer

    Parameters
    ----------
    target_gate : Optional[Type[Gate]]
        The gate type over which noise should be applied. If None
        inverse_noise_generator will be projected over all qubits that
        are not acted upon in the given gate layer
    target_noise_generator : Optional[NoiseChannelGenT]
        A callable that can be used to construct a given noise channel type
        with a predetermined error probability (or set of probabilities)
        that should be applied to qubits acted upon by the target_gate.
        If None, no noise channels will be emitted on the target qubits.
    inverse_noise_generator : Optional[NoiseChannelGenT]
        A callable that can be used to construct a given noise channel type
        with a predetermined error probability (or set of probabilities) that
        should be applied to all qubits in the circuit that are not
        operated on by target_gate in this gate layer. If None, no noise
        channels will be emitted. If target_gate is None the callable
        will be projected over all qubits that are not acted upon in
        the given gate layer.

    Returns
    -------
    Callable[[NoiseContext], Sequence[_NoiseChannel]]:
        The noise profile that can be passed to Circuit.apply_gate_noise
        so as to apply a given error profile over target_gate and another
        on all other qubits in the circuit that are not operated on by
        target_gate in a given gate layer
    """

    def noise_channel_generator(noise_context: NoiseContext) -> Sequence[_NoiseChannel]:
        noise_channels: list[_NoiseChannel] = []
        if target_gate_t is None or target_gate_t in [
            type(gate) for gate in noise_context.gate_layer.gates
        ]:
            target_gate_qubits = list(noise_context.gate_layer_qubits(target_gate_t))
            inv_gate_qubits = [
                qubit
                for qubit in noise_context.circuit.qubits
                if qubit not in target_gate_qubits
            ]
            if target_noise_generator is not None:
                noise_channels.extend(target_noise_generator(target_gate_qubits))
            if inverse_noise_generator is not None:
                noise_channels.extend(inverse_noise_generator(inv_gate_qubits))
        return noise_channels

    return noise_channel_generator


def after_clifford_depolarisation(probability: float) -> list[NoiseProfile]:
    """Returns a set of callables that can be passed to
    deltakit_circuit.Circuit.apply_gate_noise to apply DEPOLARIZE1(probability)
    operations after every single-qubit Clifford gate and
    DEPOLARIZE2(probability) operations after every two-qubit Clifford
    operation
    """
    depolarise1_generator: NoiseChannelGen = Depolarise1.generator_from_prob(
        probability
    )
    depolarise2_generator: NoiseChannelGen = Depolarise2.generator_from_prob(
        probability
    )
    return [
        lambda noise_context: depolarise1_generator(
            noise_context.gate_layer_qubits(OneQubitCliffordGate)
        ),
        lambda noise_context: depolarise2_generator(
            tuple(noise_context.gate_layer_qubits(TwoOperandGate))
        ),
    ]


def before_measure_flip_probability(probability: float) -> list[NoiseProfile]:
    """Returns a set of callables that can be passed to
    deltakit_circuit.Circuit.apply_gate_noise to apply X_ERROR(probability)
    operations before every measurement gate that is not in the X basis
    and Z_ERROR(probability) operations on every measurement gate that
    is in the X basis
    """
    x_error_generator: NoiseChannelGen = PauliXError.generator_from_prob(probability)
    z_error_generator: NoiseChannelGen = PauliZError.generator_from_prob(probability)
    return [
        lambda noise_context: x_error_generator(
            noise_context.gate_layer_qubits((MZ, MRZ, MY, MRY))
        ),
        lambda noise_context: z_error_generator(
            noise_context.gate_layer_qubits((MX, MRX))
        ),
    ]


def after_reset_flip_probability(probability: float) -> list[NoiseProfile]:
    """Returns a set of callables that can be passed to
    deltakit_circuit.Circuit.apply_gate_noise to apply X_ERROR(probability)
    operations before every reset gate that is not in the X basis
    and Z_ERROR(probability) operations on every reset gate that
    is in the X basis
    """
    x_error_generator: NoiseChannelGen = PauliXError.generator_from_prob(probability)
    z_error_generator: NoiseChannelGen = PauliZError.generator_from_prob(probability)
    return [
        lambda noise_context: x_error_generator(
            noise_context.gate_layer_qubits((MRY, MRZ, RZ, RY))
        ),
        lambda noise_context: z_error_generator(
            noise_context.gate_layer_qubits((MRX, RX))
        ),
    ]


@no_type_check
def measurement_noise_profile(
    error_probability: float,
) -> dict[type[_MeasurementGate], Callable[[_MeasurementGate], _MeasurementGate]]:
    """Return a mapping from deltakit_circuit measurement types to the callables
    that can construct a gate of that type with a error probability as
    specified

    Parameters
    ----------
    error_probability : float
        The error probability that should be applied across all measurement
        gates

    Returns
    -------
    Dict[Type[MeasurementGateT],
         Callable[[MeasurementGateT], MeasurementGateT]]
        A noise profile. Typically this would they be passed to
        Circuit.replace_gate

    Examples
    --------
    >>> import stim
    >>> import deltakit_circuit as sp
    >>> stim_circuit = stim.Circuit.generated(
    ...     "surface_code:rotated_memory_z", rounds=3, distance=3
    ... )
    >>> deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(stim_circuit)
    >>> deltakit_circuit_circuit.replace_gates(measurement_noise_profile(0.2))
    """
    return {
        MX: lambda gate: gate.__class__(gate.qubit, error_probability),
        MY: lambda gate: gate.__class__(gate.qubit, error_probability),
        MZ: lambda gate: gate.__class__(gate.qubit, error_probability),
        MRX: lambda gate: gate.__class__(gate.qubit, error_probability),
        MRY: lambda gate: gate.__class__(gate.qubit, error_probability),
        MRZ: lambda gate: gate.__class__(gate.qubit, error_probability),
        MPP: lambda gate: gate.__class__(gate.pauli_product, error_probability),
    }


NoiseProfile = Callable[[NoiseContext], _NoiseChannel | Iterable[_NoiseChannel]]
NoiseChannelGen = Callable[[Sequence[Qubit | T]], Sequence[_NoiseChannel]]
GateReplacementPolicy = Mapping[type[_Gate] | _Gate, Callable[[_Gate], _Gate]]
