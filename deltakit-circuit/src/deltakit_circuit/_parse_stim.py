# (c) Copyright Riverlane 2020-2025.
"""Module which provides methods for parsing stim circuits into deltakit_circuit
circuits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import tee, zip_longest
from typing import TypeVar, cast

import stim

from deltakit_circuit._annotations._detector import Detector, MeasurementRecord
from deltakit_circuit._annotations._observable import Observable
from deltakit_circuit._annotations._shift_coordinates import ShiftCoordinates
from deltakit_circuit._gate_layer import GateLayer
from deltakit_circuit._noise_layer import NoiseLayer
from deltakit_circuit._qubit_identifiers import (
    Coordinate,
    InvertiblePauliX,
    InvertiblePauliY,
    InvertiblePauliZ,
    MeasurementPauliProduct,
    PauliProduct,
    PauliX,
    PauliY,
    PauliZ,
    Qubit,
    SweepBit,
    _InvertiblePauliGate,
    _PauliGate,
)
from deltakit_circuit.gates import GATE_MAPPING, OneQubitResetGate, _Gate
from deltakit_circuit.gates._abstract_gates import (
    OneQubitCliffordGate,
    OneQubitMeasurementGate,
    TwoOperandGate,
)
from deltakit_circuit.gates._measurement_gates import MPP, _OneQubitMeasurementGate
from deltakit_circuit.gates._one_qubit_gates import _OneQubitCliffordGate
from deltakit_circuit.gates._reset_gates import _ResetGate
from deltakit_circuit.gates._two_qubit_gates import _TwoQubitGate
from deltakit_circuit.noise_channels import (
    NOISE_CHANNEL_MAPPING,
    CorrelatedError,
    Depolarise1,
    Depolarise2,
    ElseCorrelatedError,
    Leakage,
    PauliChannel1,
    PauliChannel2,
    PauliXError,
    PauliYError,
    PauliZError,
    Relax,
    _NoiseChannel,
)


class InstructionNotImplemented(NotImplementedError):
    """Derived error class for stim instructions which cannot be parsed by
    deltakit_circuit yet."""

    def __init__(
        self, instruction: stim.CircuitInstruction | stim.CircuitRepeatBlock
    ) -> None:
        super().__init__(f"Parsing of '{instruction}' is not implemented yet.")


def _classify_pauli_target(
    target: stim.GateTarget, qubit_mapping: Mapping[int, Qubit]
) -> _PauliGate | _InvertiblePauliGate:
    qubit = qubit_mapping.get(target.value, Qubit(target.value))
    if target.is_x_target:
        if target.is_inverted_result_target:
            return InvertiblePauliX(qubit, invert=True)
        return PauliX(qubit)
    if target.is_y_target:
        if target.is_inverted_result_target:
            return InvertiblePauliY(qubit, invert=True)
        return PauliY(qubit)
    if target.is_z_target:
        if target.is_inverted_result_target:
            return InvertiblePauliZ(qubit, invert=True)
        return PauliZ(qubit)
    msg = f"Target: {target} is not a Pauli gate target."
    raise ValueError(msg)


def _parse_single_qubit_gate_instruction(
    gate_class: type[_OneQubitCliffordGate | _ResetGate],
    instruction_targets: Iterable[stim.GateTarget],
    instruction_tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> list[GateLayer]:
    qubits = (
        qubit_mapping.get(target.value, Qubit(target.value))
        for target in instruction_targets
    )
    time_steps = group_targets(qubit for qubit in qubits)
    return [
        GateLayer(gate_class(qubit, tag=instruction_tag) for qubit in time_step)
        for time_step in time_steps
    ]


def _parse_two_qubit_gate_instruction(
    gate_class: type[_TwoQubitGate],
    instruction_targets: Sequence[stim.GateTarget],
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> GateLayer:
    targets: list[Qubit | SweepBit | MeasurementRecord] = []
    for target in instruction_targets:
        if target.is_sweep_bit_target:
            targets.append(SweepBit(target.value))
        elif target.is_measurement_record_target:
            targets.append(MeasurementRecord(target.value))
        else:
            targets.append(qubit_mapping.get(target.value, Qubit(target.value)))
    return GateLayer(gate_class.from_consecutive(targets, tag=tag))


def _parse_single_qubit_measurement(
    gate_class: type[_OneQubitMeasurementGate],
    instruction_targets: Iterable[stim.GateTarget],
    instruction_arguments: Iterable[float],
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> GateLayer:
    probability = next(iter(instruction_arguments), 0.0)
    return GateLayer(
        gate_class(
            qubit_mapping.get(target.value, Qubit(target.value)),
            probability,
            invert=target.is_inverted_result_target,
            tag=tag,
        )
        for target in instruction_targets
    )


def _parse_mpp_instruction(
    instruction_targets: Sequence[stim.GateTarget],
    instruction_arguments: Iterable[float],
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> GateLayer:
    """Function for parsing a single MPP instruction. This algorithm is
    particularly complicated because the MPP instruction can have multiple
    different gate targets as input but the algorithm is outlined as such:

    For each target in the gate targets:
        * Check if the target is a stim combiner (*) and if it is go to the
          next target.
        * If it's not a combiner convert the stim target into a deltakit_circuit qubit
          identifier. Since MPP gates have Pauli string targets this will
          return a single Pauli gate.
        * Add the deltakit_circuit Pauli gate to a list of Pauli gates.
        * If the target after the current target is a combiner that means the
          current target is part of a Pauli product rather than a single Pauli
          so go to the next target (which will be the combiner).
        * If the target after the current target is not a combiner then all
          items in the list of Pauli gates completes the Pauli string. If the
          length of the Pauli gates list is 1 then this is just a single gate
          and not a product. Otherwise make a Pauli product from the list of
          Pauli gates and add it to the gate layer.
    """
    probability = next(iter(instruction_arguments), 0.0)

    pauli_gates: list[_PauliGate | _InvertiblePauliGate] = []
    qubit_identifiers: list[
        _PauliGate | _InvertiblePauliGate | MeasurementPauliProduct
    ] = []
    current_targets, peak_targets = tee(instruction_targets)
    next(peak_targets)
    for current_target, next_target in zip_longest(current_targets, peak_targets):
        if current_target.is_combiner:
            continue
        pauli_gate = _classify_pauli_target(current_target, qubit_mapping)
        pauli_gates.append(pauli_gate)
        if next_target is None or not next_target.is_combiner:
            if len(pauli_gates) == 1:
                qubit_identifiers.append(pauli_gate)
            else:
                qubit_identifiers.append(MeasurementPauliProduct(pauli_gates))
            pauli_gates = []
    return GateLayer(
        MPP(qubit_id, probability, tag=tag) for qubit_id in qubit_identifiers
    )


def _parse_single_qubit_noise_instruction(
    noise_class: type[
        PauliXError | PauliYError | PauliZError | Depolarise1 | Leakage | Relax
    ],
    instruction_targets: Iterable[stim.GateTarget],
    probability: float,
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> NoiseLayer:
    qubits = (
        qubit_mapping.get(target.value, Qubit(target.value))
        for target in instruction_targets
    )
    return NoiseLayer(noise_class(qubit, probability, tag=tag) for qubit in qubits)


def _parse_depolarise_2_noise_instruction(
    instruction_targets: Sequence[stim.GateTarget],
    probability: float,
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> NoiseLayer:
    qubits = [
        qubit_mapping.get(target.value, Qubit(target.value))
        for target in instruction_targets
    ]
    return NoiseLayer(Depolarise2.from_consecutive(qubits, probability, tag=tag))


def _parse_pauli_channel_1_instruction(
    instruction_targets: Iterable[stim.GateTarget],
    probabilities: Iterable[float],
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> NoiseLayer:
    qubits = (
        qubit_mapping.get(target.value, Qubit(target.value))
        for target in instruction_targets
    )
    return NoiseLayer(PauliChannel1(qubit, *probabilities, tag=tag) for qubit in qubits)


def _parse_pauli_channel_2_instruction(
    instruction_targets: Iterable[stim.GateTarget],
    probabilities: Iterable[float],
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> NoiseLayer:
    qubits = [
        qubit_mapping.get(target.value, Qubit(target.value))
        for target in instruction_targets
    ]
    return NoiseLayer(PauliChannel2.from_consecutive(qubits, *probabilities, tag=tag))


def _parse_correlated_error_instruction(
    noise_class: type[CorrelatedError | ElseCorrelatedError],
    instruction_targets: Iterable[stim.GateTarget],
    probability: float,
    tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> NoiseLayer:
    pauli_product: PauliProduct = PauliProduct(
        cast(_PauliGate, _classify_pauli_target(target, qubit_mapping))
        for target in instruction_targets
    )
    return NoiseLayer(noise_class(pauli_product, probability, tag=tag))


def parse_stim_gate_instruction(
    deltakit_circuit_gate_class: type[_Gate],
    instruction_targets: Sequence[stim.GateTarget],
    instruction_arguments: Sequence[float],
    instruction_tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> GateLayer | Iterable[GateLayer]:
    """Parse a single instruction which is a gate into a gate layer.

    Parameters
    ----------
    deltakit_circuit_gate_class : Type[GateT]
        The type of gate which defined this gate instruction.
    instruction_targets : Sequence[stim.GateTarget]
        The stim instruction targets to act the gate on.
    instruction_arguments : Sequence[float]
        The additional arguments to give to the gate. These are commonly just
        the probabilities for non-deterministic gates.
    instruction_tag : (str | None)
        Tag associated with the instruction. None if no tag was present.
    qubit_mapping : Mapping[int, Qubit]
        A mapping for qubits where coordinates have been specified
        in stim. The mapping maps each qubit's index to the respective
        object of type Qubit.

    Returns
    -------
    GateLayer | Iterable[GateLayer]
        A gate layer containing the gate in the gate instruction.

    Raises
    ------
    ValueError
        If the given gate is not a recognised deltakit_circuit gate class.
    """
    if issubclass(
        deltakit_circuit_gate_class, (OneQubitCliffordGate, OneQubitResetGate)
    ):
        return _parse_single_qubit_gate_instruction(
            deltakit_circuit_gate_class,
            instruction_targets,
            instruction_tag,
            qubit_mapping,
        )
    if issubclass(deltakit_circuit_gate_class, TwoOperandGate):
        return _parse_two_qubit_gate_instruction(
            deltakit_circuit_gate_class,
            instruction_targets,
            instruction_tag,
            qubit_mapping,
        )
    if issubclass(deltakit_circuit_gate_class, OneQubitMeasurementGate):
        return _parse_single_qubit_measurement(
            deltakit_circuit_gate_class,
            instruction_targets,
            instruction_arguments,
            instruction_tag,
            qubit_mapping,
        )
    if issubclass(deltakit_circuit_gate_class, MPP):
        return _parse_mpp_instruction(
            instruction_targets, instruction_arguments, instruction_tag, qubit_mapping
        )
    msg = (
        f"Given gate class: '{deltakit_circuit_gate_class}' is not a "
        "valid deltakit_circuit gate."
    )
    raise ValueError(msg)


def parse_stim_noise_instruction(
    deltakit_circuit_noise_class: type[_NoiseChannel],
    instruction_targets: Sequence[stim.GateTarget],
    instruction_arguments: Sequence[float],
    instruction_tag: str | None,
    qubit_mapping: Mapping[int, Qubit],
) -> NoiseLayer:
    """Parse a single instruction which is a noise into a NoiseLayer.

    Parameters
    ----------
    deltakit_circuit_noise_class : Type[_NoiseChannel]
        The type of noise channel which defines this instruction.
    instruction_targets : Sequence[stim.GateTarget]
        The stim instruction targets to act the noise channel on.
    instruction_arguments : Sequence[float]
        The probabilities which define the noise channel.
    instruction_tag : (str | None)
        Tag associated with the instruction. None if no tag was present.

    Returns
    -------
    NoiseLayer
        The noise layer containing the noise channel in the stim instruction.

    Raises
    ------
    ValueError
        If the type of noise is not a recognised deltakit_circuit noise channel.
    """
    if issubclass(
        deltakit_circuit_noise_class,
        (PauliXError, PauliYError, PauliZError, Depolarise1, Leakage, Relax),
    ):
        return _parse_single_qubit_noise_instruction(
            deltakit_circuit_noise_class,
            instruction_targets,
            instruction_arguments[0],
            instruction_tag,
            qubit_mapping,
        )
    if issubclass(deltakit_circuit_noise_class, Depolarise2):
        return _parse_depolarise_2_noise_instruction(
            instruction_targets,
            instruction_arguments[0],
            instruction_tag,
            qubit_mapping,
        )
    if issubclass(deltakit_circuit_noise_class, PauliChannel1):
        return _parse_pauli_channel_1_instruction(
            instruction_targets, instruction_arguments, instruction_tag, qubit_mapping
        )
    if issubclass(deltakit_circuit_noise_class, PauliChannel2):
        return _parse_pauli_channel_2_instruction(
            instruction_targets, instruction_arguments, instruction_tag, qubit_mapping
        )
    if issubclass(deltakit_circuit_noise_class, (CorrelatedError, ElseCorrelatedError)):
        return _parse_correlated_error_instruction(
            deltakit_circuit_noise_class,
            instruction_targets,
            instruction_arguments[0],
            instruction_tag,
            qubit_mapping,
        )
    msg = (
        f"Given noise class: '{deltakit_circuit_noise_class}' is not a "
        "valid deltakit_circuit noise channel."
    )
    raise ValueError(msg)


def parse_detector(instruction: stim.CircuitInstruction) -> Detector:
    """Parse the given circuit instruction into a deltakit_circuit detector.

    Parameters
    ----------
    instruction : stim.CircuitInstruction
        The input circuit instruction

    Returns
    -------
    Detector
        A single Detector with all measurement records from the instruction.
    """
    measurement_records = (
        MeasurementRecord(gate_target.value)
        for gate_target in instruction.targets_copy()
    )
    coords = instruction.gate_args_copy()
    tag = instruction.tag if hasattr(instruction, "tag") else ""
    return Detector(
        measurement_records, Coordinate(*coords) if coords != [] else None, tag=tag
    )


def parse_observable(instruction: stim.CircuitInstruction) -> Observable:
    """Parse the given circuit instruction into a deltakit_circuit observable.

    Parameters
    ----------
    instruction : stim.CircuitInstruction
        The input circuit instruction

    Returns
    -------
    Observable
        A single Observable with all measurement records from the instruction.
    """
    tag = instruction.tag if hasattr(instruction, "tag") else ""
    return Observable(
        int(instruction.gate_args_copy()[0]),
        (
            MeasurementRecord(gate_target.value)
            for gate_target in instruction.targets_copy()
        ),
        tag=tag,
    )


def parse_shift_coords(instruction: stim.CircuitInstruction) -> ShiftCoordinates:
    """Parse the given circuit instruction into a deltakit_circuit shift coordinates.

    Parameters
    ----------
    instruction : stim.CircuitInstruction
        The input circuit instruction

    Returns
    -------
    ShiftCoordinates
        A single shift coordinates which advances the detector coordinates.
    """
    tag = instruction.tag if hasattr(instruction, "tag") else ""
    return ShiftCoordinates(instruction.gate_args_copy(), tag=tag)


def parse_circuit_instruction(
    instruction: stim.CircuitInstruction, qubit_mapping: Mapping[int, Qubit]
) -> _Layer | Iterable[_Layer]:
    """Parse a single stim circuit instruction into a gate or noise layer.

    Parameters
    ----------
    instruction : stim.CircuitInstruction
        The circuit instruction to parse.
    qubit_mapping : Mapping[int, Qubit]
        A mapping for qubits where coordinates have been specified
        in stim. The mapping maps each qubit's index to the respective
        object of type Qubit.

    Returns
    -------
    _Layer | Iterable[_Layer]
        The deltakit_circuit representation of the circuit instruction as a layer.

    Raises
    ------
    InstructionNotImplemented
        If the instruction cannot be parsed yet.
    """
    instruction_name = instruction.name
    instruction_targets = instruction.targets_copy()
    instruction_arguments = instruction.gate_args_copy()
    instruction_tag = instruction.tag if hasattr(instruction, "tag") else None
    if (gate_class := GATE_MAPPING.get(instruction_name, None)) is not None:
        return parse_stim_gate_instruction(
            gate_class,
            instruction_targets,
            instruction_arguments,
            instruction_tag,
            qubit_mapping,
        )
    if (noise_class := NOISE_CHANNEL_MAPPING.get(instruction_name, None)) is not None:
        return parse_stim_noise_instruction(
            noise_class,
            instruction_targets,
            instruction_arguments,
            instruction_tag,
            qubit_mapping,
        )
    if instruction_name == "DETECTOR":
        return parse_detector(instruction)
    if instruction_name == "OBSERVABLE_INCLUDE":
        return parse_observable(instruction)
    if instruction_name == "SHIFT_COORDS":
        return parse_shift_coords(instruction)
    raise InstructionNotImplemented(instruction)


T = TypeVar("T", bound=Hashable)  # pylint: disable=invalid-name


def group_targets(targets: Iterable[T]) -> list[set[T]]:
    """Group an iterable of qubit indices into overlapping sets of duplicate
    elements. It does not preserve order and puts items into the "earliest"
    available set.

    Examples
    --------
    [0, 1, 2, 3, 4] -> [{0, 1, 2, 3, 4}]
    [0, 1, 2, 0, 2] -> [{0, 1, 2}, {0, 2}]
    [0, 1, 2, 0, 2, 2] -> [{0, 1, 2}, {0, 2}, {2}]
    [0, 1, 2, 0, 1, 0, 1, 2] -> [{0, 1, 2}, {0, 1, 2}, {0, 1}]

    Parameters
    ----------
    targets : Iterable[T]
        An iterable elements, some of which may be duplicated.

    Returns
    -------
    List[Set[T]]
        A list of overlapping sets separating duplicate elements.
    """
    target_counts = Counter(targets)
    number_of_groups = max(target_counts.values())
    grouped_targets: list[set[T]] = [set() for _ in range(number_of_groups)]
    for target, target_count in target_counts.items():
        for group in grouped_targets[0:target_count]:
            group.add(target)
    return grouped_targets


_Layer = GateLayer | NoiseLayer | Detector | Observable | ShiftCoordinates
