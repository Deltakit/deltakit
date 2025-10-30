# (c) Copyright Riverlane 2020-2025.
"""Module which gives the abstraction of a single time step in a circuit."""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import (
    Callable,
    DefaultDict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Mapping,
    Set,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import stim
from deltakit_circuit._qubit_mapping import default_qubit_mapping
from deltakit_circuit._stim_identifiers import AppendArguments
from deltakit_circuit.gates import MPP, _Gate, _MeasurementGate
from deltakit_circuit.gates._abstract_gates import (
    Gate,
    OneQubitCliffordGate,
    OneQubitMeasurementGate,
    OneQubitResetGate,
    TwoOperandGate,
)
from deltakit_circuit.gates._one_qubit_gates import _OneQubitCliffordGate
from deltakit_circuit.gates._reset_gates import _ResetGate
from deltakit_circuit.gates._two_qubit_gates import _TwoQubitGate
from deltakit_circuit._noise_factory import GateReplacementPolicy
from deltakit_circuit._qubit_identifiers import Qubit, T, U

_NonMeasurementGate = Union[_OneQubitCliffordGate, _ResetGate, _TwoQubitGate]


class DuplicateQubitError(ValueError):
    """Error class to represent qubits being used more than once in a layer."""

    def __init__(self, gate: Gate, intersection_qubits: Iterable[Qubit]):
        super().__init__(
            f"For gate {gate}: qubits {intersection_qubits} were "
            "identified as being duplicates in the layer."
        )


class GateLayer(Generic[T]):
    """Class which represents a single time step in a circuit for gates. A
    single qubit can be acted on at most one time in each layer."""

    def __init__(self, gates: _Gate | Iterable[_Gate] | None = None):
        self._qubits: Set[Qubit] = set()
        self._non_measurement_gates: List[_NonMeasurementGate] = []
        self._measurement_gates: List[_MeasurementGate] = []
        if gates is not None:
            self.add_gates(gates)

    @property
    def gates(self) -> Tuple[_Gate, ...]:
        """Get the set of gates in this layer."""
        return tuple(self._non_measurement_gates) + tuple(self._measurement_gates)

    @property
    def qubits(self) -> FrozenSet[Qubit[T]]:
        """Get all of the qubits in this gate layer."""
        return frozenset(self._qubits)

    @property
    def measurement_gates(self) -> Tuple[_MeasurementGate, ...]:
        """Get only the measurement gates in this layer."""
        return tuple(self._measurement_gates)

    def add_gates(self, gates: _Gate | Iterable[_Gate]):
        """Add a gate to the circuit layer. Must be a recognised stim gate.

        Parameters
        ----------
        gate : GateT | Iterable[GateT]
            The gate(s) to add to this circuit layer.

        Raises
        ------
        ValueError
            If the gate specified is not recognised.
        """

        gates = (gates,) if isinstance(gates, Gate) else list(gates)
        for gate in gates:
            qubits = gate.qubits
            if intersection := self._qubits.intersection(qubits):
                raise DuplicateQubitError(gate, intersection)
            self._qubits.update(qubits)
            if isinstance(gate, (OneQubitMeasurementGate, MPP)):
                self._measurement_gates.append(gate)
            else:
                self._non_measurement_gates.append(gate)

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        """
        Transform all gates in this gate layer according to the id mapping.
        No transformation is performed if the qubits id is not in the mapping.

        Parameters
        ----------
        id_mapping : Mapping[T, U]
            A mapping of qubit types to other qubit types
        """
        new_qubits: Set[Qubit] = set()
        for gate in chain(self._non_measurement_gates, self._measurement_gates):
            # Is maybe dangerous because the mutation happens regardless of
            # whether the error is raised or not
            gate.transform_qubits(id_mapping)
            if intersection := new_qubits.intersection(gate.qubits):
                raise DuplicateQubitError(gate, intersection)
            new_qubits.update(gate.qubits)
        self._qubits = new_qubits

    def _replace_measurement_instance(
        self, old_gate: _MeasurementGate, new_gate: _MeasurementGate
    ):
        self._qubits -= set(old_gate.qubits)
        if intersection := self._qubits.intersection(new_gate.qubits):
            raise DuplicateQubitError(new_gate, intersection)
        self._measurement_gates[self._measurement_gates.index(old_gate)] = new_gate
        self._qubits.update(new_gate.qubits)

    def _replace_non_measurement_instance(
        self, old_gate: _NonMeasurementGate, new_gate: _NonMeasurementGate
    ):
        self._non_measurement_gates.remove(old_gate)
        self._qubits -= set(old_gate.qubits)
        self.add_gates(new_gate)

    def _replace_all_measurement_types(
        self,
        gate_type: Type[_MeasurementGate],
        gate_generator: Callable[[_MeasurementGate], _MeasurementGate],
    ):
        for index, gate in enumerate(self._measurement_gates):
            if isinstance(gate, gate_type):
                new_gate = gate_generator(gate)
                self._qubits -= set(gate.qubits)
                if intersection := self._qubits.intersection(new_gate.qubits):
                    raise DuplicateQubitError(new_gate, intersection)
                self._measurement_gates[index] = new_gate
                self._qubits.update(new_gate.qubits)

    def _replace_all_non_measurement_types(
        self,
        gate_type: Type[_NonMeasurementGate],
        gate_generator: Callable[[_NonMeasurementGate], _NonMeasurementGate],
    ):
        new_gates = []
        for gate in self._non_measurement_gates.copy():
            if isinstance(gate, gate_type):
                self._non_measurement_gates.remove(gate)
                self._qubits -= set(gate.qubits)
                new_gates.append(gate_generator(gate))
        self.add_gates(new_gates)

    @no_type_check
    def replace_gates(self, replacement_policy: GateReplacementPolicy):
        """Replace all gates according to the gate replacement policy

        Replacing a measurement gate with a non measurement gate or a non
        measurement gate with a measurement gate will lead to unexpected
        behaviour

        Parameters
        ----------
        replacement_policy : Mapping[Type[GateT] | GateT,
                                     Callable[[GateT], GateT]]
            A mapping from gates or gate types to be replaced in this gate
            layer to the callables that return the new gate.

            If the key in the mapping is a type, all gates of that type
            will be replaced according to the associated callble. Otherwise,
            if the key is an instance of a particular gate, all gates that
            are equal to that particular gate will be changed according to the
            associated callable.
        """
        for gate, gate_generator in replacement_policy.items():
            if isinstance(gate, (OneQubitMeasurementGate, MPP)):
                if gate in self._measurement_gates:
                    self._replace_measurement_instance(gate, gate_generator(gate))
            elif isinstance(
                gate, (OneQubitCliffordGate, OneQubitResetGate, TwoOperandGate)
            ):
                if gate in self._non_measurement_gates:
                    self._replace_non_measurement_instance(gate, gate_generator(gate))
            elif issubclass(gate, (OneQubitMeasurementGate, MPP)):
                self._replace_all_measurement_types(gate, gate_generator)
            elif issubclass(
                gate, (OneQubitCliffordGate, OneQubitResetGate, TwoOperandGate)
            ):
                self._replace_all_non_measurement_types(gate, gate_generator)

    def _collect_gates(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> List[AppendArguments]:
        """Collect all of the same gate types together."""
        gate_args = []
        unordered_gates: DefaultDict[
            Type[_NonMeasurementGate], List[stim.GateTarget]
        ] = defaultdict(list)
        for non_measurement_gate in self._non_measurement_gates:
            unordered_gates[non_measurement_gate.__class__].extend(
                non_measurement_gate.stim_targets(qubit_mapping)
            )
        for unordered_gate, targets in unordered_gates.items():
            gate_args.append(
                AppendArguments(unordered_gate.stim_string, tuple(targets), (0,))
            )
        for measurement_gate in self._measurement_gates:
            gate_args.append(
                AppendArguments(
                    measurement_gate.stim_string,
                    measurement_gate.stim_targets(qubit_mapping),
                    (measurement_gate.probability,),
                )
            )
        return gate_args

    def permute_stim_circuit(
        self,
        stim_circuit: stim.Circuit,
        qubit_mapping: Mapping[Qubit[T], int] | None = None,
    ):
        """Updates stim_circuit with the stim circuit which contains the gates
        specified in this GateLayer.

        Parameters
        ----------
        stim_circuit : stim.Circuit
            The stim circuit to be updated with the stim representation of
            this gate layer

        qubit_mapping : Mapping[Qubit[T], int] | None, optional
            A mapping between each qubit in this layer and an integer which is
            necessary for outputting a stim circuit. By default None which
            means a default mapping is used.
        """
        qubit_mapping = (
            default_qubit_mapping(self.qubits)
            if qubit_mapping is None
            else qubit_mapping
        )
        for gate_string, targets, error_probability in self._collect_gates(
            qubit_mapping
        ):
            if error_probability == (0,):
                stim_circuit.append(gate_string, targets)
            else:
                stim_circuit.append(gate_string, targets, error_probability)

    def approx_equals(
        self,
        other: object,
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 0,
    ) -> bool:
        """Determine whether two gate layers are approximately equal
        within a tolerance. The tolerance accounts for small differences
        in the error probabilities of measurement gates. All other gates
        and properties must be equal.

        Parameters
        ----------
        other : object
            The other object to which to compare this gate layer.
        rel_tol : float
            The allowed relative difference between the error probabilities
            of two measurement gates, if this is larger than that calculated
            from abs_tol. Has the same meaning as in math.isclose.
            By default, 1e-9.
        abs_tol : float, optional
            The allowed absolute difference between the error probabilities
            of two measurement gates, if this is larger than that calculated
            from rel_tol. Has the same meaning as in math.isclose.
            By default, 0.0.

        Returns
        -------
        bool
            Whether the two gate layers are approximately equal.
        """
        # pylint: disable=protected-access
        return (
            isinstance(other, GateLayer)
            and set(self._non_measurement_gates) == set(other._non_measurement_gates)
            and len(self._measurement_gates) == len(other._measurement_gates)
            and all(
                self_gate.approx_equals(other_gate, rel_tol=rel_tol, abs_tol=abs_tol)
                for self_gate, other_gate in zip(
                    self._measurement_gates, other._measurement_gates, strict=True
                )
            )
        )

    def __hash__(self):
        return hash(
            (
                tuple(self._qubits),
                tuple(self._non_measurement_gates),
                tuple(self._measurement_gates),
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GateLayer)
            and set(self._non_measurement_gates) == set(other._non_measurement_gates)
            and self._measurement_gates == other._measurement_gates
        )

    def __repr__(self) -> str:
        indent = 4 * " "
        gate_layer_lines = ["GateLayer(["]
        gate_layer_lines.extend(f"{indent}{repr(gate)}" for gate in self.gates)
        gate_layer_lines.append("])")
        return "\n".join(gate_layer_lines)
