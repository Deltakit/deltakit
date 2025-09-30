# (c) Copyright Riverlane 2020-2025.
"""Module containing the circuit class."""

from __future__ import annotations

import collections.abc as cabc
from copy import deepcopy
from enum import IntEnum, auto
from itertools import chain
from typing import (
    Callable,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    get_args,
    no_type_check,
)

import stim
from deltakit_circuit._parse_stim import parse_circuit_instruction
from deltakit_circuit._qubit_mapping import default_qubit_mapping
from deltakit_circuit._annotations._detector import Detector
from deltakit_circuit._annotations._observable import Observable
from deltakit_circuit._annotations._shift_coordinates import ShiftCoordinates
from deltakit_circuit._gate_layer import GateLayer
from deltakit_circuit.gates._abstract_gates import OneQubitMeasurementGate
from deltakit_circuit.gates._measurement_gates import MPP, _MeasurementGate
from deltakit_circuit._noise_factory import (
    GateReplacementPolicy,
    NoiseContext,
    NoiseProfile,
)
from deltakit_circuit._noise_layer import NoiseLayer
from deltakit_circuit._qubit_identifiers import Coordinate, Qubit, T, U


class _Comparable(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol type for objects that can be compared."""

    def __lt__(self, __o: object) -> bool:  # pragma: no cover
        ...


class Circuit(Generic[T]):  # pylint: disable=too-many-public-methods
    """The deltakit_circuit circuit class. The circuit is implemented as a mutable list
    of layers where each layer is a gate layer, a noise layer, an
    annotation or another circuit.

    Parameters
    ----------
    layers: LayerT | Iterable[LayerT] | None, optional
        The layers to include in this circuit. Circuits are mutable so layers
        can be added to the circuit after construction. By default, None.
    iterations: int, optional
        The number of times this circuit is run. If this circuit specifies a
        repeat block then iterations should be greater than 1. By default, 1.
    """

    class LayerAdjacency(IntEnum):
        """Enum used in the apply_gate_noise to specify whether noise layers
        should be added before or after the gate layer.
        """

        BEFORE = auto()
        AFTER = auto()

    def __init__(
        self, layers: Layer | Iterable[Layer] | None = None, iterations: int = 1
    ):
        self._layers: List[Layer] = []
        self.iterations = iterations
        self._qubit_uid_type: Optional[type] = None
        if layers is not None:
            self.append_layers(layers)

    @property
    def qubit_uid_type(self) -> Optional[type]:
        """Get the type of the qubit unique identifiers used in this circuit.
        If the circuit has not been parsed yet this will be None.
        """
        if self._qubit_uid_type is None:
            self._qubit_uid_type = next(
                (type(qubit.unique_identifier) for qubit in self.qubits), None
            )
        return self._qubit_uid_type

    @property
    def layers(self) -> List[Layer]:
        """Get all the layers in this circuit. This property should not be
        modified directly and instead layers should be added using the
        append_layers method.
        """
        return self._layers

    @property
    def iterations(self) -> int:
        """Get the number of times this circuit is run."""
        return self._iterations

    @iterations.setter
    def iterations(self, number_of_iterations: int):
        """A setter for the number of iterations that specifies the number of
        times a circuit is to be run.

        Raises
        ------
        ValueError
            If number_of_iterations is 0 or a negative number. Stim does not
            allow this, stating that produces ambiguity as to whether
            observables and qubits that are mentioned in the block "exist".
        """
        if number_of_iterations < 1:
            raise ValueError(
                "Stim does not allow repeat blocks to be repeated less than "
                f"1 time. Requested repeats was {number_of_iterations}."
            )
        self._iterations = number_of_iterations

    @property
    def is_noisy(self) -> bool:
        """Whether this circuit contains any noise layers or measurement
        gates that are marked as noisy."""
        for layer in self._layers:
            if (
                isinstance(layer, NoiseLayer)
                or isinstance(layer, Circuit)
                and layer.is_noisy
            ):
                return True
        return any(gate.probability > 0 for gate in self.measurement_gates)

    def noise_layers(self, include_nested: bool = True) -> List[NoiseLayer[T]]:
        """All the noise layers in this circuit."""
        noise_layers = []
        for layer in self._layers:
            if isinstance(layer, NoiseLayer):
                noise_layers.append(layer)
            elif isinstance(layer, Circuit) and include_nested:
                noise_layers.extend(layer.noise_layers(include_nested))
        return noise_layers

    def gate_layers(self, include_nested: bool = True) -> List[GateLayer[T]]:
        """All the gate layers in this circuit."""
        gate_layers = []
        for layer in self._layers:
            if isinstance(layer, GateLayer):
                gate_layers.append(layer)
            elif isinstance(layer, Circuit) and include_nested:
                gate_layers.extend(layer.gate_layers(include_nested))
        return gate_layers

    def detectors(self, include_nested: bool = True) -> List[Detector]:
        """All of the detectors in this circuit."""
        detectors = []
        for layer in self._layers:
            if isinstance(layer, Detector):
                detectors.append(layer)
            elif isinstance(layer, Circuit) and include_nested:
                detectors.extend(layer.detectors(include_nested))
        return detectors

    @property
    def measurement_gates(self) -> Tuple[_MeasurementGate, ...]:
        """Get the ordered measurement gates in this circuit."""
        return tuple(
            chain.from_iterable(
                layer.measurement_gates
                for layer in self.layers
                if isinstance(layer, (GateLayer, Circuit))
            )
        )

    @property
    def qubits(self) -> FrozenSet[Qubit[T]]:
        """Get all the qubits that are used in this circuit and all nested
        circuits."""
        qubits: Set[Qubit[T]] = set()
        for layer in self._layers:
            if not isinstance(layer, (Detector, Observable, ShiftCoordinates)):
                qubits.update(layer.qubits)
        return frozenset(qubits)

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        """
        Transform this circuit's qubits and detectors according to the id
        mapping. No transformation is performed if the qubit's id is not in
        the mapping.

        Parameters
        ----------
        id_mapping : Mapping[T, U]
            A mapping of qubit types to other qubit types
        """
        if len(id_mapping.values()) != len(set(id_mapping.values())):
            raise ValueError(
                "The ID mapping is not bijective, all values must be unique."
            )
        for layer in self._layers:
            if isinstance(layer, (GateLayer, NoiseLayer, Circuit)):
                layer.transform_qubits(id_mapping)
            elif isinstance(layer, Detector):
                layer.transform_coordinates(id_mapping)

    def append_layers(self, layers: Layer | Iterable[Layer]):
        """Append any layers to the end of this circuit. If a layer consisting
        of a circuit with a single iteration is added to this circuit, it is
        flattened, i.e. all of the layers of that circuit are appended into
        this circuit. If the nested circuit has more than one iteration then no
        flattening occurs and this is analogous to inserting a repeat block
        into stim.

        Please note that this append operation works by reference, the layers
        input into the circuit are not copied.

        Parameters
        ----------
        layers : LayerT | Iterable[LayerT]
            The layer or layers to append.

        Raises
        ------------
        TypeError
            When types of qubit UIDs are not homogeneous.
        ValueError
            When a layer is not of type LayerT

        Examples
        --------
        >>> import deltakit_circuit as sp
        >>> circuit = sp.Circuit()
        >>> nested_circuit = sp.Circuit(sp.GateLayer(sp.gates.X(0)))
        >>> circuit.append_layers(nested_circuit)
        >>> circuit.layers
        [GateLayer([
            X(Qubit(0))
        ])]
        """
        if isinstance(layers, cabc.Generator):
            layers = tuple(layers)
        elif not isinstance(layers, cabc.Iterable):
            layers = (layers,)
        for layer in layers:
            if not isinstance(layer, tuple(LAYERS)):
                raise ValueError(f"Layer type is not one of {LAYERS}")
            if isinstance(layer, Circuit) and layer.iterations == 1:
                self._layers.extend(layer.layers)
            else:
                self._layers.append(layer)

            if hasattr(layer, "qubits"):
                if self.qubit_uid_type is not None:
                    if not all(
                        isinstance(qubit.unique_identifier, self.qubit_uid_type)
                        for qubit in layer.qubits
                    ):
                        raise TypeError(
                            "All Qubit._unique_identifier fields "
                            "must be of the same type"
                        )

    def apply_gate_noise(
        self,
        noise_profile: NoiseProfile | Iterable[NoiseProfile],
        adjacency: LayerAdjacency,
        recursive: bool = True,
    ):
        """Apply gate noise according to the noise profile. This will add a
        noise layer into the circuit either before or after a gate layer if
        there's a gate in the gate layer which is also in the noise profile.
        If no gates are found in the gate layer which are in the noise profile
        no noise layers are added for that gate layer.

        Parameters
        ----------
        noise_profile : Mapping[Type[GateT],\
                                Callable[[GateT], \
                                         NoiseChannelT | \
                                         Iterable[NoiseChannelT]]]
            A mapping of gate type to the noise channel to apply. Every time
            the gate is found in a gate layer the noise channel is generated
            and added to the noise layer.
        adjacency : LayerAdjacency
            Whether the gate noise should be applied before or after gates.
        recursive : bool, optional
            Whether to recurse into nested circuits; by default, True.

        Examples
        --------
        >>> import deltakit_circuit as sp
        >>> circuit = sp.Circuit(sp.GateLayer(sp.gates.X(0)))
        >>> circuit.apply_gate_noise(
        ...     lambda noise_context: [
        ...         sp.noise_channels.PauliXError(qubit, 0.1) for qubit in
        ...         noise_context.gate_layer_qubits(sp.gates.X)],
        ...     sp.Circuit.LayerAdjacency.AFTER)
        >>> circuit.layers
        [GateLayer([
            X(Qubit(0))
        ]), NoiseLayer([
            X_ERROR(Qubit(0), probability=0.1)
        ])]

        >>> circuit = sp.Circuit(sp.GateLayer(sp.gates.X(0)))
        >>> circuit.apply_gate_noise(
        ...     lambda noise_context: [
        ...         sp.noise_channels.PauliXError(qubit, 0.1) for qubit in
        ...         noise_context.gate_layer_qubits(sp.gates.X)],
        ...     sp.Circuit.LayerAdjacency.BEFORE)
        >>> circuit.layers
        [NoiseLayer([
            X_ERROR(Qubit(0), probability=0.1)
        ]), GateLayer([
            X(Qubit(0))
        ])]
        """
        noise_profile = (
            [noise_profile]
            if not isinstance(noise_profile, Iterable)
            else noise_profile
        )
        new_layers: List[Layer] = []
        for layer in self._layers:
            if isinstance(layer, GateLayer):
                noise_layer: NoiseLayer = NoiseLayer()
                for noise_generator in noise_profile:
                    noise_layer.add_noise_channels(
                        noise_generator(NoiseContext(self, layer))
                    )
                if len(noise_layer.noise_channels) > 0:
                    if adjacency is self.LayerAdjacency.BEFORE:
                        new_layers.extend((noise_layer, layer))
                    else:
                        new_layers.extend((layer, noise_layer))
                else:
                    new_layers.append(layer)
            elif isinstance(layer, Circuit) and recursive:
                layer.apply_gate_noise(noise_profile, adjacency, recursive)
                new_layers.append(layer)
            else:
                new_layers.append(layer)
        self._layers = new_layers

    def replace_gates(
        self, replacement_policy: GateReplacementPolicy, recursive: bool = True
    ):
        """Replace gates in the circuit with other gates. This can either be
        replacing all gates of the same type or all particular instances of a
        gate. In particular this can be used to apply measurement noise to the
        measurement gates by specifying the keys in the replacement policy to
        be measurement gates.

        Replacing a measurement gate with a non measurement gate or a non
        measurement gate with a measurement gate will lead to unexpected
        behaviour

        Parameters
        ----------
        replacement_policy : GateReplacementPolicy
            A mapping of gates to function which generates a new gate. Every
            time the gate key is found in a gate layer the gate in the gate
            layer is replaced with the new generated one. The keys can either
            be types of gate, in which case all gates of that type will be
            replaced or it can be an instance of a gate and then all gates
            which are equal to that gate will be replaced.
        recursive : bool, optional
            Whether to apply the replacement policy to nested circuits, by
            default True.

        Examples
        --------
        >>> import deltakit_circuit as sp
        >>> circuit = sp.Circuit(sp.GateLayer(sp.gates.MX(0)))
        >>> circuit.replace_gates(
        ...     {sp.gates.MX: lambda gate: sp.gates.MX(gate.qubit, 0.01)}
        ... )
        >>> circuit.gate_layers
        [GateLayer([
            MX(Qubit(0), probability=0.01)
        ])]

        You can also use this to manipulate the non-zero probabilities of
        existing measurement gates.

        >>> circuit = sp.Circuit(sp.GateLayer(sp.gates.MZ(0, 0.001)))
        >>> circuit.replace_gates(
        ...     {
        ...         sp.gates.MZ: lambda gate: sp.gates.MZ(
        ...             gate.qubit, gate.probability * 10
        ...         )
        ...     }
        ... )
        >>> circuit.gate_layers
        [GateLayer([
            MZ(Qubit(0), probability=0.01)
        ])]
        """
        for layer in self._layers:
            if isinstance(layer, GateLayer):
                layer.replace_gates(replacement_policy)
            elif isinstance(layer, Circuit) and recursive:
                layer.replace_gates(replacement_policy, recursive)

    def remove_noise(self, recursive: bool = True):
        """Remove all noise layers from this circuit and make any noisy
        measurement gates noiseless.

        Parameters
        ----------
        recursive: bool, optional
            Whether to remove noise from any nested circuits; by default, True.
        """
        self._layers = [
            layer for layer in self.layers if not isinstance(layer, NoiseLayer)
        ]
        for layer in self.layers:
            if isinstance(layer, GateLayer):
                for gate in layer.gates:
                    if isinstance(gate, OneQubitMeasurementGate):
                        layer.replace_gates(
                            {
                                gate: lambda gate: type(gate)(gate.qubit)
                            }
                        )
                    elif isinstance(gate, MPP):
                        layer.replace_gates(
                            {
                                gate: lambda gate: type(gate)(gate.pauli_product)
                            }
                        )
            elif isinstance(layer, Circuit) and recursive:
                layer.remove_noise(recursive)

    @no_type_check
    def reorder_detectors(
        self,
        /,
        key: Callable[[Detector], _Comparable] | None = None,
        reverse: bool = False,
    ):
        """Reorder the detectors in this circuit according to the given key.
        Detectors are re-arranged within within certain "blocks" such that the
        logical behaviour is the same. The blocks are defined by gate layers
        which contain measurements and nested circuits.

        Parameters
        ----------
        key : Callable[[Detector], _Comparable] | None, optional
            The key to sort the detectors by, by default None which will
            use the lexical order of the coordinate.
            If passing a callable remember that some detectors might not have
            a coordinate so there should be a check in the callable for that.
        reverse : bool
            Whether to reverse the sorting provided by key. This argument is
            passed directly to the ``sorted`` function, by default False.
        """

        def lexical_order(detector: Detector) -> Coordinate | Literal[0]:
            return coordinate if (coordinate := detector.coordinate) is not None else 0

        sort_key = lexical_order if key is None else key

        sorted_layers: List[Layer] = []
        unsorted_detectors: List[Detector] = []
        for layer in self._layers:
            if isinstance(layer, Detector):
                unsorted_detectors.append(layer)
                continue
            if (
                isinstance(layer, GateLayer) and len(layer.measurement_gates) > 0
            ) or isinstance(layer, ShiftCoordinates):
                sorted_layers.extend(
                    sorted(unsorted_detectors, key=sort_key, reverse=reverse)
                )
                unsorted_detectors = []
            elif isinstance(layer, Circuit):
                sorted_layers.extend(
                    sorted(unsorted_detectors, key=sort_key, reverse=reverse)
                )
                unsorted_detectors = []
                layer.reorder_detectors(key=sort_key, reverse=reverse)
            sorted_layers.append(layer)
        sorted_layers.extend(sorted(unsorted_detectors, key=sort_key, reverse=reverse))
        self._layers = sorted_layers

    def as_stim_circuit(
        self, qubit_mapping: Mapping[Qubit[T], int] | None = None
    ) -> stim.Circuit:
        """Get the equivalent Stim circuit from this deltakit_circuit circuit.

        Parameters
        ----------
        qubit_mapping: Mapping[Qubit[T], int] | None, optional
            A way to associate an integer with every qubit type. This is
            necessary because Stim represents qubits as single integers. This
            mapping should be injective. If not specified, deltakit_circuit will try to
            create a mapping from the qubits specified.

        Returns
        -------
        stim.Circuit
            The equivalent Stim circuit.

        Examples
        --------
        >>> import deltakit_circuit as sp
        >>> circuit = sp.Circuit(sp.GateLayer([sp.gates.X(0), sp.gates.Y(1)]))
        >>> circuit.as_stim_circuit()
        stim.Circuit('''
            X 0
            Y 1
        ''')
        >>> circuit.as_stim_circuit(qubit_mapping={Qubit(0): 1, Qubit(1): 0})
        stim.Circuit('''
            X 1
            Y 0
        ''')
        """
        stim_circuit = stim.Circuit()
        self.permute_stim_circuit(stim_circuit, qubit_mapping)
        return stim_circuit

    def permute_stim_circuit(
        self,
        stim_circuit: stim.Circuit,
        qubit_mapping: Mapping[Qubit[T], int] | None = None,
    ):
        """Convert this circuit to a Stim circuit and append it to the end of
        the given circuit. The helper method `as_stim_circuit` should be used
        to create a new Stim circuit.

        Parameters
        ----------
        stim_circuit : stim.Circuit
            The Stim circuit to append this circuit to.
        qubit_mapping: Mapping[Qubit[T], int] | None, optional
            A way to associate an integer for every qubit type. This is
            necessary because Stim represents qubits as single integers. This
            mapping should be injective. If not specified, deltakit_circuit will try to
            create a mapping from the qubits specified.
        """
        if qubit_mapping is None:
            qubit_mapping = default_qubit_mapping(self.qubits)
        inner_stim_circuit = stim.Circuit()
        if self.iterations == 1:
            for qubit in self.qubits:
                qubit.permute_stim_circuit(inner_stim_circuit, qubit_mapping)
        gate_layers = self.gate_layers()
        last_gate_layer = gate_layers[-1] if gate_layers else None
        for layer in self.layers:
            layer.permute_stim_circuit(inner_stim_circuit, qubit_mapping)
            # Append a tick after every gate layer but not after the last one
            if isinstance(layer, GateLayer) and layer is not last_gate_layer:
                inner_stim_circuit.append("TICK")
        stim_circuit += self.iterations * inner_stim_circuit

    def as_detector_error_model(  # noqa: PLR0913
        self,
        decompose_errors: bool = False,
        flatten_loops: bool = False,
        allow_gauge_detectors: bool = False,
        approximate_disjoint_errors: bool | float = False,
        ignore_decomposition_failures: bool = False,
        block_decomposition_from_introducing_remnant_edges: bool = False,
    ) -> stim.DetectorErrorModel:
        # pylint: disable=line-too-long,too-many-arguments
        """Returns a stim.DetectorErrorModel describing the error process
        in the circuit. Arguments are described here:
        https://github.com/quantumlib/Stim/blob/main/doc/
        python_api_reference_vDev.md#stim.Circuit.detector_error_model
        """
        if isinstance(approximate_disjoint_errors, float):
            if not 0 <= approximate_disjoint_errors <= 1:
                raise ValueError(
                    "approximate_disjoint_errors is not a valid probability"
                )

        return self.as_stim_circuit().detector_error_model(
            decompose_errors=decompose_errors,
            flatten_loops=flatten_loops,
            allow_gauge_detectors=allow_gauge_detectors,
            approximate_disjoint_errors=approximate_disjoint_errors,
            ignore_decomposition_failures=ignore_decomposition_failures,
            block_decomposition_from_introducing_remnant_edges=block_decomposition_from_introducing_remnant_edges,
        )

    @classmethod
    def from_stim_circuit(
        cls,
        stim_circuit: stim.Circuit,
        qubit_mapping: Mapping[int, Qubit[Coordinate]] | None = None,
    ) -> Circuit:
        """Parse a Stim circuit into a deltakit_circuit circuit.

        Parameters
        ----------
        stim_circuit : stim.Circuit
            The Stim circuit to convert into deltakit_circuit.
        qubit_mapping: Mapping[int, Qubit[Coordinate]] | None, optional
            An optional qubit mapping can be used to map qubit indices to
            coordinates. In almost all cases the qubit mapping can be obtained
            from the Stim circuit so you should leave this as None unless
            necessary.

        Returns
        -------
        Circuit
            The equivalent deltakit_circuit circuit.

        Raises
        ------
        InstructionNotImplemented
            If instruction cannot be parsed.

        Examples
        --------
        >>> import stim
        >>> import deltakit_circuit as sp
        >>> stim_circuit = stim.Circuit('''
        ... X 0 1 2
        ... Y 1 2
        ... CX 0 1
        ... ''')
        >>> sp.Circuit.from_stim_circuit(stim_circuit)
        Circuit([
            GateLayer([
                X(Qubit(0))
                X(Qubit(1))
                X(Qubit(2))
            ])
            GateLayer([
                Y(Qubit(1))
                Y(Qubit(2))
            ])
            GateLayer([
                CX(control=Qubit(0), target=Qubit(1))
            ])
        ], iterations=1)
        """
        qubit_mapping = (
            {
                index: Qubit(Coordinate(*coords), index)
                for index, coords in stim_circuit.get_final_qubit_coordinates().items()
            }
            if qubit_mapping is None
            else qubit_mapping
        )

        layers: List[Layer] = []
        for instruction in stim_circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                repeated_circuit = cls.from_stim_circuit(
                    instruction.body_copy(), qubit_mapping
                )
                repeated_circuit.iterations = instruction.repeat_count
                # If the stim circuit only consists of a repeat block just
                # return the parsed circuit
                if len(stim_circuit) == 1:
                    return repeated_circuit
                layers.append(repeated_circuit)
            elif isinstance(
                instruction, stim.CircuitInstruction
            ) and instruction.name not in ("TICK", "QUBIT_COORDS"):
                if isinstance(
                    (
                        circuit_layers := parse_circuit_instruction(
                            instruction, qubit_mapping
                        )
                    ),
                    Iterable,
                ):
                    layers.extend(circuit_layers)
                else:
                    layers.append(circuit_layers)
        return cls(layers)

    def approx_equals(
        self,
        other: object,
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 0,
    ) -> bool:
        """Determine whether two circuits are approximately equal within a
        tolerance. The tolerance accounts for small differences in the
        error probabilities of measurement gates and noise channels. All
        other gates, noise channels, annotations and properties must be equal.

        Parameters
        ----------
        other : object
            The other object to which to compare this gate layer.
        rel_tol : float
            The allowed relative difference between the error probabilities
            of two measurement gates or noise channels, if this is larger
            than that calculated from abs_tol. Has the same meaning as in
            math.isclose.
            By default, 1e-9.
        abs_tol : float, optional
            The allowed absolute difference between the error probabilities
            of two measurement gates or noise channels, if this is larger
            than that calculated from rel_tol. Has the same meaning as in
            math.isclose.
            By default, 0.0.

        Returns
        -------
        bool
            Whether the two circuits are approximately equal.
        """
        if not isinstance(other, Circuit):
            return False

        if self.iterations != other.iterations:
            return False

        if len(self.layers) != len(other.layers):
            return False

        for self_layer, other_layer in zip(self.layers, other.layers, strict=True):
            if isinstance(self_layer, (Detector, Observable, ShiftCoordinates)):
                if not self_layer == other_layer:
                    return False
            elif not self_layer.approx_equals(
                other_layer, rel_tol=rel_tol, abs_tol=abs_tol
            ):
                return False
        return True

    def flatten(self) -> Circuit:
        """
        Flatten a deltakit_circuit circuit to remove iteration blocks.
        Will keep a circuit structure exactly the same except for
        expanding out repeat blocks.

        Returns
        -------
        Circuit
            Flattened circuit.
        """
        circuit_layers = []
        for layer in self.layers:
            if isinstance(layer, Circuit):
                circuit_layers.extend(layer.flatten().layers)
            else:
                circuit_layers.append(layer)
        return Circuit(circuit_layers * self._iterations)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Circuit)
            and self.layers == other.layers
            and self.iterations == other.iterations
        )

    def __hash__(self) -> int:
        raise NotImplementedError(
            "Hash is expected to be implemented in constant time but there is not easy "
            "way of achieving that complexity with the current Circuit internals. If "
            "you get this error, please open an issue on "
            "https://github.com/Deltakit/deltakit/issues/new/choose."
        )

    def __repr__(self) -> str:
        indent = 4 * " "
        newline = "\n"
        circuit_lines = ["Circuit(["]
        circuit_lines.extend(
            f"{indent}{repr(layer).replace(newline, f'{newline}{indent}')}"
            for layer in self._layers
        )
        circuit_lines.append(f"], iterations={self.iterations})")
        return "\n".join(circuit_lines)

    def __copy__(self) -> Circuit:
        """
        Any copy of a circuit is a deep copy
        Returns
        -------
        Circuit
            A deep copy of the circuit.
        """
        return Circuit(deepcopy(self.layers), self.iterations)

    def detectors_gates(
        self, measurement_gate_stack: List[_MeasurementGate] | None = None
    ) -> List[Tuple[Detector, List[_MeasurementGate]]]:
        """Return the gates associated with each detector in this circuit,
        including nested circuits. This is returned by resolving the lookback
        indices in the full context of the circuit.

        Where a repeated detector can refer to multiple possible measurement
        gates, this logic returns the reference for the last repeated instance
        of the detector.

        Parameters
        ----------
        measurement_gate_stack : List[_MeasurementGate] | None, optional
            Any measurements from outside the current circuit, by default None

        Returns
        -------
        List[Tuple[Detector, List[_MeasurementGate]]]
            List of tuples. Tuple is a detector, and the measurement gate it
            is formed from.
        """
        measurements = (
            measurement_gate_stack if measurement_gate_stack is not None else []
        )
        detectors_gates: List[Tuple[Detector, List[_MeasurementGate]]] = []

        for iteration in range(1, self.iterations + 1):
            for layer in self.layers:
                if isinstance(layer, Circuit) and iteration == self.iterations:
                    detectors_gates.extend(layer.detectors_gates(measurements))
                elif isinstance(layer, GateLayer):
                    measurements.extend(layer.measurement_gates)
                elif isinstance(layer, Detector):
                    if iteration == self.iterations:
                        detectors_gates.append(
                            (
                                layer,
                                [
                                    measurements[measurement.lookback_index]
                                    for measurement in layer.measurements
                                ],
                            )
                        )

        return detectors_gates


Layer = Union[GateLayer, NoiseLayer, Circuit, Detector, Observable, ShiftCoordinates]
LAYERS = get_args(Layer)
