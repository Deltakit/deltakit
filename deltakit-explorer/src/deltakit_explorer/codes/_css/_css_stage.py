# (c) Copyright Riverlane 2020-2025.
"""
This module contains a class CSSStage, which encapsulates the logic
of stage generation for quantum memory experiment using CSS codes.
"""

import itertools
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property

import stim
from deltakit_circuit import Circuit, Coordinate, GateLayer, Qubit
from deltakit_circuit._basic_maps import BASIS_TO_PAULI
from deltakit_circuit.gates import (
    MPP,
    MX,
    RX,
    I,
    OneQubitCliffordGate,
    OneQubitMeasurementGate,
    OneQubitResetGate,
    TwoOperandGate,
)

from deltakit_explorer.codes._css._detectors import (
    _calculate_detector_coordinates,
    get_between_round_detectors_and_coordinate_shifts,
)
from deltakit_explorer.codes._css._observables import _construct_observables
from deltakit_explorer.codes._css._stabiliser_helper_functions import (
    _get_data_qubits_from_stabilisers,
    _transform_stabiliser,
    get_entangling_layer,
)
from deltakit_explorer.codes._stabiliser import Stabiliser


class CSSStage:
    """
    Class representing a computational stage in a CSS code experiment. Such a
    stage can consist of some initial qubit measurements, some number of
    rounds of syndrome extraction and some final qubit resets. A full
    experiment can then be constructed by putting together several stages.

    Parameters
    ----------
    stabilisers : Optional[Sequence[Sequence[Stabiliser]]]
        The stabilisers to be measured using syndrome extraction. Each sequence
        gives those stabilisers which are to be measured simultaneously, and
        thus each stabiliser within a sequence should have the same length. The
        sequences are in order of measurement. By default, None.
    num_rounds : int, optional
        The number of rounds of syndrome extraction to be performed. By
        default, 0.
    first_round_measurements : Optional[Sequence[Union[MPP, OneQubitMeasurementGate]]]
        Measurement gates to be included before the first round of syndrome
        extraction. By default, None.
    first_round_gates : Optional[Iterable[Union[OneQubitCliffordGate, TwoOperandGate]]]
        Clifford unitary gates to be included before the first round of syndrome extraction.
        The gates can be general as long as

        - no qubit appears in two gates, and
        - each gate acts on at least one data qubit.

        By default, None.
    final_round_resets : Optional[Iterable[OneQubitResetGate]]
        Reset gates to be included after the final round of syndrome extraction.
        By default, None.
    observable_definitions : Optional[Dict[int, Iterable[Union[Qubit, MPP, OneQubitMeasurementGate]]]]
        Definitions of observables to be included after the initial measurements
        and first round of syndrome extraction. The dictionary keys give the
        observable indices and the values give the measurements which should be
        included in the observable. A measurement is specified with a qubit, in
        which case the most recent one-qubit measurement performed upon the qubit
        should be included, or a measurement gate, in which case the outcome of the
        most recent time the measurement gate was performed should be included.
        By default, None.
    use_ancilla_qubits : bool
        Specifies whether the Code will use an ordered sequence of Stabilisers or not.
        If the code is ordered, one ancilla qubit must be provided for each stabiliser.
        If the code is not ordered, no ancilla qubits must be provided.

    Attributes
    ----------
    measurements_as_stabilisers : Tuple[Stabiliser, ...]
        First round of measurements as a Tuple of Stabilisers. For each measurement,
        the measured operator and the qubit used to measure the outcome are given
        by Stabiliser attributes paulis and ancilla_qubit, respectively.
    final_round_resets : Tuple[Stabiliser, ...]
        Final round resets as a Tuple of Stabilisers. For each reset, the reset
        basis and qubit to reset are given by Stabiliser attributes paulis and
        ancilla_qubit, respectively.
    ordered_stabilisers : Tuple[Stabiliser, ...]
        Stabilisers in order of measurement.
    resets_only : bool
        Whether the stage consists only of final round resets.
    allowable_final_stage : bool
        Whether the stage can be used as the final stage in an experiment.
    first_round : Circuit
        The circuit for the first round of syndrome extraction, including
        the first round data qubit measurements and the subsequent observable
        definitions.
    remaining_rounds : Circuit
        The circuit for all but the first round of syndrome extraction.
        This includes detectors as well as the final round resets.
    """

    def __init__(
        self,
        stabilisers: Sequence[Sequence[Stabiliser]] | None = None,
        num_rounds: int = 0,
        first_round_measurements: Sequence[MPP | OneQubitMeasurementGate] | None = None,
        first_round_gates: Iterable[OneQubitCliffordGate | TwoOperandGate] | None = None,
        final_round_resets: Iterable[OneQubitResetGate] | None = None,
        observable_definitions: Mapping[int, Iterable[Qubit | MPP | OneQubitMeasurementGate]] | None = None,
        use_ancilla_qubits: bool | None = None,
    ):
        self._stabilisers = () if stabilisers is None else stabilisers
        self._num_rounds = num_rounds
        self._use_ancilla_qubits = use_ancilla_qubits

        if self._num_rounds < 0:
            msg = "Number of rounds must be non-negative."
            raise ValueError(msg)
        if self._num_rounds > 0 and all(
            len(stabiliser_set) == 0 for stabiliser_set in self._stabilisers
        ):
            msg = "Non-zero number of rounds requires non-zero number of stabilisers."
            raise ValueError(msg)
        if self._num_rounds == 0 and any(
            len(stabiliser_set) > 0 for stabiliser_set in self._stabilisers
        ):
            msg = "Non-zero number of stabilisers requires non-zero number of rounds."
            raise ValueError(msg)

        self._first_round_measurements = (
            ()
            if first_round_measurements is None
            else tuple(first_round_measurements)
        )
        self._first_round_gates = (
            frozenset() if first_round_gates is None else frozenset(first_round_gates)
        )
        self._gate_qubits = self._calculate_gate_qubits(self._first_round_gates)
        self._final_round_resets = (
            frozenset() if final_round_resets is None else frozenset(final_round_resets)
        )
        self._observable_definitions = (
            {} if observable_definitions is None else observable_definitions
        )

        self._check_disjointness(
            self._stabilisers, self._first_round_measurements, self._final_round_resets
        )

        self._first_round_data_qubits = _get_data_qubits_from_stabilisers(
            self._stabilisers
        )

        self._check_first_round_gates(
            stabilisers=self._stabilisers,
            measurements=self._first_round_measurements,
            gates=self._first_round_gates,
            first_round_data_qubits=self._first_round_data_qubits,
        )

        if self._use_ancilla_qubits is None:
            self._use_ancilla_qubits = self._determine_circuit_construction_method(
                self.ordered_stabilisers
            )

        self._detector_coordinates = _calculate_detector_coordinates(
            self.ordered_stabilisers
        )

    @staticmethod
    def _calculate_gate_qubits(
        first_round_gates: frozenset[OneQubitCliffordGate | TwoOperandGate],
    ) -> tuple[Qubit]:
        """
        Calculate the qubits on which the gates act.
        """
        gate_qubits = tuple(
            itertools.chain.from_iterable(gate.qubits for gate in first_round_gates)
        )
        if len(gate_qubits) > len(set(gate_qubits)):
            msg = "Qubits in first_round_gates have to be unique."
            raise ValueError(msg)
        return gate_qubits

    @staticmethod
    def _check_disjointness(
        stabilisers: Sequence[Sequence[Stabiliser]],
        measurements: Sequence[OneQubitMeasurementGate],
        resets: Iterable[OneQubitResetGate],
    ) -> None:
        """
        Check if measurements and resets act on different qubits than stabilisers.
        """
        stabiliser_qubits = set(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(
                    stabiliser.data_qubits for stabiliser in stabiliser_set
                )
                for stabiliser_set in stabilisers
            )
        )
        measurement_qubits = {
            gate.qubit
            for gate in measurements
            if isinstance(gate, OneQubitMeasurementGate)
        }
        reset_qubits = {gate.qubit for gate in resets}

        if not stabiliser_qubits.isdisjoint(measurement_qubits):
            msg = (
                "Initial measurement qubits and qubits in stabilisers "
                "should be disjoint."
            )
            raise ValueError(msg)

        if not stabiliser_qubits.isdisjoint(reset_qubits):
            msg = "Final reset qubits and qubits in stabilisers should be disjoint."
            raise ValueError(msg)

    @staticmethod
    def _check_first_round_gates(
        stabilisers: Sequence[Sequence[Stabiliser]],
        measurements: Sequence[OneQubitMeasurementGate],
        gates: frozenset[OneQubitCliffordGate | TwoOperandGate],
        first_round_data_qubits: Sequence[Qubit],
    ) -> None:
        """
        Check that, if first_round_gates is not empty, then the following hold:
          1) first_round_measurements is empty,
          2) stabilisers is not empty,
          3) each gate from first_round gates acts on at least one data qubit.
        """
        if len(gates) == 0:
            return
        # Check 1)
        if len(measurements) != 0:
            msg = (
                "If first_round_gates is non-empty, then first_round_measurements has "
                "to be empty."
            )
            raise ValueError(msg)
        # Check 2).
        if all(
            len(simultaneous_stabilisers) == 0
            for simultaneous_stabilisers in stabilisers
        ):
            msg = (
                "The stabilisers parameter is empty, which is not allowed when "
                "first_round_gates is not empty."
            )
            raise ValueError(msg)
        # Check 3).
        for gate in gates:
            if len(set(gate.qubits).intersection(first_round_data_qubits)) == 0:
                msg = (
                    f"The gate {gate} from first_round_gates is not supported on any "
                    "data qubits, which is not allowed."
                )
                raise ValueError(msg)

    @staticmethod
    def _determine_ancilla_uniqueness(
        stabilisers: Sequence[Sequence[Stabiliser]],
    ) -> bool:
        """
        Determine whether all ancilla qubits are unique.

        Returns
        -------
        bool
            True if all ancilla qubits are unique, False otherwise.
        """
        ancilla_qubits = set()
        for stabiliser_set in stabilisers:
            for stabiliser in stabiliser_set:
                if (qubit := stabiliser.ancilla_qubit) in ancilla_qubits:
                    return False
                ancilla_qubits.add(qubit)
        return True

    @staticmethod
    def _determine_circuit_construction_method(
        stabilisers: tuple[Stabiliser, ...],
    ) -> bool:
        """
        Check whether the stabilisers have ancilla qubits defined and thus determine
        whether the full syndrome extraction circuit should be constructed or if
        MPPs should be used.

        Parameters
        ----------
        stabilisers : Tuple[Stabiliser, ...]
            The stabilisers for which we wish to perform syndrome extraction.

        Returns
        -------
        bool
            Whether the full syndrome extraction circuit should be constructed (True)
            or MPPs should be used (False).

        Raises
        ------
        ValueError
            If there is a mix of stabilisers with and without an ancilla qubit
            defined.
        """
        ancillas_defined = [
            isinstance(stabiliser.ancilla_qubit, Qubit) for stabiliser in stabilisers
        ]
        if all(ancillas_defined):
            return True
        if not any(ancillas_defined):
            return False
        msg = "Either all stabilisers or no stabilisers should have an ancilla defined."
        raise ValueError(msg)

    def _construct_mpp_syndrome_extraction_layers(self) -> list[GateLayer]:
        """Construct the syndrome extraction circuit for the stabilisers using MPPs."""
        layers = []
        for stabilisers_set in self._stabilisers:
            if len(stabilisers_set) == 0:
                continue

            layers.append(
                GateLayer({I(qubit) for qubit in self._first_round_data_qubits})
            )

            for stabiliser in stabilisers_set:
                mpp = MPP([pauli for pauli in stabiliser.paulis if pauli is not None])
                # Fill mpp_layers such that no qubit clashes occur
                # Start a new layer if last layer prohibits current qubit
                if any(qubit in layers[-1].qubits for qubit in mpp.qubits):
                    layers.append(GateLayer(mpp))
                else:
                    layers[-1].add_gates(mpp)

        return layers

    def _construct_full_syndrome_extraction_layers(self) -> list[GateLayer]:
        """
        Construct the syndrome extraction circuit for the stabilisers using gates.
        """
        layers: list[GateLayer] = []
        for timestep, timestep_stabs in enumerate(self._stabilisers):
            ancilla_qubits = tuple(stab.ancilla_qubit for stab in timestep_stabs)
            layers.append(GateLayer(RX(q) for q in ancilla_qubits))

            if timestep == 0:
                layers.append(GateLayer(I(q) for q in self._first_round_data_qubits))

            num_layers = len(timestep_stabs[0].paulis)
            layers.extend(
                get_entangling_layer(timestep_stabs, i) for i in range(num_layers)
            )

            layers.append(GateLayer(MX(q) for q in ancilla_qubits))

        return layers

    def _construct_syndrome_extraction_layers(self) -> list[GateLayer]:
        """
        Construct the layers for the syndrome extraction circuit.
        """
        if self._use_ancilla_qubits:
            return self._construct_full_syndrome_extraction_layers()
        return self._construct_mpp_syndrome_extraction_layers()

    @property
    def detector_coordinates(self) -> tuple[Coordinate, ...]:
        """
        Tuple of coordinates of detectors, computed with respect
        to stabilisers.

        Returns
        -------
        Tuple[Coordinate, ...]
            Tuple of detector coordinates.
        """
        return self._detector_coordinates

    @cached_property
    def measurements_as_stabilisers(self) -> tuple[Stabiliser, ...]:
        """
        Tuple of stabiliser objects representing measurements.

        Returns
        -------
        Tuple[Stabiliser, ...]
            Tuple of stabiliser objects.
        """
        stabs = []
        for gate in self._first_round_measurements:
            if isinstance(gate, OneQubitMeasurementGate):
                stabs.append(
                    Stabiliser(
                        paulis=(BASIS_TO_PAULI[gate.basis](gate.qubit),),
                        ancilla_qubit=gate.qubit,
                    )
                )
            else:
                stabs.append(Stabiliser(paulis=gate.pauli_product))
        return tuple(stabs)

    @cached_property
    def resets_as_stabilisers(self) -> tuple[Stabiliser, ...]:
        """
        Tuple of stabiliser objects representing final round resets.

        Returns
        -------
        Tuple[Stabiliser, ...]
            Tuple of stabiliser objects.
        """
        return tuple(
            Stabiliser(
                paulis=(BASIS_TO_PAULI[gate.basis](gate.qubit),),
                ancilla_qubit=gate.qubit,
            )
            for gate in self._final_round_resets
        )

    @cached_property
    def ordered_stabilisers(self) -> tuple[Stabiliser, ...]:
        """
        Flattened version of the stabilisers we measure in the last round of the stage,
        in order of measurement. This is constructed straight from self._stabilisers.
        """
        return tuple(
            itertools.chain.from_iterable(
                stabiliser_set for stabiliser_set in self._stabilisers
            )
        )

    @cached_property
    def stabilisers_before(self) -> tuple[Stabiliser, ...]:
        """
        A transformed version of ordered_stabilisers with removed ancilla_qubits that
        correspond to the stabilisers before first_round_gates. As ancillas are changed
        to Nones, this should be used only for operator_repr comparison.

        Returns
        -------
        Tuple[Stabiliser, ...]
            The stabilisers before applying the gates in self.first_round_gates.
        """
        data_qubits = tuple(self._first_round_data_qubits)
        tableau_qubits: tuple[Qubit] = data_qubits + self._gate_qubits

        # Create tableau that implements _first_round_gates
        tableau = stim.Tableau(len(tableau_qubits))
        for gate in self._first_round_gates:
            gate_tableau = stim.Tableau.from_named_gate(gate.stim_string)
            qubits = list(gate.qubits)
            tableau.append(
                gate=gate_tableau,
                targets=[tableau_qubits.index(qubit) for qubit in qubits],
            )
        inverse_tableau = tableau.inverse()

        return tuple(
            _transform_stabiliser(stab, tableau_qubits, inverse_tableau)
            for stab in self.ordered_stabilisers
        )

    @cached_property
    def resets_only(self) -> bool:
        """
        True if none of ordered stabilisers, first round measurements,
        and observable definitions are present.

        Returns
        -------
        bool
            True if resets only.
        """
        return not (
            len(self.ordered_stabilisers) > 0
            or len(self._first_round_measurements) > 0
            or len(self._observable_definitions) > 0
        )

    @cached_property
    def allowable_final_stage(self) -> bool:
        """
        Return whether a stage is allowable as the final stage in an experiment.
        True if there are no final round resets and one of the following holds:
        - there is no stabiliser measurement
        - stabilisers are measured for one round only without the use of ancillas

        Returns
        -------
        bool
            True if the above properties hold, else False.
        """
        if len(self._final_round_resets) > 0:
            return False
        return not (
            len(self.ordered_stabilisers) > 0
            and (self._num_rounds > 1 or self._use_ancilla_qubits)
        )

    @property
    def first_round(self) -> Circuit:
        """
        Circuit layers of the first stage round.

        Returns
        -------
        Circuit
            A circuit with the first stage round.
        """
        layers: list[GateLayer] = []
        current_layer = GateLayer()
        for gate in list(self._first_round_measurements) + list(
            self._first_round_gates
        ):
            if any(qubit in current_layer.qubits for qubit in gate.qubits):
                layers.append(current_layer)
                current_layer = GateLayer(gate)
            else:
                current_layer.add_gates(gate)
        if len(current_layer.gates) != 0:
            layers.append(current_layer)

        if self._num_rounds >= 1:
            layers += self._construct_syndrome_extraction_layers()

        mmt_gates = [mmt for layer in layers for mmt in layer.measurement_gates]
        layers += _construct_observables(self._observable_definitions, mmt_gates)

        return Circuit(layers)

    @property
    def remaining_rounds(self) -> Circuit:
        """
        The rest of the circuit layers, which do not include
        actions specific to the first layer,
        and attach final rounds resets.

        Returns
        -------
        Circuit
            A Circuit with stage rounds.
        """
        layers = []
        if self._num_rounds > 1:
            detectors = get_between_round_detectors_and_coordinate_shifts(
                self._detector_coordinates,
            )
            layers.append(
                Circuit(
                    self._construct_syndrome_extraction_layers()
                    + detectors,
                    iterations=self._num_rounds - 1,
                )
            )

        if len(self._final_round_resets) > 0:
            layers.append(GateLayer(self._final_round_resets))

        return Circuit(layers)

    def __eq__(self, other: object) -> bool:
        """
        Calculate whether other is equal to self.

        Parameters
        ----------
        other : object
            Other object for which to calculate equality with self.

        Returns
        -------
        bool
            Whether other and self are equal.
        """
        if isinstance(other, CSSStage):
            return (
                self._stabilisers == other._stabilisers
                and self._num_rounds == other._num_rounds
                and self._first_round_measurements == other._first_round_measurements
                and self._final_round_resets == other._final_round_resets
                and self._observable_definitions == other._observable_definitions
                and self._first_round_gates == other._first_round_gates
            )
        return False
