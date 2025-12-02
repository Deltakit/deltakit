# (c) Copyright Riverlane 2020-2025.
import itertools

import pytest
from deltakit_circuit import (Circuit, Coordinate, GateLayer,
                              MeasurementRecord, Observable, PauliX, PauliY,
                              PauliZ, Qubit)
from deltakit_circuit._basic_types import Coord2D
from deltakit_circuit.gates import C_XYZ, MX, MZ, RX, RZ, SWAP, H, S
from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._css._detectors import \
    _calculate_detector_coordinates
from deltakit_explorer.codes._planar_code._rotated_planar_code import \
    RotatedPlanarCode
from deltakit_explorer.codes._stabiliser import Stabiliser

from ._data_css_stage import (CSSStageTestComponents, data_x_stage,
                              data_z_stage, example_simultaneous_stabilisers,
                              final_round_with_mpps_stage, half_transv_h_stage,
                              measurements_and_observables_only_stage,
                              resets_only_stage, stabiliser_meas_stage,
                              stabiliser_reset_stage,
                              stabiliser_reset_stage_many_rounds,
                              stabiliser_stage, stabiliser_stage_spaced,
                              transv_h_stage, transv_h_with_reset_stage,
                              transv_swap_stage, transv_swap_with_reset_stage)
from ._data_css_stage_full_stage import (
    full_stage_1_round, full_stage_1_round_with_mpps, full_stage_4_rounds,
    full_stage_many_rounds_no_ancilla,
    full_stage_many_rounds_not_using_ancilla, full_stage_many_rounds_spaced,
    full_stage_many_rounds_spaced_no_ancilla)

all_example_stage_comps = [
    stabiliser_stage,
    stabiliser_stage_spaced,
    stabiliser_meas_stage,
    stabiliser_reset_stage,
    stabiliser_reset_stage_many_rounds,
    full_stage_1_round,
    full_stage_1_round_with_mpps,
    full_stage_4_rounds,
    full_stage_many_rounds_spaced,
    full_stage_many_rounds_no_ancilla,
    full_stage_many_rounds_spaced_no_ancilla,
    final_round_with_mpps_stage,
    resets_only_stage,
    measurements_and_observables_only_stage,
    data_x_stage,
    data_z_stage,
    transv_h_stage,
    transv_h_with_reset_stage,
    transv_swap_stage,
    transv_swap_with_reset_stage,
    half_transv_h_stage,
]
all_example_stages = [comp.stage for comp in all_example_stage_comps]


class TestCalculateDetectorCoordinates:
    @pytest.mark.parametrize(
        "stabilisers, expected_coordinates",
        [
            (
                [
                    Stabiliser([PauliX(0)], Qubit(1)),
                    Stabiliser([PauliX(2)], Qubit(3)),
                    Stabiliser([PauliX(10)], Qubit(5)),
                ],
                (Coordinate(1, 0), Coordinate(3, 0), Coordinate(5, 0)),
            ),
            (
                [
                    Stabiliser([PauliX(0)], Qubit((1, 1, 7))),
                    Stabiliser([PauliX(2)], Qubit((3, 0, 2))),
                ],
                (Coordinate(1.0, 1.0, 7.0, 0.0), Coordinate(3.0, 0.0, 2.0, 0.0)),
            ),
            (
                [
                    Stabiliser([PauliX(0)], Qubit((1, 1, 7))),
                    Stabiliser([PauliX(2)], Qubit((3, 0, 2))),
                    Stabiliser([PauliX((2, 2))], Qubit((3, 0))),
                ],
                (Coordinate(0, 0), Coordinate(1, 0), Coordinate(2, 0)),
            ),
            (
                [
                    Stabiliser([PauliX(0), PauliY(1)]),
                    Stabiliser([PauliX(2), PauliZ(3)], Qubit((3, 0, 2))),
                ],
                (Coordinate(0.5, 0), Coordinate(2.5, 0)),
            ),
            (
                [
                    Stabiliser([PauliX(0), PauliZ(1)], Qubit((1, 1, 7))),
                    Stabiliser([PauliX((2, 2, 2)), PauliX((0, 2, 3))], Qubit((3, 0))),
                ],
                (Coordinate(1.0, 1.0, 7.0, 0.0), Coordinate(1.0, 2.0, 2.5, 0.0)),
            ),
            (
                [
                    Stabiliser([PauliX((2, 2, 2)), PauliX((0, 2, 3))], Qubit((3, 0))),
                    Stabiliser([PauliX(0), PauliZ(1)], Qubit((1, 1, 7))),
                ],
                (Coordinate(0, 0), Coordinate(1, 0)),
            ),
            (
                [
                    Stabiliser([PauliX((2, 2, 2)), PauliX((0, 2, 3))], Qubit((3, 0))),
                    Stabiliser([PauliX(0), PauliZ(1)], Qubit(("apple", 0))),
                ],
                (Coordinate(0, 0), Coordinate(1, 0)),
            ),
            (
                [
                    Stabiliser([PauliX(0), PauliY(3)]),
                    Stabiliser([PauliX(1), PauliZ(2)], Qubit((3, 0, 2))),
                ],
                (Coordinate(1.2, 0), Coordinate(1.8, 0)),
            ),
            (
                [
                    Stabiliser([PauliX(7)], Qubit(1.2)),
                    Stabiliser([PauliX(0), PauliY(3)]),
                    Stabiliser([PauliX(1), PauliZ(2)], Qubit((3, 0, 2))),
                ],
                (Coordinate(0, 0), Coordinate(1, 0), Coordinate(2, 0)),
            ),
            (
                [
                    Stabiliser([PauliX(7)], Qubit(1.2)),
                    Stabiliser([PauliX(0), PauliY(3)]),
                    Stabiliser([PauliX(1), PauliZ((2, 1))]),
                ],
                (Coordinate(0, 0), Coordinate(1, 0)),
            ),
            (
                [
                    Stabiliser([PauliX(7)], Qubit(1.2)),
                    Stabiliser([PauliX(0), PauliY(3)]),
                    Stabiliser([PauliX(1), PauliZ("test")]),
                ],
                (Coordinate(0, 0), Coordinate(1, 0)),
            ),
        ],
    )
    def test_calculate_detector_coordinates(self, stabilisers, expected_coordinates):
        actual_coordinates = _calculate_detector_coordinates(stabilisers)
        assert all(
            actual_element == pytest.approx(expected_element)
            for actual_coordinate, expected_coordinate in zip(
                actual_coordinates, expected_coordinates
            )
            for actual_element, expected_element in zip(
                actual_coordinate, expected_coordinate
            )
        )


class TestCSSStageWithMeasurementsAndObservablesOnly:
    @pytest.fixture
    def stage_class(self):
        return CSSStage

    @pytest.fixture
    def test_stage(self):
        return measurements_and_observables_only_stage.stage

    def test_resets_only_is_false(self, test_stage):
        assert not test_stage.resets_only

    def test_allowable_final_stage_is_true(self, test_stage):
        assert test_stage.allowable_final_stage

    @pytest.mark.parametrize(
        "first_round_measurements, observable_definitions, expected_first_round",
        [
            ([MZ(0)], None, Circuit([GateLayer([MZ(0)])])),
            (
                [MZ(0)],
                {0: [Qubit(0)]},
                Circuit([GateLayer([MZ(0)]), Observable(0, {MeasurementRecord(-1)})]),
            ),
            ([MX(0), MZ(1)], None, Circuit([GateLayer([MX(0), MZ(1)])])),
            (
                [MZ(0), MX(1)],
                {0: [Qubit(0)]},
                Circuit(
                    [GateLayer([MZ(0), MX(1)]), Observable(0, {MeasurementRecord(-2)})]
                ),
            ),
            (
                [MZ(0), MX(1), MZ(2)],
                {0: [Qubit(0)], 1: [Qubit(2)]},
                Circuit(
                    [
                        GateLayer([MZ(0), MX(1), MZ(2)]),
                        Observable(0, {MeasurementRecord(-3)}),
                        Observable(1, {MeasurementRecord(-1)}),
                    ]
                ),
            ),
            (
                [MZ(0), MX(1)],
                {0: [Qubit(0), Qubit(1)]},
                Circuit(
                    [
                        GateLayer([MZ(0), MX(1)]),
                        Observable(0, {MeasurementRecord(-1), MeasurementRecord(-2)}),
                    ]
                ),
            ),
        ],
    )
    def test_first_round(
        self,
        first_round_measurements,
        observable_definitions,
        expected_first_round,
        stage_class,
    ):
        stage = stage_class(
            first_round_measurements=first_round_measurements,
            observable_definitions=observable_definitions,
        )
        assert stage.first_round == expected_first_round

    def test_first_round_raises_error_when_observable_definitions_contains_non_measured_qubit(
        self, stage_class
    ):
        with pytest.raises(
            ValueError,
            match=r".*has not been measured and thus its measurement result "
            "cannot be included in a logical observable.",
        ):
            stage_class(
                first_round_measurements=[MZ(0)], observable_definitions={0: [Qubit(1)]}
            ).first_round

    def test_remaining_rounds(self, stage_class):
        assert (
            stage_class(
                first_round_measurements=[MZ(0)], observable_definitions={0: [Qubit(0)]}
            ).remaining_rounds
            == Circuit()
        )


class TestCSSStageWithMPPFinalRound:
    @pytest.fixture
    def test_stage_components(self):
        return final_round_with_mpps_stage

    @pytest.fixture
    def test_stage(self):
        return final_round_with_mpps_stage.stage

    def test_resets_only_is_false(self, test_stage):
        assert not test_stage.resets_only

    def test_allowable_final_stage_is_true(self, test_stage):
        assert test_stage.allowable_final_stage

    def test_first_round(self, test_stage_components):
        assert (
            test_stage_components.expected_first_round
            == test_stage_components.stage.first_round
        )

    def test_remaining_rounds(self, test_stage_components):
        assert (
            test_stage_components.expected_remaining_rounds
            == test_stage_components.stage.remaining_rounds
        )

    def test_measurements_as_stabilisers(self, test_stage_components):
        assert (
            test_stage_components.expected_measurements_as_stabilisers
            == test_stage_components.stage.measurements_as_stabilisers
        )


class TestCSSStageWithResetsOnly:
    @pytest.fixture
    def stage_class(self):
        return CSSStage

    @pytest.fixture
    def test_stage(self):
        return resets_only_stage.stage

    def test_resets_only_is_true(self, test_stage):
        assert test_stage.resets_only

    def test_allowable_final_stage_is_false(self, test_stage):
        assert not test_stage.allowable_final_stage

    @pytest.mark.parametrize(
        "final_round_resets, expected_remaining_rounds_circuit",
        [
            ([RZ(0)], Circuit(GateLayer([RZ(0)]))),
            ([RX(0), RZ(6), RX(3)], Circuit(GateLayer([RX(0), RZ(6), RX(3)]))),
        ],
    )
    def test_remaining_rounds(
        self,
        final_round_resets,
        expected_remaining_rounds_circuit,
        stage_class,
    ):
        stage = stage_class(
            final_round_resets=final_round_resets,
        )
        assert stage.remaining_rounds == expected_remaining_rounds_circuit

    def test_first_round(self, stage_class):
        assert (
            stage_class(
                final_round_resets=[RZ(0), RX(1)],
            ).first_round
            == Circuit()
        )


class TestCSSStageWithAllParameters:
    def test_raises_error_when_num_rounds_is_negative(self):
        with pytest.raises(ValueError, match="Number of rounds must be non-negative."):
            CSSStage(
                num_rounds=-1,
                stabilisers=example_simultaneous_stabilisers,
            )

    @pytest.mark.parametrize("stabilisers", [[[]], None, [[], []]], ())
    def test_raises_error_when_stabilisers_is_empty_but_num_rounds_is_positive(
        self, stabilisers
    ):
        with pytest.raises(
            ValueError,
            match="Non-zero number of rounds requires non-zero number of stabilisers.",
        ):
            CSSStage(
                num_rounds=1,
                stabilisers=stabilisers,
            )

    @pytest.mark.parametrize(
        "stabilisers",
        [
            [[Stabiliser((PauliZ(0),))]],
            ((Stabiliser((PauliZ(0),)),),),
            [[], [Stabiliser((PauliZ(0),))]],
        ],
    )
    def test_raises_error_when_stabilisers_is_not_empty_but_num_rounds_is_zero(
        self, stabilisers
    ):
        with pytest.raises(
            ValueError,
            match="Non-zero number of stabilisers requires non-zero number of rounds.",
        ):
            CSSStage(
                num_rounds=0,
                stabilisers=stabilisers,
            )

    def test_raises_error_if_measured_qubits_are_not_disjoint_from_stabilisers(
        self,
    ):
        with pytest.raises(
            ValueError,
            match="Initial measurement qubits and qubits in stabilisers should be disjoint.",
        ):
            CSSStage(
                num_rounds=1,
                stabilisers=example_simultaneous_stabilisers,
                first_round_measurements=[MZ(Qubit(Coord2D(1, 1)))],
            )

    @pytest.mark.parametrize(
        "stabilisers, final_round_resets",
        [
            (example_simultaneous_stabilisers, [RZ(Qubit(Coord2D(1, 1)))]),
        ],
    )
    def test_raises_error_if_reset_qubits_are_not_disjoint_from_stabilisers(
        self, stabilisers, final_round_resets
    ):
        with pytest.raises(
            ValueError,
            match="Final reset qubits and qubits in stabilisers should be disjoint.",
        ):
            CSSStage(
                num_rounds=1,
                stabilisers=stabilisers,
                final_round_resets=final_round_resets,
            )

    @pytest.mark.parametrize(
        "stage_test_comps",
        [
            stabiliser_stage,
            stabiliser_stage_spaced,
            stabiliser_meas_stage,
            stabiliser_reset_stage,
            stabiliser_reset_stage_many_rounds,
            full_stage_1_round,
            full_stage_1_round_with_mpps,
            full_stage_4_rounds,
            full_stage_many_rounds_spaced,
            full_stage_many_rounds_no_ancilla,
            full_stage_many_rounds_not_using_ancilla,
            full_stage_many_rounds_spaced_no_ancilla,
            data_x_stage,
            data_z_stage,
        ],
    )
    def test_resets_only_is_false(self, stage_test_comps: CSSStageTestComponents):
        assert not stage_test_comps.stage.resets_only

    @pytest.mark.parametrize(
        "stage_test_comps",
        [
            stabiliser_stage,
            stabiliser_stage_spaced,
            stabiliser_meas_stage,
            stabiliser_reset_stage,
            stabiliser_reset_stage_many_rounds,
            full_stage_1_round,
            full_stage_1_round_with_mpps,
            full_stage_4_rounds,
            full_stage_many_rounds_spaced,
            full_stage_many_rounds_no_ancilla,
            full_stage_many_rounds_not_using_ancilla,
            full_stage_many_rounds_spaced_no_ancilla,
            data_x_stage,
            data_z_stage,
        ],
    )
    def test_allowable_final_stage_is_false(
        self, stage_test_comps: CSSStageTestComponents
    ):
        assert not stage_test_comps.stage.allowable_final_stage

    @pytest.mark.parametrize(
        "stage_test_comps",
        [
            stabiliser_stage,
            stabiliser_stage_spaced,
            stabiliser_meas_stage,
            stabiliser_reset_stage,
            stabiliser_reset_stage_many_rounds,
            full_stage_1_round,
            full_stage_1_round_with_mpps,
            full_stage_4_rounds,
            full_stage_many_rounds_spaced,
            full_stage_many_rounds_no_ancilla,
            full_stage_many_rounds_not_using_ancilla,
            full_stage_many_rounds_spaced_no_ancilla,
            data_x_stage,
            data_z_stage,
        ],
    )
    def test_first_round(self, stage_test_comps: CSSStageTestComponents):
        assert (
            stage_test_comps.stage.first_round == stage_test_comps.expected_first_round
        )

    @pytest.mark.parametrize(
        "stage_test_comps",
        [
            stabiliser_stage,
            stabiliser_stage_spaced,
            stabiliser_meas_stage,
            stabiliser_reset_stage,
            stabiliser_reset_stage_many_rounds,
            full_stage_1_round,
            full_stage_1_round_with_mpps,
            full_stage_4_rounds,
            full_stage_many_rounds_spaced,
            full_stage_many_rounds_no_ancilla,
            full_stage_many_rounds_not_using_ancilla,
            full_stage_many_rounds_spaced_no_ancilla,
            data_x_stage,
            data_z_stage,
        ],
    )
    def test_remaining_rounds(self, stage_test_comps: CSSStageTestComponents):
        assert (
            stage_test_comps.stage.remaining_rounds
            == stage_test_comps.expected_remaining_rounds
        )

    @pytest.mark.parametrize(
        "stage_test_comps",
        [
            stabiliser_reset_stage,
            full_stage_1_round,
        ],
    )
    def test_measurements_as_stabilisers(
        self, stage_test_comps: CSSStageTestComponents
    ):
        assert (
            stage_test_comps.stage.measurements_as_stabilisers
            == stage_test_comps.expected_measurements_as_stabilisers
        )

    @pytest.mark.parametrize(
        "stage_test_comps",
        [
            stabiliser_meas_stage,
            full_stage_1_round,
        ],
    )
    def test_resets_as_stabilisers(self, stage_test_comps: CSSStageTestComponents):
        actual_resets_as_stabilisers = stage_test_comps.stage.resets_as_stabilisers
        expected_resets_as_stabilisers = stage_test_comps.expected_resets_as_stabilisers
        assert len(actual_resets_as_stabilisers) == len(expected_resets_as_stabilisers)
        for stabiliser in actual_resets_as_stabilisers:
            assert stabiliser in expected_resets_as_stabilisers

    @pytest.mark.parametrize(
        "stage, expected_ordered_stabilisers",
        [
            (
                CSSStage(
                    first_round_measurements=[MZ(0)],
                    observable_definitions={0: [Qubit(0)]},
                    final_round_resets={RZ(1)},
                ),
                (),
            ),
            (full_stage_1_round.stage, full_stage_1_round.expected_ordered_stabilisers),
        ],
    )
    def test_ordered_stabilisers(self, stage, expected_ordered_stabilisers):
        assert stage.ordered_stabilisers == expected_ordered_stabilisers

    @pytest.mark.parametrize("stage", all_example_stages)
    def test___eq___returns_true_for_equal_stages(self, stage):
        assert stage == CSSStage(
            stabilisers=stage._stabilisers,
            num_rounds=stage._num_rounds,
            first_round_measurements=stage._first_round_measurements,
            final_round_resets=stage._final_round_resets,
            observable_definitions=stage._observable_definitions,
            first_round_gates=stage._first_round_gates,
        )

    def test___eq___returns_true_for_equal_stages_constructed_differently(self):
        assert (
            full_stage_many_rounds_no_ancilla.stage
            == full_stage_many_rounds_not_using_ancilla.stage
        )

    @pytest.mark.parametrize(
        "stage1, stage2", itertools.combinations(all_example_stages, 2)
    )
    def test___eq___returns_false_for_not_equal_stages(self, stage1, stage2):
        assert stage1 != stage2

    @pytest.mark.parametrize("stage", all_example_stages[:-6])
    def test_paulis_agree_for_stabilisers_before_and_ordered_stabilisers(self, stage):
        assert all(
            stab_1.operator_repr == stab_2.operator_repr
            for stab_1, stab_2 in zip(
                stage.ordered_stabilisers, stage.stabilisers_before
            )
        )


class TestTransversalHStage:
    def test_error_with_empty_stabilisers_and_non_empty_first_round_gates(self):
        with pytest.raises(
            ValueError,
            match="The stabilisers parameter is empty, which is not allowed when "
            "first_round_gates is not empty.",
        ):
            CSSStage(stabilisers=[], num_rounds=0, first_round_gates=[H(Coord2D(1, 1))])

    @pytest.mark.parametrize(
        "first_round_gates",
        [
            [H(Coord2D(-1, 1))],
            [H(qubit) for qubit in RotatedPlanarCode(3, 4).data_qubits],
            [H(Coord2D(1, 1)), S(Coord2D(0, 0))],
        ],
    )
    def test_error_first_round_gates_incompatible_with_data_qubits(
        self, first_round_gates
    ):
        with pytest.raises(
            ValueError,
            match=r"The gate.*from first_round_gates is not supported on any "
            "data qubits, which is not allowed.",
        ):
            CSSStage(
                stabilisers=RotatedPlanarCode(3, 3).stabilisers,
                num_rounds=1,
                first_round_gates=first_round_gates,
            )

    @pytest.mark.parametrize(
        "stage_comps", [transv_h_stage, transv_h_with_reset_stage, half_transv_h_stage]
    )
    def test_first_round(self, stage_comps: CSSStageTestComponents):
        stage = stage_comps.stage
        stage_without_trv_h = CSSStage(
            stabilisers=stage._stabilisers,
            num_rounds=stage._num_rounds,
            final_round_resets=stage._final_round_resets,
        )
        assert stage.first_round == Circuit(
            [GateLayer(stage._first_round_gates), *stage_without_trv_h.first_round.layers]
        )


class TestTransversalSWAPStage:
    def test_error_with_empty_stabilisers_and_non_empty_first_round_gates(self):
        with pytest.raises(
            ValueError,
            match="The stabilisers parameter is empty, which is not allowed when "
            "first_round_gates is not empty.",
        ):
            CSSStage(
                stabilisers=[],
                num_rounds=0,
                first_round_gates=[SWAP(Coord2D(-1, -1), Coord2D(0, 0))],
            )

    @pytest.mark.parametrize(
        "first_round_gates",
        [
            [SWAP(Coord2D(-1, -1), Coord2D(0, 0))],
            [SWAP(Coord2D(4, 2), Coord2D(2, 2))],
        ],
    )
    def test_raises_error_when_swap_does_not_contain_exactly_one_data_qubit(
        self, first_round_gates
    ):
        with pytest.raises(
            ValueError,
            match=r"The gate.*from first_round_gates is not supported on any "
            "data qubits, which is not allowed.",
        ):
            CSSStage(
                stabilisers=RotatedPlanarCode(3, 3).stabilisers,
                num_rounds=1,
                first_round_gates=first_round_gates,
            )

    @pytest.mark.parametrize(
        "first_round_gates",
        [
            [SWAP(Coord2D(1, 1), Coord2D(_, _)) for _ in [0, 2]],
            [SWAP(dq, Coord2D(0, 0)) for dq in RotatedPlanarCode(3, 3).data_qubits],
        ],
    )
    def test_raises_error_with_qubit_duplicates_in_first_round_gates(
        self, first_round_gates
    ):
        with pytest.raises(
            ValueError,
            match="Qubits in first_round_gates have to be unique.",
        ):
            CSSStage(
                stabilisers=RotatedPlanarCode(3, 3).stabilisers,
                num_rounds=1,
                first_round_gates=first_round_gates,
            )

    @pytest.mark.parametrize(
        "stage", [transv_swap_stage.stage, transv_swap_with_reset_stage.stage]
    )
    def test_first_round(self, stage):
        stage_without_trv_h = CSSStage(
            stabilisers=stage._stabilisers,
            num_rounds=stage._num_rounds,
            final_round_resets=stage._final_round_resets,
        )
        assert stage.first_round == Circuit(
            [GateLayer(stage._first_round_gates), *stage_without_trv_h.first_round.layers]
        )


class TestNonSymmetricCliffordFirstRoundGatesCase:
    """
    Tests if stabilisers are transformed as expected when first_round_gates contains
    Cliffords that are not their own inverses.
    """

    def test_stabilisers_before(self):
        expected_as_set = {
            Stabiliser(
                paulis=[
                    PauliZ(Coord2D(1, 1)),
                    PauliZ(Coord2D(1, 3)),
                    PauliZ(Coord2D(3, 1)),
                    PauliZ(Coord2D(3, 3)),
                ],
                ancilla_qubit=Coord2D(2, 2),
            ),
            Stabiliser(
                paulis=[PauliY(Coord2D(1, 3)), PauliY(Coord2D(3, 3))],
                ancilla_qubit=Coord2D(2, 4),
            ),
            Stabiliser(
                paulis=[PauliY(Coord2D(1, 1)), PauliY(Coord2D(3, 1))],
                ancilla_qubit=Coord2D(2, 0),
            ),
        }
        stage = CSSStage(
            stabilisers=RotatedPlanarCode(2, 2).stabilisers,
            num_rounds=2,
            first_round_gates=[C_XYZ(dq) for dq in RotatedPlanarCode(2, 2).data_qubits],
        )
        assert len(expected_as_set) == 3
        # Note: the schedules for patch_class and rotated_planar_code are different,
        # that's why we test with operator_repr.
        assert all(
            stab.operator_repr in [stab.operator_repr for stab in expected_as_set]
            for stab in stage.stabilisers_before
        )
