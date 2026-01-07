# (c) Copyright Riverlane 2020-2025.

import pytest
import stim

import deltakit_circuit as sp
from deltakit_circuit._detector_manipulation import (
    _condense_detector_calls,
    _get_detectors_to_remove,
    _get_ordered_detector_calls,
)


class TestDetectorTrimming:
    def test_detectors_can_be_ordered_in_unnested_circuit(self):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1)),
            sp.Detector(sp.MeasurementRecord(-2)),
            sp.Detector(sp.MeasurementRecord(-3)),
        ]
        deltakit_circuit_circuit = sp.Circuit(detectors)
        assert _get_ordered_detector_calls(deltakit_circuit_circuit) == detectors

    def test_detectors_can_be_ordered_given_nested_circuits(self):
        detector1 = sp.Detector(sp.MeasurementRecord(-1))
        detector2 = sp.Detector(sp.MeasurementRecord(-2))
        deltakit_circuit_circuit = sp.Circuit(
            [detector1, sp.Circuit(detector2, iterations=3)], iterations=2
        )
        assert _get_ordered_detector_calls(deltakit_circuit_circuit) == [
            detector1,
            detector2,
            detector2,
            detector2,
            detector1,
            detector2,
            detector2,
            detector2,
        ]

    def test_detectors_can_be_condensed_in_unnested_circuits(self):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1)),
            sp.Detector(sp.MeasurementRecord(-2)),
            sp.Detector(sp.MeasurementRecord(-1)),
        ]
        deltakit_circuit_circuit = sp.Circuit(detectors)
        ordered_detector_calls = _get_ordered_detector_calls(deltakit_circuit_circuit)
        assert _condense_detector_calls(ordered_detector_calls) == [[0], [1], [2]]

    def test_detectors_can_be_condensed_given_nested_circuits(self):
        detector1 = sp.Detector(sp.MeasurementRecord(-1))
        detector2 = sp.Detector(sp.MeasurementRecord(-2))
        deltakit_circuit_circuit = sp.Circuit(
            [detector1, sp.Circuit(detector2, iterations=3)], iterations=2
        )
        ordered_detector_calls = _get_ordered_detector_calls(deltakit_circuit_circuit)
        assert _condense_detector_calls(ordered_detector_calls) == [
            [0, 4],
            [1, 2, 3, 5, 6, 7],
        ]

    def test_detectors_are_specified_for_removal_if_all_of_a_detectors_dem_ids_are_specified_for_removal(
        self,
    ):
        assert _get_detectors_to_remove([2, 3], [[2, 3], [4, 5]]) == [0]

    def test_no_detectors_are_specified_for_removal_if_a_detectors_dem_ids_are_not_all_specified_for_removal(
        self,
    ):
        assert _get_detectors_to_remove([], [[2, 3], [4, 5]]) == []

    def test_warning_is_raised_if_a_subset_of_a_detectors_dem_ids_are_specified_for_removal(
        self,
    ):
        with pytest.warns(
            RuntimeWarning,
            match="Detector is being specified for removal via "
            "an incomplete list of DEM detector IDs.*",
        ):
            _get_detectors_to_remove([2], [[2, 3], [4, 5]])

    @pytest.fixture
    def stim_circuit_with_and_without_detectors_0_4_5_10(
        self,
    ) -> tuple[stim.Circuit, stim.Circuit]:
        return (
            stim.Circuit.generated(
                "repetition_code:memory",
                rounds=4,
                distance=5,
                before_round_data_depolarization=0.1,
                before_measure_flip_probability=0.1,
            ),
            stim.Circuit(
                """R 0 1 2 3 4 5 6 7 8
                TICK
                DEPOLARIZE1(0.1) 0 2 4 6 8
                CX 0 1 2 3 4 5 6 7
                TICK
                CX 2 1 4 3 6 5 8 7
                TICK
                X_ERROR(0.1) 1 3 5 7
                MR 1 3 5 7
                TICK
                DETECTOR(3, 0) rec[-3]
                DETECTOR(5, 0) rec[-2]
                DETECTOR(7, 0) rec[-1]
                REPEAT 3 {
                    DEPOLARIZE1(0.1) 0 2 4 6 8
                    CX 0 1 2 3 4 5 6 7
                    TICK
                    CX 2 1 4 3 6 5 8 7
                    TICK
                    X_ERROR(0.1) 1 3 5 7
                    MR 1 3 5 7
                    SHIFT_COORDS(0, 1)
                    DETECTOR(5, 0) rec[-6] rec[-2]
                    DETECTOR(7, 0) rec[-5] rec[-1]
                }
                X_ERROR(0.1) 0 2 4 6 8
                M 0 2 4 6 8
                DETECTOR(1, 1) rec[-5] rec[-4] rec[-9]
                DETECTOR(3, 1) rec[-8] rec[-4] rec[-3]
                DETECTOR(7, 1) rec[-6] rec[-1] rec[-2]
                OBSERVABLE_INCLUDE(0) rec[-1]"""
            ),
        )

    @pytest.mark.parametrize(
        ("stim_circuit", "detectors_to_remove", "expected_stim_circuit"),
        [
            (
                stim.Circuit("""DETECTOR(3, 0) rec[-3]
                DETECTOR(5, 0) rec[-2]
                DETECTOR(7, 0) rec[-1]"""),
                [0],
                stim.Circuit("""DETECTOR(5, 0) rec[-2]
                DETECTOR(7, 0) rec[-1]"""),
            ),
            (
                stim.Circuit("""DETECTOR(3, 0) rec[-3]
                DETECTOR(5, 0) rec[-2]
                DETECTOR(7, 0) rec[-1]"""),
                [0, 1, 2],
                stim.Circuit(),
            ),
            (
                stim.Circuit("""DETECTOR(3, 0) rec[-3]
                X_ERROR(0.1) 1 3 5 7
                MR 1 3 5 7
                DETECTOR(5, 0) rec[-2]
                DETECTOR(7, 0) rec[-1]"""),
                [0, 2],
                stim.Circuit("""X_ERROR(0.1) 1 3 5 7
                MR 1 3 5 7
                DETECTOR(5, 0) rec[-2]"""),
            ),
        ],
    )
    def test_detectors_can_be_removed_from_circuits_without_repeat_blocks(
        self, stim_circuit, detectors_to_remove, expected_stim_circuit
    ):
        stim_circuit = sp.trim_detectors(
            stim_circuit, dem_detectors_to_eliminate=detectors_to_remove
        )
        assert stim_circuit == expected_stim_circuit

    def test_ability_to_trim_detectors_that_are_both_inside_and_outside_a_repeat_block(
        self, stim_circuit_with_and_without_detectors_0_4_5_10
    ):
        stim_circuit, expected_stim_circuit = (
            stim_circuit_with_and_without_detectors_0_4_5_10
        )
        nested_detector_indices = list(range(4, 13, 4)) + list(range(5, 14, 4))
        stim_circuit = sp.trim_detectors(
            stim_circuit, dem_detectors_to_eliminate=[0, 18, *nested_detector_indices]
        )
        assert sp.Circuit.from_stim_circuit(
            stim_circuit
        ) == sp.Circuit.from_stim_circuit(expected_stim_circuit)

    def test_warning_is_raised_if_only_a_subset_of_a_detectors_ids_are_specified_for_elimination(
        self, stim_circuit_with_and_without_detectors_0_4_5_10
    ):
        stim_circuit, _ = stim_circuit_with_and_without_detectors_0_4_5_10
        with pytest.warns(
            RuntimeWarning,
            match="Detector is being specified for removal via "
            "an incomplete list of DEM detector IDs.*",
        ):
            sp.trim_detectors(stim_circuit, dem_detectors_to_eliminate=[4])
