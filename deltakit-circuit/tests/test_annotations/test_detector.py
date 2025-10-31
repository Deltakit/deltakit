# (c) Copyright Riverlane 2020-2025.
from copy import copy, deepcopy
import pytest
import stim
from deltakit_circuit import Coordinate, Detector, MeasurementRecord


class TestMeasurementRecord:
    def test_error_is_raises_when_non_negative_lookback_index(self):
        with pytest.raises(ValueError, match="Lookback index should be negative."):
            MeasurementRecord(7)

    def test_example_equality_between_measurement_records(self):
        assert MeasurementRecord(-5) == MeasurementRecord(-5)

    @pytest.mark.parametrize(
        "record_1, record_2",
        [(MeasurementRecord(-5), -5), (MeasurementRecord(-5), MeasurementRecord(-12))],
    )
    def test_example_inequality_between_measurement_records(self, record_1, record_2):
        assert record_1 != record_2

    def test_repr_of_measurement_records_matches_expected_representation(self):
        assert repr(MeasurementRecord(-3)) == "MeasurementRecord(-3)"


class TestDetector:
    @pytest.mark.parametrize("coordinate", [None, Coordinate(0, 1, 2)])
    @pytest.mark.parametrize(
        "measurement_records",
        [MeasurementRecord(-1), (MeasurementRecord(-1), MeasurementRecord(-2))],
    )
    def test_two_detectors_are_equal_if_both_measurements_and_coordinates_are_equal(
        self, measurement_records, coordinate
    ):
        assert Detector(measurement_records, coordinate) == Detector(
            measurement_records, coordinate
        )

    def test_two_detectors_are_not_equal_if_their_coordinates_differ(self):
        assert Detector(MeasurementRecord(-1), Coordinate(0, 1)) != Detector(
            MeasurementRecord(-1), Coordinate(1, 2)
        )

    def test_two_detectors_are_not_equal_if_their_measurement_records_differ(self):
        assert Detector(MeasurementRecord(-1), Coordinate(0, 1)) != Detector(
            MeasurementRecord(-2), Coordinate(0, 1)
        )

    @pytest.mark.parametrize(
        "detector, expected_repr",
        [
            (
                Detector(MeasurementRecord(-1)),
                "Detector([MeasurementRecord(-1)], coordinate=None)",
            ),
            (
                Detector([MeasurementRecord(-1), MeasurementRecord(-2)]),
                "Detector([MeasurementRecord(-1), MeasurementRecord(-2)], coordinate=None)",
            ),
            (
                Detector(MeasurementRecord(-10), coordinate=(0, 1, 2)),
                "Detector([MeasurementRecord(-10)], coordinate=Coordinate(0, 1, 2))",
            ),
        ],
    )
    def test_repr_of_detectors_matches_expected_representation(
        self, detector: Detector, expected_repr: str
    ):
        assert repr(detector) == expected_repr

    @pytest.mark.parametrize(
        "detector, expected_circuit",
        [
            (Detector([MeasurementRecord(-1)]), stim.Circuit("DETECTOR rec[-1]")),
            (Detector(MeasurementRecord(-4)), stim.Circuit("DETECTOR rec[-4]")),
            (
                Detector(MeasurementRecord(-1), Coordinate(0, 1, 2)),
                stim.Circuit("DETECTOR(0, 1, 2) rec[-1]"),
            ),
        ],
    )
    def test_stim_circuit_of_single_detector_with_single_record(
        self, detector: Detector, expected_circuit, empty_circuit
    ):
        detector.permute_stim_circuit(empty_circuit)
        assert empty_circuit == expected_circuit

    @pytest.mark.parametrize(
        "detector",
        [
            (
                Detector(
                    [
                        MeasurementRecord(-1),
                        MeasurementRecord(-1),
                        MeasurementRecord(-2),
                    ]
                )
            ),
            (
                Detector(
                    [
                        MeasurementRecord(-5),
                        MeasurementRecord(-4),
                        MeasurementRecord(-9),
                    ]
                )
            ),
            (
                Detector(
                    [MeasurementRecord(-1), MeasurementRecord(-2)], Coordinate(0, 1, 2)
                )
            ),
        ],
    )
    def test_stim_circuit_of_single_detector_with_multiple_records(
        self, detector: Detector, empty_circuit
    ):
        detector.permute_stim_circuit(empty_circuit)
        detector_circuit_str = str(empty_circuit)
        assert all(
            f"rec[{record.lookback_index}]" in detector_circuit_str
            for record in detector.measurements
        )

    @pytest.mark.parametrize("iterable", [[0, 1, 2], (0, 2, 1), range(4)])
    def test_passing_iterable_for_coordinates_converts_it_to_coordinate_class(
        self, iterable
    ):
        assert isinstance(
            Detector(MeasurementRecord(-1), iterable).coordinate, Coordinate
        )

    @pytest.mark.parametrize("tag", [None, "", "sjkdhf", "λ", "leaky<0>"])
    def test_tag_on_instruction(self, tag: str | None) -> None:
        detector = Detector(
            [MeasurementRecord(-1), MeasurementRecord(-2)], Coordinate(0, 1, 2), tag=tag
        )
        assert detector.tag == tag


class TestCoordinateTransforms:
    def test_detector_with_none_coordinate_maps_to_none(self):
        detector = Detector(MeasurementRecord(-1))
        detector.transform_coordinates({Coordinate(0, 0, 1): Coordinate(0, 0, 2)})
        assert detector.coordinate is None

    def test_detector_with_coordinates_not_in_mapping_does_not_change(self):
        coordinate = Coordinate(0, 0, 1)
        detector = Detector(MeasurementRecord(-1), coordinate)
        detector.transform_coordinates({})
        assert detector.coordinate == coordinate

    def test_detector_with_coordinates_in_mapping_changes_coordinates(self):
        detector = Detector(MeasurementRecord(-1), Coordinate(0, 0, 1))
        detector.transform_coordinates({Coordinate(0, 0, 1): Coordinate(0, 1, 1)})
        assert detector.coordinate == Coordinate(0, 1, 1)

    @pytest.mark.parametrize(
        "detector",
        (
            Detector(MeasurementRecord(-1), Coordinate(0, 0, 1)),
            Detector(MeasurementRecord(-1), Coordinate(0, 0, 0)),
            Detector(MeasurementRecord(-2), Coordinate(0, 0, 1)),
        ),
    )
    def test_detector_hash(self, detector: Detector):
        assert hash(detector) == hash(detector)
        assert hash(detector) == hash(copy(detector))
        assert hash(detector) == hash(deepcopy(detector))
