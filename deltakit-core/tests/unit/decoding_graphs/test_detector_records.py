# (c) Copyright Riverlane 2020-2025.
"""Tests for decoding DetectorRecord datastructure."""

from itertools import chain

import pytest
from deltakit_core.decoding_graphs import DetectorRecord


class TestDetectorRecord:
    @pytest.mark.parametrize(
        "detector_record, expected_coord",
        [
            (DetectorRecord((2, 6)), (2, 6, 0)),
            (DetectorRecord(), (0,)),
            (DetectorRecord((1,), 5), (1, 5)),
            (DetectorRecord(time=5), (5,)),
        ],
    )
    def test_full_coordinates_are_spatial_then_time(
        self, detector_record: DetectorRecord, expected_coord: tuple[int, ...]
    ):
        assert detector_record.full_coord == expected_coord

    def test_detector_record_default_properties(self):
        default_detector_record = DetectorRecord()
        assert default_detector_record.spatial_coord == ()
        assert default_detector_record.time == 0

    def test_detector_record_logical_flip_equality_is_defined(self, random_generator):
        coord = random_generator.integers(0, 100, 2)
        time = random_generator.integers(0, 100)
        assert DetectorRecord(coord, time) == DetectorRecord(coord, time)

    def test_detector_record_logical_flip_inequality_is_defined(self, random_generator):
        coord = random_generator.integers(0, 100, 2)
        time = random_generator.integers(0, 100)
        assert DetectorRecord(coord, time) != DetectorRecord(coord, time + 1)

    def test_detector_record_from_dict_returns_correct_detector_record(
        self, random_generator
    ):
        coord = tuple(random_generator.integers(0, 100, 2))
        time = random_generator.integers(0, 100)
        record_from_dict = DetectorRecord.from_dict(
            {"time": time, "spatial_coord": coord, "extra_key": 7}
        )
        assert record_from_dict == DetectorRecord(coord, time)

    def test_detector_record_from_iterable_returns_correct_detector_record(
        self, random_generator
    ):
        spatial_coordinate = random_generator.integers(0, 100, 2).tolist()
        time = random_generator.integers(0, 100)
        assert DetectorRecord.from_sequence(
            list(chain(spatial_coordinate, [time]))
        ) == DetectorRecord(tuple(spatial_coordinate), time)

    def test_detector_record_from_empty_iterable_returns_empty_detector_record(self):
        assert DetectorRecord.from_sequence([]) == DetectorRecord()

    def test_detector_record_from_single_item_iterable_puts_item_as_time_coord(self):
        assert DetectorRecord.from_sequence([2]) == DetectorRecord(time=2)

    def test_can_set_spatial_coord(self):
        detector_record = DetectorRecord((2, 6))
        detector_record.spatial_coord = (2, 3, 5)
        assert detector_record.spatial_coord == (2, 3, 5)

    def test_setting_spatial_coord_updates_full_coord(self):
        detector_record = DetectorRecord((2, 6), 2)
        detector_record.spatial_coord = (2, 3, 5)
        assert detector_record.full_coord == (2, 3, 5, 2)

    def test_can_set_time_coord(self):
        detector_record = DetectorRecord((2, 6), time=0)
        detector_record.time = 2
        assert detector_record.time == 2

    def test_setting_time_updates_full_coord(self):
        detector_record = DetectorRecord((2, 6), time=0)
        detector_record.time = 2
        assert detector_record.full_coord == (2, 6, 2)
