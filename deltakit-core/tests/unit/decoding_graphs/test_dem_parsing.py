# (c) Copyright Riverlane 2020-2025.
from itertools import chain
from collections.abc import Iterable
from unittest.mock import MagicMock

import stim

import pytest
from deltakit_core.decoding_graphs import (
    DecodingHyperEdge,
    DemParser,
    DetectorRecord,
    DetectorRecorder,
    EdgeRecord,
    LogicalsInEdges,
    observable_warning,
)

from deltakit_core.decoding_graphs._dem_parsing import CoordinateOffset, DetectorCounter
from pytest_mock import MockerFixture


class TestCoordinateOffset:
    # Note about ignored lint rule below:
    # CoordinateOffset inherits from tuple, so ruff would prefer the below lines to be
    # (*CoordinateOffset(), 0, 1, 2, 3). But we explicitly want to test the "+"
    # implementation, so we use it and ignore the rule.
    def test_adding_tuple_to_empty_coordinate_offset_returns_coordinate_offset_of_tuple(
        self,
    ):
        assert CoordinateOffset() + (0, 1, 2, 3) == CoordinateOffset((0, 1, 2, 3))  # noqa: RUF005

    def test_adding_initialised_offset_to_tuple_returns_correct_offset(self):
        assert CoordinateOffset((1, 1, 1)) + (1, 1, 1) == CoordinateOffset((2, 2, 2))  # noqa: RUF005

    def test_adding_tuple_to_initialised_offset_returns_correct_offset(self):
        assert (1, 1, 1) + CoordinateOffset((1, 1, 1)) == CoordinateOffset((2, 2, 2))  # noqa: RUF005

    def test_adding_initialised_offset_to_longer_tuple_returns_correct_offset(self):
        assert CoordinateOffset((1, 1)) + (1, 1, 1) == CoordinateOffset((2, 2, 1))  # noqa: RUF005


def empty_handler(*args, **kwargs): ...


def dem_repeat(
    num_repeats: int, instructions: Iterable[str]
) -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(
        "\n".join(chain((f"repeat {num_repeats} {{",), instructions, ("}",)))
    )


class TestDemParser:
    @pytest.fixture
    def empty_parser(self):
        return DemParser(empty_handler, empty_handler, empty_handler)

    def test_detector_offset_is_increased_when_parsing_shift_detectors_instruction(
        self, empty_parser: DemParser
    ):
        empty_parser.parse(stim.DetectorErrorModel("shift_detectors 2"))
        assert empty_parser._detector_offset == 2

    def test_detector_offset_increased_when_parsing_shift_detectors_in_repeat_block(
        self, empty_parser: DemParser
    ):
        num_repeats = 3
        detector_shift = 2
        empty_parser.parse(
            dem_repeat(num_repeats, [f"shift_detectors {detector_shift}"])
        )
        assert empty_parser._detector_offset == num_repeats * detector_shift

    def test_coordinate_offset_is_increased_when_parsing_shift_detectors_with_coordinates(
        self, empty_parser: DemParser
    ):
        empty_parser.parse(stim.DetectorErrorModel("shift_detectors(1, 1) 0"))
        assert empty_parser._coordinate_offset == CoordinateOffset((1, 1))

    def test_coordinate_offset_is_increased_when_parsing_shift_detectors_in_repeat_block(
        self, empty_parser: DemParser
    ):
        num_repeats = 3
        empty_parser.parse(dem_repeat(num_repeats, ["shift_detectors(1, 1, 1) 0"]))
        assert empty_parser._coordinate_offset == CoordinateOffset((3, 3, 3))


class TestErrorHandling:
    @pytest.fixture
    def mock_error_handler(self, mocker: MockerFixture) -> MagicMock:
        return mocker.MagicMock()

    @pytest.fixture
    def mock_parser(self, mock_error_handler: MagicMock) -> DemParser:
        return DemParser(mock_error_handler, empty_handler, empty_handler)

    def test_error_handler_is_called_when_parsing_error_instruction(
        self, mock_error_handler: MagicMock, mock_parser: DemParser
    ):
        dem = stim.DetectorErrorModel("error(0.01) D0 D1")
        mock_parser.parse(dem)
        mock_error_handler.assert_called_once_with(dem[0], 0)

    def test_error_handler_called_number_of_repeats_with_error_in_repeat_block(
        self, mock_error_handler: MagicMock, mock_parser: DemParser
    ):
        num_repeats = 3
        error = "error(0.01) D0 D1"
        mock_parser.parse(dem_repeat(num_repeats, [error]))
        assert mock_error_handler.call_count == num_repeats
        mock_error_handler.assert_called_with(stim.DetectorErrorModel(error)[0], 0)

    def test_error_handler_called_with_correct_detector_offset(
        self, mock_error_handler: MagicMock, mock_parser: DemParser
    ):
        detector_offset = 3
        dem = stim.DetectorErrorModel(
            "\n".join([f"shift_detectors {detector_offset}", "error(0.01) D0 D1"])
        )
        mock_parser.parse(dem)
        mock_error_handler.assert_called_once_with(dem[-1], detector_offset)

    def test_error_handler_called_with_correct_detector_offset_within_repeat_block(
        self, mock_error_handler: MagicMock, mock_parser: DemParser
    ):
        num_repeats = 3
        detector_shift = 2
        error = stim.DetectorErrorModel("error(0.01) D0 D1")
        mock_parser.parse(
            dem_repeat(num_repeats, [f"shift_detectors {detector_shift}"]) + error
        )
        mock_error_handler.assert_called_once_with(
            error[0], num_repeats * detector_shift
        )


class TestDetectorHandling:
    @pytest.fixture
    def mock_detector_handler(self, mocker: MockerFixture) -> MagicMock:
        return mocker.MagicMock()

    @pytest.fixture
    def mock_parser(self, mock_detector_handler: MagicMock) -> DemParser:
        return DemParser(empty_handler, mock_detector_handler, empty_handler)

    def test_detector_handler_is_called_when_parsing_detector_instruction(
        self, mock_detector_handler: MagicMock, mock_parser: DemParser
    ):
        dem = stim.DetectorErrorModel("detector D0")
        mock_parser.parse(dem)
        mock_detector_handler.assert_called_once_with(dem[0], 0, CoordinateOffset())

    def test_detector_handler_called_num_repeat_times_with_detector_within_repeat_block(
        self, mock_detector_handler: MagicMock, mock_parser: DemParser
    ):
        num_repeats = 3
        detector = "detector D0"
        mock_parser.parse(dem_repeat(num_repeats, [detector]))
        assert mock_detector_handler.call_count == num_repeats
        mock_detector_handler.assert_called_with(
            stim.DetectorErrorModel(detector)[0], 0, CoordinateOffset()
        )

    def test_detector_handler_called_with_correct_detector_offset_after_shift_instruction(
        self, mock_detector_handler: MagicMock, mock_parser: DemParser
    ):
        dem = stim.DetectorErrorModel("\n".join(["shift_detectors 2", "detector D0"]))
        mock_parser.parse(dem)
        mock_detector_handler.assert_called_once_with(dem[-1], 2, CoordinateOffset())

    def test_detector_handler_called_with_correct_coordinate_offset_after_shift_instruction(
        self, mock_detector_handler: MagicMock, mock_parser: DemParser
    ):
        dem = stim.DetectorErrorModel(
            "\n".join(["shift_detectors(1, 3) 0", "detector D0"])
        )
        mock_parser.parse(dem)
        mock_detector_handler.assert_called_once_with(
            dem[-1], 0, CoordinateOffset((1, 3))
        )

    def test_detector_handler_called_with_correct_detector_offset_after_repeat_block(
        self, mock_detector_handler: MagicMock, mock_parser: DemParser
    ):
        num_repeats = 4
        detector_shift = 3
        dem = dem_repeat(
            num_repeats, [f"shift_detectors {detector_shift}"]
        ) + stim.DetectorErrorModel("detector D3")
        mock_parser.parse(dem)
        mock_detector_handler.assert_called_once_with(
            dem[-1], num_repeats * detector_shift, CoordinateOffset()
        )

    def test_detector_handler_called_with_correct_coordinate_offset_after_repeat_block(
        self, mock_detector_handler: MagicMock, mock_parser: DemParser
    ):
        num_repeats = 4
        dem = dem_repeat(
            num_repeats, ["shift_detectors(1, 1, 1) 0"]
        ) + stim.DetectorErrorModel("detector D3")
        mock_parser.parse(dem)
        mock_detector_handler.assert_called_once_with(
            dem[-1], 0, CoordinateOffset((4, 4, 4))
        )


class TestLogicalObservableHandling:
    @pytest.fixture
    def mock_logical_handler(self, mocker: MockerFixture) -> MagicMock:
        return mocker.MagicMock()

    @pytest.fixture
    def mock_parser(self, mock_logical_handler: MagicMock) -> DemParser:
        return DemParser(empty_handler, empty_handler, mock_logical_handler)

    def test_logical_observable_called_with_correct_logical_instruction(
        self, mock_logical_handler: MagicMock, mock_parser: DemParser
    ):
        dem = stim.DetectorErrorModel("logical_observable L0")
        mock_parser.parse(dem)
        mock_logical_handler.assert_called_once_with(dem[0])


class TestDetectorRecorder:
    @pytest.fixture
    def detector_recorder(self) -> DetectorRecorder:
        return DetectorRecorder()

    @pytest.fixture
    def dem_parser(self, detector_recorder: DetectorRecorder) -> DemParser:
        return DemParser(empty_handler, detector_recorder, empty_handler)

    def test_calling_detector_recorder_with_detector_instruction_adds_to_dictionary(
        self, detector_recorder: DetectorRecorder
    ):
        dem = stim.DetectorErrorModel("detector(0, 0) D1")
        detector_recorder(dem[0], 0, CoordinateOffset())
        assert detector_recorder.detector_records == {1: DetectorRecord((0,), 0)}

    @pytest.mark.parametrize(
        "detector_error_model, expected_detector_records",
        [
            (
                stim.DetectorErrorModel("detector(0, 1, 1) D0"),
                {0: DetectorRecord((0, 1), 1)},
            ),
            (
                stim.DetectorErrorModel(
                    "\n".join(["shift_detectors(1, 1, 0) 1", "detector(1, 1, 1) D2"])
                ),
                {3: DetectorRecord((2, 2), 1)},
            ),
        ],
    )
    def test_parsing_dem_file_with_detector_recorder_gives_correct_detector_records(
        self,
        dem_parser: DemParser,
        detector_error_model: stim.DetectorErrorModel,
        expected_detector_records: dict[int, DetectorRecord],
    ):
        dem_parser.parse(detector_error_model)
        assert dem_parser.detector_handler.detector_records == expected_detector_records


class TestDetectorCounter:
    @pytest.fixture
    def detector_counter(self) -> DetectorCounter:
        return DetectorCounter()

    @pytest.fixture
    def dem_parser(self, detector_counter: DetectorCounter) -> DemParser:
        return DemParser(detector_counter, empty_handler, empty_handler)

    def test_calling_detector_counter_with_detector_instruction_adds_to_counts(
        self, detector_counter: DetectorCounter
    ):
        dem = stim.DetectorErrorModel("detector(0, 0) D1\nerror(0.01) D1")
        detector_counter(dem[1], 0)
        assert detector_counter.counter == {1: 1}

    @pytest.mark.parametrize(
        "detector_error_model, expected_detector_counts",
        [
            (
                stim.DetectorErrorModel(
                    "detector(0, 1, 1) D0\n"
                    "detector(0, 1, 0) D1\n"
                    "error(0.01) D0 D1\n"
                    "error(0.01) D1\n"
                    "error(0.01) D0\n"
                ),
                {1: 2, 2: 1},
            ),
            (
                stim.DetectorErrorModel(
                    "detector(0, 1, 1) D0\n"
                    "detector(0, 1, 0) D1\n"
                    "detector(0, 2, 0) D2\n"
                    "error(0.01) D0 D1 D2\n"
                    "error(0.01) D0 D1\n"
                    "error(0.01) D1\n"
                    "error(0.01) D0 D1\n"
                    "error(0.01) D1\n"
                    "error(0.01) D0\n"
                ),
                {1: 3, 2: 2, 3: 1},
            ),
        ],
    )
    def test_parsing_dem_file_with_detector_counter_gives_correct_detector_counts(
        self,
        detector_counter: DetectorCounter,
        dem_parser: DemParser,
        detector_error_model: stim.DetectorErrorModel,
        expected_detector_counts: dict[int, int],
    ):
        dem_parser.parse(detector_error_model)
        assert detector_counter.counter == expected_detector_counts

    @pytest.mark.parametrize(
        "detector_error_model, expected_detector_max",
        [
            (
                stim.DetectorErrorModel(
                    "detector(0, 1, 1) D0\n"
                    "detector(0, 1, 0) D1\n"
                    "error(0.01) D0 D1\n"
                    "error(0.01) D1\n"
                    "error(0.01) D0\n"
                ),
                2,
            ),
            (
                stim.DetectorErrorModel(
                    "detector(0, 1, 1) D0\n"
                    "detector(0, 1, 0) D1\n"
                    "detector(0, 2, 0) D2\n"
                    "error(0.01) D0 D1 D2\n"
                    "error(0.01) D0 D1\n"
                    "error(0.01) D1\n"
                    "error(0.01) D0 D1\n"
                    "error(0.01) D1\n"
                    "error(0.01) D0\n"
                ),
                3,
            ),
        ],
    )
    def test_parsing_dem_file_with_detector_counter_gives_correct_detector_count_max(
        self,
        detector_counter: DetectorCounter,
        dem_parser: DemParser,
        detector_error_model: stim.DetectorErrorModel,
        expected_detector_max: int,
    ):
        dem_parser.parse(detector_error_model)
        assert detector_counter.max_num_detectors() == expected_detector_max


class TestObservableWarning:
    @pytest.fixture
    def observable_parser(self) -> DemParser:
        return DemParser(empty_handler, empty_handler, observable_warning)

    def test_warning_is_raised_if_logical_observable_instruction_is_in_dem(
        self, observable_parser: DemParser
    ):
        with pytest.warns(
            UserWarning, match="Isolated logical observables L1 declared in DEM file."
        ):
            observable_parser.parse(stim.DetectorErrorModel("logical_observable L1"))


class TestLogicalsInEdges:
    @pytest.mark.skip(
        reason="Stim doesn't allow adjacent separators in the DEM format."
    )
    def test_logical_in_edges_with_empty_edges_does_not_add_empty_edge(self):
        error_handler = LogicalsInEdges(0)
        dem = stim.DetectorErrorModel("error(0.01) D0 ^ ^ D1")
        error_handler(dem[0], 0)
        assert error_handler.edges == {
            DecodingHyperEdge({0}, p_err=0.01),
            DecodingHyperEdge({0}, p_err=0.01),
        }

    @pytest.mark.parametrize(
        "dem, detector_offset, expected_edge_records",
        [
            (
                stim.DetectorErrorModel("error(0.01) D0 D1"),
                0,
                {DecodingHyperEdge({0, 1}): EdgeRecord(p_err=0.01)},
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 D1"),
                2,
                {DecodingHyperEdge({2, 3}): EdgeRecord(p_err=0.01)},
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 ^ D1"),
                0,
                {
                    DecodingHyperEdge({0}): EdgeRecord(p_err=0.01),
                    DecodingHyperEdge({1}): EdgeRecord(p_err=0.01),
                },
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 ^ D1"),
                4,
                {
                    DecodingHyperEdge({4}): EdgeRecord(p_err=0.01),
                    DecodingHyperEdge({5}): EdgeRecord(p_err=0.01),
                },
            ),
        ],
    )
    def test_calling_without_logical_errors_adds_edges_at_correct_offset(
        self,
        dem: stim.DetectorErrorModel,
        detector_offset: int,
        expected_edge_records: dict[DecodingHyperEdge, EdgeRecord],
    ):
        error_handler = LogicalsInEdges(0)
        error_handler(dem[0], detector_offset)
        assert error_handler._edge_records == expected_edge_records

    @pytest.mark.parametrize(
        "dem, detector_offset, expected_logicals",
        [
            (
                stim.DetectorErrorModel("error(0.01) D0 L0"),
                0,
                [{DecodingHyperEdge({0})}],
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 L0"),
                2,
                [{DecodingHyperEdge({2})}],
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 ^ D1 L0"),
                0,
                [{DecodingHyperEdge({1})}],
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 L0 ^ D1"),
                0,
                [{DecodingHyperEdge({0})}],
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 L2"),
                0,
                [set(), set(), {DecodingHyperEdge({0})}],
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 D1 L0 L1"),
                0,
                [{DecodingHyperEdge({0, 1})}, {DecodingHyperEdge({0, 1})}],
            ),
        ],
    )
    def test_calling_with_logical_error_adds_edges_to_correct_logicals(
        self,
        dem: stim.DetectorErrorModel,
        detector_offset: int,
        expected_logicals: list[set[DecodingHyperEdge]],
    ):
        error_handler = LogicalsInEdges(len(expected_logicals))
        error_handler(dem[0], detector_offset)
        assert error_handler.logicals == expected_logicals

    @pytest.mark.parametrize(
        "dem, expected_edges, expected_logicals",
        [
            (
                dem_repeat(3, ["shift_detectors 4", "error(0.01) D0"]),
                {
                    DecodingHyperEdge({4}),
                    DecodingHyperEdge({8}),
                    DecodingHyperEdge({12}),
                },
                [],
            ),
            (
                dem_repeat(2, ["shift_detectors 2", "error(0.01) D0 L0"]),
                {DecodingHyperEdge({2}), DecodingHyperEdge({4})},
                [{DecodingHyperEdge({2}), DecodingHyperEdge({4})}],
            ),
            (
                dem_repeat(2, ["shift_detectors 2", "error(0.01) D0 ^ D1 L0"]),
                {
                    DecodingHyperEdge({2}),
                    DecodingHyperEdge({3}),
                    DecodingHyperEdge({4}),
                    DecodingHyperEdge({5}),
                },
                [{DecodingHyperEdge({3}), DecodingHyperEdge({5})}],
            ),
        ],
    )
    def test_parsing_dem_with_logicals_in_edges_has_expected_edges_and_logicals(
        self,
        dem: stim.DetectorErrorModel,
        expected_edges: set[DecodingHyperEdge],
        expected_logicals: list[set[DecodingHyperEdge]],
    ):
        parser = DemParser(
            LogicalsInEdges(len(expected_logicals)), empty_handler, empty_handler
        )
        parser.parse(dem)
        assert parser.error_handler.edges == expected_edges
        assert parser.error_handler.logicals == expected_logicals

    @pytest.mark.parametrize(
        "stim_circuit",
        [
            stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=3,
                rounds=3,
                after_clifford_depolarization=0.01,
            ),
            stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=10,
                rounds=10,
                after_clifford_depolarization=0.01,
            ),
        ],
    )
    @pytest.mark.parametrize("decompose_errors", [False, True])
    def test_each_logical_is_subset_of_all_edges(
        self, stim_circuit: stim.Circuit, decompose_errors: bool
    ):
        parser = DemParser(
            LogicalsInEdges(stim_circuit.num_observables), empty_handler, empty_handler
        )
        parser.parse(
            stim_circuit.detector_error_model(decompose_errors=decompose_errors)
        )
        assert all(
            logical.issubset(parser.error_handler.edges)
            for logical in parser.error_handler.logicals
        )
