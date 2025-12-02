# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import math
from typing import Literal

import pytest
import stim
from deltakit_core.decoding_graphs import (
    DecodingHyperEdge,
    DecodingHyperMultiGraph,
    DetectorRecord,
    EdgeRecord,
)
from deltakit_core.decoding_graphs._explained_dem_parsing import (
    depolarising_as_independent,
    extract_logicals,
    noise_probability,
    parse_explained_dem,
)
from pytest_mock.plugin import MockerFixture

PauliNoiseT = Literal["X_ERROR", "Y_ERROR", "Z_ERROR"]
DepolariseNoiseT = Literal["DEPOLARIZE1", "DEPOLARIZE2"]
NoiseChannelT = PauliNoiseT | DepolariseNoiseT


def _get_stim_target(
    noise_channel: NoiseChannelT | str,
) -> stim.CircuitTargetsInsideInstruction:
    return stim.CircuitTargetsInsideInstruction(
        gate=noise_channel,
        args=[0.01],
        target_range_start=0,
        target_range_end=1,
        targets_in_range=[],
    )


def _get_error_location(noise_channel: NoiseChannelT, p_err: float):
    return stim.CircuitErrorLocation(
        tick_offset=0,
        flipped_pauli_product=[],
        flipped_measurement=None,
        instruction_targets=stim.CircuitTargetsInsideInstruction(
            gate=noise_channel,
            args=[p_err],
            target_range_start=0,
            target_range_end=1,
            targets_in_range=[],
        ),
        stack_frames=[],
    )


@pytest.mark.parametrize(
    "depolarising_probability, num_qubits, independent_probability",
    [(0.5, 1, 0.21132486540518708), (0.1, 2, 0.00700025260754783)],
)
def test_converting_known_depolarising_probability_gives_expected_independent_probability(
    depolarising_probability: float, num_qubits: int, independent_probability: float
):
    assert math.isclose(
        depolarising_as_independent(depolarising_probability, num_qubits),
        independent_probability,
    )


@pytest.mark.parametrize("num_qubits, mixing_probability", [(1, 3 / 4), (2, 15 / 16)])
def test_error_is_raised_if_depolarising_probability_is_above_mixing_probability(
    num_qubits: int, mixing_probability: float
):
    epsilon = 1e-5
    with pytest.raises(
        ValueError, match=r"Depolarising probability cannot be above the.*"
    ):
        depolarising_as_independent(mixing_probability + epsilon, num_qubits)


@pytest.mark.parametrize(
    "bad_target",
    [
        _get_stim_target(gate)
        for gate in (
            "CORRELATED_ERROR",
            "ELSE_CORRELATED_ERROR",
            "PAULI_CHANNEL_1",
            "PAULI_CHANNEL_2",
        )
    ],
)
def test_error_is_raised_if_noise_probability_target_is_not_valid_type(bad_target):
    with pytest.raises(TypeError, match=r"Unsupported gate type:.*"):
        noise_probability(bad_target)


@pytest.mark.parametrize("gate_name", ["DEPOLARIZE1", "DEPOLARIZE2"])
def test_noise_probability_calls_independent_probability_method_when_given_depolarising_noise(
    mocker: MockerFixture, gate_name: DepolariseNoiseT
):
    mocked_noise = mocker.patch(
        "deltakit_core.decoding_graphs._explained_dem_parsing.depolarising_as_independent"
    )
    noise_probability(_get_stim_target(gate_name))
    # Last number of the gate name defines the number of qubits to act on.
    mocked_noise.assert_called_once_with(0.01, int(gate_name[-1]))


@pytest.mark.parametrize("pauli_error", ["X_ERROR", "Y_ERROR", "Z_ERROR"])
def test_noise_probability_returns_gate_probability_of_pauli_error(
    pauli_error: PauliNoiseT,
):
    assert noise_probability(_get_stim_target(pauli_error)) == 0.01


class TestExplainedDemParsing:
    def test_error_is_raised_if_target_separator_is_in_the_explained_dem(self):
        error = stim.ExplainedError(
            dem_error_terms=[
                stim.DemTargetWithCoords(dem_target=stim.target_separator(), coords=[])
            ],
            circuit_error_locations=[],
        )
        with pytest.raises(
            TypeError, match="Target separators should not be in the explained DEM."
        ):
            parse_explained_dem([error])

    def test_error_with_multiple_dem_terms_has_all_nodes_in_graph(self):
        error = stim.ExplainedError(
            dem_error_terms=[
                stim.DemTargetWithCoords(
                    dem_target=stim.target_relative_detector_id(0),
                    coords=[1.0, 2.0, 3.0],
                ),
                stim.DemTargetWithCoords(
                    dem_target=stim.target_relative_detector_id(1), coords=[0.0, 1.2, 1]
                ),
            ],
            circuit_error_locations=[],
        )
        assert parse_explained_dem([error]).nodes == [0, 1]

    def test_coordinate_information_in_explained_error_is_in_detector_records(self):
        error = stim.ExplainedError(
            dem_error_terms=[
                stim.DemTargetWithCoords(
                    dem_target=stim.target_relative_detector_id(0),
                    coords=[1.0, 2.0, 3.0],
                )
            ],
            circuit_error_locations=[],
        )
        graph = parse_explained_dem([error])
        assert graph.detector_records[0] == DetectorRecord((1.0, 2.0), time=3.0)

    def test_error_with_no_circuit_error_locations_has_no_edges(self):
        error = stim.ExplainedError(
            dem_error_terms=[
                stim.DemTargetWithCoords(
                    dem_target=stim.target_relative_detector_id(0),
                    coords=[1.0, 2.0, 3.0],
                ),
            ],
            circuit_error_locations=[],
        )
        assert len(parse_explained_dem([error]).edges) == 0

    def test_there_is_an_edge_for_every_error_location_in_an_error(self):
        mock_error_location = _get_error_location("X_ERROR", 0.01)
        error = stim.ExplainedError(
            dem_error_terms=[
                stim.DemTargetWithCoords(
                    dem_target=stim.target_relative_detector_id(0), coords=[0.0]
                )
            ],
            circuit_error_locations=[mock_error_location, mock_error_location],
        )
        assert len(parse_explained_dem([error]).edges) == 2

    def test_edge_records_have_the_correct_p_err(self):
        p_err = 0.01
        mock_error_location = _get_error_location("X_ERROR", p_err)
        error = stim.ExplainedError(
            dem_error_terms=[
                stim.DemTargetWithCoords(
                    dem_target=stim.target_relative_detector_id(0), coords=[0.0]
                )
            ],
            circuit_error_locations=[mock_error_location],
        )
        graph = parse_explained_dem([error])
        assert next(iter(graph.edge_records.values())).p_err == p_err

    @pytest.mark.parametrize("logicals_affected", [[0], [0, 1]])
    def test_logical_information_is_correctly_preserved_in_the_edge_records(
        self, logicals_affected: list[int]
    ):
        mock_error_location = _get_error_location("X_ERROR", 0.01)
        error = stim.ExplainedError(
            dem_error_terms=[
                stim.DemTargetWithCoords(
                    dem_target=stim.target_relative_detector_id(0), coords=[0.01, 0.02]
                ),
            ]
            + [
                stim.DemTargetWithCoords(
                    dem_target=stim.target_logical_observable_id(el), coords=[]
                )
                for el in logicals_affected
            ],
            circuit_error_locations=[mock_error_location],
        )
        graph = parse_explained_dem([error])
        assert (
            next(iter(graph.edge_records.values()))["logicals_affected"]
            == logicals_affected
        )


class TestExtractLogicals:
    @pytest.fixture(scope="class")
    def decoding_edge(self):
        return DecodingHyperEdge({0, 1, 2})

    def test_edges_affecting_logical_zero_are_in_zeroth_logical(
        self, decoding_edge: DecodingHyperEdge
    ):
        graph = DecodingHyperMultiGraph(
            [(decoding_edge, EdgeRecord(0, logicals_affected=[0]))]
        )
        assert extract_logicals(graph) == [{(decoding_edge, 0)}]

    def test_edges_affecting_logical_one_are_in_first_logical_only(
        self, decoding_edge: DecodingHyperEdge
    ):
        graph = DecodingHyperMultiGraph(
            [(decoding_edge, EdgeRecord(0, logicals_affected=[1]))]
        )
        assert extract_logicals(graph) == [set(), {(decoding_edge, 0)}]
