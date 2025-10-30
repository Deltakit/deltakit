# (c) Copyright Riverlane 2020-2025.
from pathlib import Path

import pytest
import stim
from deltakit_core.decoding_graphs import NXDecodingGraph
from deltakit_core.decoding_graphs._decoding_graph_tools import (
    compute_graph_distance,
    filter_to_data_edges,
    filter_to_measure_edges,
    graph_to_json,
    nx_graph_from_json,
    unweight_graph,
)
from deltakit_decode.utils._graph_circuit_helpers import (
    parse_stim_circuit,
    stim_circuit_to_graph_dem,
)


class TestGraphToJSON:
    def test_2_boundary_graph_to_json_raises_exception(self):
        graph = NXDecodingGraph.from_edge_list(
            [(i, (i + 1) % 17) for i in range(17)], boundaries=[16, 8]
        )
        logicals = [[(4, 5)]]
        with pytest.raises(
            ValueError, match="JSON graph representation supports maximum one boundary"
        ):
            graph_to_json(graph, logicals)

    @pytest.fixture(
        scope="class",
        params=[
            "surface_code:rotated_memory_x",
            "surface_code:unrotated_memory_z",
        ],
    )
    def stim_circuit(self, request):
        distance = 5
        stim_circ = stim.Circuit.generated(
            request.param,
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01,
            before_round_data_depolarization=0.01,
        )
        return stim_circ

    def test_full_graph_to_json_on_code_tasks_matches_original_graph(
        self, stim_circuit
    ):
        graph, logicals, _ = parse_stim_circuit(stim_circuit)
        json_str = graph_to_json(graph, logicals, full=True)
        reconstructed_graph, reconstructed_logicals = nx_graph_from_json(json_str)
        assert set(reconstructed_graph.nodes) == set(graph.nodes)
        assert set(reconstructed_graph.edges) == set(graph.edges)
        assert all(
            log == rec_log
            for log, rec_log in zip(logicals, reconstructed_logicals, strict=True)
        )
        assert reconstructed_graph.detector_records == graph.detector_records
        assert reconstructed_graph.edge_records == graph.edge_records

    def test_graph_to_json_on_code_tasks_matches_original_graph(self, stim_circuit):
        graph, logicals, _ = parse_stim_circuit(stim_circuit)
        json_str = graph_to_json(graph, logicals)
        reconstructed_graph, reconstructed_logicals = nx_graph_from_json(json_str)
        assert set(reconstructed_graph.nodes) == set(graph.nodes)
        assert set(reconstructed_graph.edges) == set(graph.edges)
        assert all(
            log == rec_log
            for log, rec_log in zip(logicals, reconstructed_logicals, strict=True)
        )


def test_stim_circuit_to_graph_dem_does_not_decompose_the_rep_code():
    stim_rep_code = stim.Circuit.generated(
        "repetition_code:memory",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.1,
    )

    assert str(stim_circuit_to_graph_dem(stim_rep_code)).find("^") == -1


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
def test_stim_circuit_to_graph_dem_does_decompose_non_rep_codes(code_task):
    stim_rep_code = stim.Circuit.generated(
        code_task, distance=5, rounds=5, after_clifford_depolarization=0.1
    )

    assert str(stim_circuit_to_graph_dem(stim_rep_code)).find("^") != -1


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_compute_graph_distance(code_task, distance):
    stim_circ = stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    graph, logicals, stim_circ = parse_stim_circuit(stim_circ)
    assert compute_graph_distance(graph, logicals) == distance


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_compute_graph_weighted_distance(code_task, distance):
    stim_circ = stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    graph, logicals, stim_circ = parse_stim_circuit(stim_circ)
    computed_distance = compute_graph_distance(graph, logicals, weighted=True)
    assert isinstance(computed_distance, float)
    min_weight = min(record.weight for record in graph.edge_records.values())
    max_weight = max(record.weight for record in graph.edge_records.values())
    assert computed_distance >= min_weight * distance
    assert computed_distance <= max_weight * distance


@pytest.mark.parametrize(
    "circuit_path, expected_distance",
    [
        (Path("stim/circuit_logical_off_boundary.stim"), 3),
        (Path("stim/circuit_two_equivalent_logicals.stim"), 3),
        (Path("stim/circuit_multi_logicals.stim"), 3),
    ],
)
def test_compute_graph_distance_from_file(
    circuit_path, expected_distance, reference_data_dir
):
    stim_circ = stim.Circuit.from_file(reference_data_dir / circuit_path)
    graph, logicals, stim_circ = parse_stim_circuit(stim_circ)
    assert compute_graph_distance(graph, logicals) == expected_distance


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
def test_unweight_graph(code_task):
    distance = 5
    stim_circ = stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    unweighted_graph, _, stim_circ = parse_stim_circuit(stim_circ)
    unweight_graph(unweighted_graph)
    assert all(
        record["weight"] == 1 for record in unweighted_graph.edge_records.values()
    )
    # check that p_err property is set so that the computed weight is also the same
    assert all(record.weight == 1 for record in unweighted_graph.edge_records.values())


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_filter_to_edges(code_task, distance):
    stim_circ = stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    graph, _, _ = parse_stim_circuit(stim_circ)
    data_edges = filter_to_data_edges(graph)
    assert data_edges  # not empty
    for edge in data_edges:
        assert edge.is_spacelike(graph.detector_records)

    measure_edges = filter_to_measure_edges(graph)
    assert measure_edges  # not empty
    for edge in measure_edges:
        assert edge.is_timelike(graph.detector_records)
