"""Unit tests for hypergraph annotation utilities focused on edge-based window ids."""

import networkx as nx
import pytest

try:
    import lestim as stim
except ImportError:
    import stim

from deltakit_core.decoding_graphs._decoding_graph import (
    DecodingCode,
    DecodingHyperGraph,
)
from deltakit_core.decoding_graphs._dem_parsing import dem_to_hypergraph_and_logicals
from deltakit_core.decoding_graphs._hypergraph_annotations_tools import (
    annotate_edges_with_window_ids,
    get_unique_logical_window_ids,
    get_unique_window_ids,
    get_window_id_transfer_graph,
    separate_edges_per_window_id,
    separate_logicals_by_window_id,
)


@pytest.fixture(scope="module")
def rotmem_code() -> DecodingCode:
    """Build a DecodingCode (hypergraph + logicals) for a rotated memory Z surface code."""
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=5,
        rounds=25,
        before_round_data_depolarization=0.001,
        after_clifford_depolarization=0.001,
        before_measure_flip_probability=0.001,
    )
    dem = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=False,
    )

    hypergraph, logicals = dem_to_hypergraph_and_logicals(dem)
    return DecodingCode(hypergraph, logicals)


@pytest.fixture(scope="module")
def edge_window_graph(rotmem_code: DecodingCode) -> DecodingHyperGraph:
    """Annotate edges with window ids using the time coordinate as the window id."""

    def coordinate_to_window_id(coordinate):
        return coordinate[-1]

    annotate_edges_with_window_ids(rotmem_code.hypergraph, coordinate_to_window_id)
    return rotmem_code.hypergraph


def test_annotate_edge_window_id_everything_window_id(
    edge_window_graph: DecodingHyperGraph,
) -> None:
    """All edges receive an integer window_id after annotation."""
    for edge_record in edge_window_graph.edge_records.values():
        assert isinstance(edge_record["window_id"], int)


def test_annotated_edge_window_ids_are_contiguous(
    edge_window_graph: DecodingHyperGraph,
) -> None:
    """Window ids cover a contiguous range from 0..25 (26 rounds)."""
    window_ids = set()
    for edge_record in edge_window_graph.edge_records.values():
        window_ids.add(edge_record["window_id"])
    assert tuple(sorted(window_ids)) == tuple(range(26))


def test_get_window_id_transfer_graph_from_nodes(
    edge_window_graph: DecodingHyperGraph,
) -> None:
    """Transfer graph contains directed edges only from i to i+1 with no cycles."""
    transfer_graph = get_window_id_transfer_graph(edge_window_graph)
    expected_transfer_graph = nx.DiGraph([(i, i + 1) for i in range(25)])
    assert nx.utils.graphs_equal(transfer_graph, expected_transfer_graph)


def test_get_unique_window_ids(edge_window_graph: DecodingHyperGraph) -> None:
    """Unique edge window ids equal the expected set of windows."""
    window_ids = get_unique_window_ids(edge_window_graph)
    assert window_ids == set(range(26))


def test_get_unique_logical_window_ids(
    edge_window_graph: DecodingHyperGraph, rotmem_code: DecodingCode
) -> None:
    """Logical window ids equal the expected contiguous set (derived from logical supports)."""
    logical_window_ids = get_unique_logical_window_ids(
        edge_window_graph, rotmem_code.logicals
    )
    assert logical_window_ids == set(range(26))


def test_separate_logicals_by_window_id(
    edge_window_graph: DecodingHyperGraph, rotmem_code: DecodingCode
) -> None:
    """Logicals are partitioned by window id with no overlap between adjacent windows."""
    separated_logicals = separate_logicals_by_window_id(
        edge_window_graph, rotmem_code.logicals
    )
    assert len(separated_logicals) == 26
    for lbl in list(separated_logicals.keys())[:-1]:
        all_logicals1 = set.union(*(set(log) for log in separated_logicals[lbl]))
        all_logicals2 = set.union(*(set(log) for log in separated_logicals[lbl + 1]))
        assert len(all_logicals1.intersection(all_logicals2)) == 0
    assert sorted(separated_logicals.keys()) == list(separated_logicals.keys())


def test_separate_edges_per_window_id(edge_window_graph: DecodingHyperGraph) -> None:
    """Test edges are grouped per window id with full coverage and no duplication."""
    per_label = separate_edges_per_window_id(edge_window_graph, enforce_window_ids=True)

    assert set(per_label.keys()) == get_unique_window_ids(edge_window_graph)

    # union of per-label edges equals all edges
    union_edges = (
        set().union(*(set(v) for v in per_label.values())) if per_label else set()
    )
    all_edges = set(edge_window_graph.edge_records.keys())
    assert union_edges == all_edges

    # each edge appears exactly once and under the correct label
    seen = set()
    for lbl, edges in per_label.items():
        for e in edges:
            assert edge_window_graph.edge_records[e]["window_id"] == lbl
            assert e not in seen
            seen.add(e)


def test_separate_edges_per_window_id_enforce_window_ids_detects_unlabelled() -> None:
    """If enforce_window_ids=True and an edge lacks window_id we raise ValueError."""
    edge1 = (0, 1)
    edge2 = (1, 2)
    hg = DecodingHyperGraph([edge1, edge2])
    hg.edge_records[next(iter(hg.edge_records.keys()))]["window_id"] = 0

    with pytest.raises(ValueError, match="unlabelled edges"):
        separate_edges_per_window_id(hg, enforce_window_ids=True)

    # without enforcement it should silently ignore the unlabelled edge
    result = separate_edges_per_window_id(hg, enforce_window_ids=False)
    assert list(result.keys()) == [0]
    assert len(result[0]) == 1
