"""Unit tests for windowing utilities."""

import pytest

from deltakit_core.decoding_graphs import (
    DecodingHyperEdge,
    DecodingHyperGraph,
    DetectorRecord,
    EdgeRecord,
)
from deltakit_core.decoding_graphs._windowing_utils import (
    connect_dangling_to_boundary_hypergraph,
    expand_nodes_to_time_span,
    induce_subhypergraph,
    nodes_within_radius,
    relabel_hypergraph_nodes_contiguously,
)


class TestNodesWithinRadius:
    def test_basic_growth_from_single_start_node(self) -> None:
        edge_data = [
            (DecodingHyperEdge((0, 1)), EdgeRecord()),
            (DecodingHyperEdge((1, 2, 3)), EdgeRecord()),
            (DecodingHyperEdge((3, 4)), EdgeRecord()),
            (DecodingHyperEdge((4, 5, 6)), EdgeRecord()),
        ]
        hg = DecodingHyperGraph(edge_data)
        assert nodes_within_radius(hg, {0}, 0) == {0}
        assert nodes_within_radius(hg, {0}, 1) == {0, 1}
        assert nodes_within_radius(hg, {0}, 2) == {0, 1, 2, 3}
        assert nodes_within_radius(hg, {0}, 3) == {0, 1, 2, 3, 4}
        assert nodes_within_radius(hg, {0}, 4) == {0, 1, 2, 3, 4, 5, 6}

    def test_negative_radius_raises(self) -> None:
        hg = DecodingHyperGraph([(0, 1)])
        with pytest.raises(ValueError, match="Radius must be non-negative"):
            nodes_within_radius(hg, {0}, -1)

    def test_growth_on_cyclic_hypergraph(self) -> None:
        """Test growth on a more complex cyclic hypergraph."""
        edge_data = [
            (DecodingHyperEdge((0, 1, 2)), EdgeRecord()),
            (DecodingHyperEdge((2, 3)), EdgeRecord()),
            (DecodingHyperEdge((3, 4, 5)), EdgeRecord()),
            (DecodingHyperEdge((5, 1)), EdgeRecord()),
            (DecodingHyperEdge((4, 6)), EdgeRecord()),
            (DecodingHyperEdge((6, 7, 8)), EdgeRecord()),
            (DecodingHyperEdge((8, 0)), EdgeRecord()),
        ]
        hg = DecodingHyperGraph(edge_data)

        r0 = nodes_within_radius(hg, {0}, 0)
        r1 = nodes_within_radius(hg, {0}, 1)
        r2 = nodes_within_radius(hg, {0}, 2)
        r3 = nodes_within_radius(hg, {0}, 3)

        assert r0 == {0}
        assert r1 == {0, 1, 2, 8}
        assert r2 == {0, 1, 2, 3, 5, 6, 7, 8}
        assert r3 == set(range(9))

        assert r0.issubset(r1)
        assert r1.issubset(r2)
        assert r2.issubset(r3)


class TestInducedSubhypergraph:
    def test_filters_edges_and_detector_records(self) -> None:
        edge_data = [
            (DecodingHyperEdge((0, 1)), EdgeRecord(0.1)),
            (DecodingHyperEdge((1, 2)), EdgeRecord(0.2)),
            (DecodingHyperEdge((2, 3)), EdgeRecord(0.3)),
            (DecodingHyperEdge((3,)), EdgeRecord(0.4)),
            (DecodingHyperEdge((0, 2)), EdgeRecord(0.5)),
        ]
        det_records = {i: DetectorRecord() for i in range(5)}
        original = DecodingHyperGraph(edge_data, detector_records=det_records)

        nodes = {0, 1, 2}
        sub = induce_subhypergraph(original, nodes)

        expected_edges = {
            DecodingHyperEdge((0, 1)),
            DecodingHyperEdge((1, 2)),
            DecodingHyperEdge((0, 2)),
        }
        assert set(sub.edges) == expected_edges

        assert sub.edge_records[DecodingHyperEdge((1, 2))].p_err == 0.2
        assert sub.edge_records[DecodingHyperEdge((0, 1))].p_err == 0.1
        assert sub.edge_records[DecodingHyperEdge((0, 2))].p_err == 0.5

        assert set(sub.detector_records.keys()) == nodes


class TestConnectDanglingToBoundary:
    def test_adds_and_combines_unary_boundary_edges(self) -> None:
        edge_data = [
            (DecodingHyperEdge((0, 1)), EdgeRecord(0.1)),  # inside
            (DecodingHyperEdge((1, 2)), EdgeRecord(0.2)),  # inside
            (DecodingHyperEdge((2, 9)), EdgeRecord(0.3)),  # leaves via 2
            (DecodingHyperEdge((2, 8, 7)), EdgeRecord(0.4)),  # leaves via 2
            (DecodingHyperEdge((1, 7)), EdgeRecord(0.5)),  # leaves via 1
            (DecodingHyperEdge((2,)), EdgeRecord(0.05)),  # existing unary on 2 (inside)
            (DecodingHyperEdge((3,)), EdgeRecord(0.6)),  # unrelated node
        ]
        det_records = {i: DetectorRecord() for i in range(10)}
        original = DecodingHyperGraph(edge_data, detector_records=det_records)

        nodes = {0, 1, 2}
        sub = induce_subhypergraph(original, nodes)

        folded = connect_dangling_to_boundary_hypergraph(original, sub)

        assert DecodingHyperEdge((0, 1)) in folded.edge_records
        assert DecodingHyperEdge((1, 2)) in folded.edge_records

        assert folded.edge_records[DecodingHyperEdge((1,))].p_err == 0.5
        assert folded.edge_records[DecodingHyperEdge((2,))].p_err == pytest.approx(
            0.464
        )

        assert folded.detector_records == sub.detector_records

    def test_no_external_edges_means_no_change(self) -> None:
        edge_data = [
            (DecodingHyperEdge((0, 1)), EdgeRecord(0.1)),
            (DecodingHyperEdge((1, 2)), EdgeRecord(0.2)),
            (DecodingHyperEdge((2,)), EdgeRecord(0.05)),
        ]
        det_records = {0: DetectorRecord(), 1: DetectorRecord(), 2: DetectorRecord()}
        original = DecodingHyperGraph(edge_data, detector_records=det_records)
        sub = induce_subhypergraph(original, {0, 1, 2})

        folded = connect_dangling_to_boundary_hypergraph(original, sub)

        assert set(folded.edges) == set(sub.edges)
        assert folded.edge_records == sub.edge_records
        assert folded.detector_records == sub.detector_records


class TestRelabelHypergraphNodesContiguously:
    def test_relabels_sparse_node_ids(self) -> None:
        # hypergraph whose node ids are sparse/non-contiguous
        edge_data = [
            (DecodingHyperEdge((10, 20)), EdgeRecord()),
            (DecodingHyperEdge((20, 30, 40)), EdgeRecord()),
            (DecodingHyperEdge((40, 10)), EdgeRecord()),
            (DecodingHyperEdge((50,)), EdgeRecord()),
        ]
        hg = DecodingHyperGraph(edge_data)
        relabelled, mapping = relabel_hypergraph_nodes_contiguously(hg)

        assert set(mapping.keys()) == set(hg.nodes)
        assert set(mapping.values()) == set(range(len(hg.nodes)))

        used_new_nodes: set[int] = set()
        for edge in relabelled.edges:
            used_new_nodes.update(edge.vertices)
        assert used_new_nodes == set(range(len(hg.nodes)))

        reverse = {v: k for k, v in mapping.items()}
        for old_edge, _ in edge_data:
            remapped = DecodingHyperEdge(mapping[n] for n in old_edge)
            assert remapped in relabelled.edge_records
            recovered = DecodingHyperEdge(reverse[n] for n in remapped)
            assert recovered == old_edge


class TestExpandNodesToTimeSpan:
    def test_expands_to_full_time_interval(self) -> None:
        det_recs = {i: DetectorRecord(time=i) for i in range(6)}
        edge_data = []
        for i in range(1, 6):
            edge_data.append((DecodingHyperEdge((i - 1, i)), EdgeRecord()))
        hg = DecodingHyperGraph(edge_data, detector_records=det_recs)
        start_nodes = {1, 3, 5}
        expanded = expand_nodes_to_time_span(hg, start_nodes)
        assert expanded == {1, 2, 3, 4, 5}

    def test_empty_nodes_no_change(self) -> None:
        hg = DecodingHyperGraph(
            [(DecodingHyperEdge((0, 1)), EdgeRecord())],
            detector_records={0: DetectorRecord(time=0), 1: DetectorRecord(time=1)},
        )
        assert expand_nodes_to_time_span(hg, set()) == set()
