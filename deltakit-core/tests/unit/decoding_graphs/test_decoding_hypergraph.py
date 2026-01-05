# (c) Copyright Riverlane 2020-2025.
"""Tests for decoding hypergraph datastructure."""

import re
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pytest

from deltakit_core.decoding_graphs import (
    DecodingHyperEdge,
    DecodingHyperGraph,
    DetectorRecord,
    EdgeRecord,
    OrderedSyndrome,
)


def decoding_hypergraph_with_hyperedges():
    edge_ints = [(0, 1, 2), (2, 3, 6), (1, 4, 5), (6, 5, 4), (0, 6, 5)]
    hyperedges = [DecodingHyperEdge(edge) for edge in edge_ints]
    return DecodingHyperGraph(hyperedges)


def decoding_hypergraph_without_hyperedges():
    edge_ints = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3),
        (3, 6),
        (2, 5),
        (1, 4),
        (6, 5),
        (5, 4),
    ]
    hyperedges = [DecodingHyperEdge(edge) for edge in edge_ints]
    return DecodingHyperGraph(hyperedges)


def random_syndrome(random_generator) -> int:
    return random_generator.integers(100)


@pytest.fixture
def random_hypergraph(random_generator):
    random_nodes = {random_syndrome(random_generator) for _ in range(10)}
    hyperedges = {DecodingHyperEdge(pair) for pair in combinations(random_nodes, 3)}
    return DecodingHyperGraph(hyperedges)


class TestDecodingHyperGraph:
    @pytest.fixture(
        params=[
            decoding_hypergraph_with_hyperedges(),
            decoding_hypergraph_without_hyperedges(),
            "random_hypergraph",
        ]
    )
    def example_hypergraph(self, request: pytest.FixtureRequest):
        param = request.param
        if isinstance(param, str):
            param = request.getfixturevalue(param)
        return param

    @pytest.mark.parametrize(
        "edge_data",
        [
            [(0, 1, 2), (1, 2, 3)],
            {DecodingHyperEdge((0, 1, 2)), DecodingHyperEdge((1, 2, 3))},
            (DecodingHyperEdge((0, 1, 2)), (1, 2, 3)),
            ((0, 1, 2), (DecodingHyperEdge((1, 2, 3)), EdgeRecord(0))),
        ],
    )
    def test_hypergraph_can_be_constructed_using_different_edge_data_formats(
        self, edge_data
    ):
        reference_edge_data = [
            (DecodingHyperEdge((0, 1, 2)), EdgeRecord(0)),
            (DecodingHyperEdge((1, 2, 3)), EdgeRecord(0)),
        ]
        reference_graph = DecodingHyperGraph(reference_edge_data)
        hypergraph = DecodingHyperGraph(edge_data)
        assert set(hypergraph.edges) == set(reference_graph.edges)
        assert hypergraph.edge_records == reference_graph.edge_records

    @pytest.mark.parametrize(
        ("hypergraph", "expected_edges"),
        [
            (
                decoding_hypergraph_with_hyperedges(),
                {
                    frozenset({0, 1, 2}),
                    frozenset({2, 3, 6}),
                    frozenset({1, 4, 5}),
                    frozenset({6, 5, 4}),
                    frozenset({0, 6, 5}),
                },
            ),
            (
                decoding_hypergraph_without_hyperedges(),
                {
                    frozenset({0, 1}),
                    frozenset({0, 2}),
                    frozenset({0, 3}),
                    frozenset({1, 2}),
                    frozenset({2, 3}),
                    frozenset({3, 6}),
                    frozenset({2, 5}),
                    frozenset({1, 4}),
                    frozenset({6, 5}),
                    frozenset({5, 4}),
                },
            ),
        ],
    )
    def test_hypergraph_has_expected_edges(self, hypergraph, expected_edges):
        hypergraph_edges = {edge.vertices for edge in hypergraph.edges}
        assert hypergraph_edges == expected_edges

    @pytest.mark.parametrize(
        ("hypergraph", "expected_nodes"),
        [
            (decoding_hypergraph_with_hyperedges(), {0, 1, 2, 3, 4, 5, 6}),
            (decoding_hypergraph_without_hyperedges(), {0, 1, 2, 3, 4, 5, 6}),
        ],
    )
    def test_hypergraph_has_expected_nodes(self, hypergraph, expected_nodes):
        assert set(hypergraph.nodes) == expected_nodes

    def test_get_edge_returns_expected_edge(self, example_hypergraph):
        assert all(
            example_hypergraph.get_edge(*edge.vertices) == edge
            for edge in example_hypergraph.edges
        )

    @pytest.mark.parametrize(
        ("hypergraph", "non_existent_hyperedge"),
        [
            (
                decoding_hypergraph_with_hyperedges(),
                (1, 2, 3),
            ),
            (
                decoding_hypergraph_without_hyperedges(),
                (0, 3),
            ),
        ],
    )
    def test_get_edge_raises_error_for_nonexistent_hyperedge(
        self, hypergraph, non_existent_hyperedge
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Detectors {(non_existent_hyperedge,)} do not belong to any edge in the graph."
            ),
        ):
            hypergraph.get_edge(non_existent_hyperedge)

    @pytest.mark.parametrize(
        ("graph", "edge_dets", "expected_record"),
        [
            (
                DecodingHyperGraph(
                    [
                        (DecodingHyperEdge([0, 1]), EdgeRecord()),
                        (DecodingHyperEdge([1, 2, 0]), EdgeRecord(0.5)),
                    ]
                ),
                (1, 2, 0),
                EdgeRecord(0.5),
            ),
            (
                DecodingHyperGraph(
                    [
                        (DecodingHyperEdge([0, 1]), EdgeRecord(0.5)),
                        (DecodingHyperEdge([1, 2]), EdgeRecord()),
                    ]
                ),
                (0, 1),
                EdgeRecord(0.5),
            ),
            (
                DecodingHyperGraph(
                    [
                        (DecodingHyperEdge([0, 1]), EdgeRecord()),
                        (DecodingHyperEdge([1, 2, 3]), EdgeRecord(edge_refinement=0.1)),
                    ],
                ),
                (1, 2, 3),
                EdgeRecord(edge_refinement=0.1),
            ),
        ],
    )
    def test_get_edge_record_from_edge(
        self,
        graph: DecodingHyperGraph,
        edge_dets: tuple[int],
        expected_record: EdgeRecord,
    ):
        assert graph.get_edge_record(*edge_dets).data == expected_record.data

    def test_get_edges_returns_expected_edge(self, example_hypergraph):
        assert all(
            example_hypergraph.get_edges(*edge.vertices) == [edge]
            for edge in example_hypergraph.edges
        )

    @pytest.mark.parametrize(
        ("edge_ints", "expected_boundary_ints"),
        [
            ([(0, 1, 2), (2, 3, 6), (1, 4, 5)], []),
            ([(0, 1), (2, 3), (5,), (5, 6)], [5]),
            ([(0, 1, 2), (1,), (2, 3, 6), (1, 4, 5, 6), (2, 3), (2,), (9,)], [1, 2, 9]),
            ([(0,)], [0]),
        ],
    )
    def test_boundary_edges_returns_expected_edges(
        self, edge_ints, expected_boundary_ints
    ):
        hyperedges = [DecodingHyperEdge(edge) for edge in edge_ints]
        expected_boundary_edges = {
            DecodingHyperEdge([det]) for det in expected_boundary_ints
        }
        assert (
            set(DecodingHyperGraph(hyperedges).boundary_edges)
            == expected_boundary_edges
        )

    @pytest.mark.parametrize(
        ("hypergraph", "syndromes", "expected_edges"),
        [
            (
                decoding_hypergraph_without_hyperedges(),
                [0, 1, 5],
                [
                    {frozenset({0, 3}), frozenset({0, 2}), frozenset({0, 1})},
                    {frozenset({0, 1}), frozenset({1, 2}), frozenset({1, 4})},
                    {frozenset({2, 5}), frozenset({4, 5}), frozenset({6, 5})},
                ],
            ),
            (
                decoding_hypergraph_with_hyperedges(),
                [0, 1, 5],
                [
                    {frozenset({0, 1, 2}), frozenset({0, 6, 5})},
                    {frozenset({1, 4, 5}), frozenset({0, 1, 2})},
                    {frozenset({1, 4, 5}), frozenset({0, 6, 5}), frozenset({4, 5, 6})},
                ],
            ),
        ],
    )
    def test_incident_edges_return_expected_edges(
        self, hypergraph, syndromes, expected_edges
    ):
        for syndrome, expected_edge in zip(syndromes, expected_edges, strict=True):
            incident_edges = {
                frozenset(edge) for edge in hypergraph.incident_edges(syndrome)
            }
            assert incident_edges == expected_edge

    @pytest.mark.parametrize(
        ("hypergraph", "expected_neighbours"),
        [
            (
                decoding_hypergraph_without_hyperedges(),
                {0: {1, 2, 3}, 1: {0, 2, 4}, 5: {2, 4, 6}},
            ),
            (
                decoding_hypergraph_with_hyperedges(),
                {0: {1, 2, 5, 6}, 1: {0, 2, 4, 5}, 5: {0, 1, 4, 6}},
            ),
        ],
    )
    def test_neighbours_return_expected_nodes(self, hypergraph, expected_neighbours):
        neighbours = {
            syndrome: set(hypergraph.neighbors(syndrome))
            for syndrome in expected_neighbours
        }
        assert neighbours == expected_neighbours

    @pytest.mark.parametrize(
        ("edge_data", "detector_records", "expected_boundary", "expected_nx_edges"),
        [
            (
                [(1, 2), (2, 3), (2,), (3,)],
                {1: DetectorRecord(), 2: DetectorRecord(), 3: DetectorRecord()},
                frozenset({4}),
                {
                    frozenset({1, 2}),
                    frozenset({2, 3}),
                    frozenset({2, 4}),
                    frozenset({3, 4}),
                },
            ),
            (frozenset(), {}, frozenset(), frozenset()),
            (
                [(1,)],
                {1: DetectorRecord()},
                frozenset({2}),
                {frozenset({1, 2})},
            ),
            (
                [
                    (DecodingHyperEdge((0, 1)), EdgeRecord(p_err=0.2)),
                    (DecodingHyperEdge((1,)), EdgeRecord(p_err=0.3)),
                ],
                {0: DetectorRecord(time=1), 1: DetectorRecord(time=10)},
                frozenset({2}),
                {frozenset({0, 1}), frozenset({1, 2})},
            ),
            (
                [
                    (0, 1),
                    (2, 3),
                ],
                {
                    0: DetectorRecord(),
                    1: DetectorRecord(),
                    2: DetectorRecord(),
                    3: DetectorRecord(),
                },
                frozenset(),
                {frozenset({0, 1}), frozenset({2, 3})},
            ),
            ([], {1: DetectorRecord(), 2: DetectorRecord()}, frozenset(), set()),
        ],
    )
    def test_hypergraph_lowered_to_nx_graph_correctly_maps_nodes_and_edges(
        self, edge_data, detector_records, expected_boundary, expected_nx_edges
    ):
        hyper_graph = DecodingHyperGraph(edge_data, detector_records=detector_records)
        nx_graph = hyper_graph.to_nx_decoding_graph()
        nx_graph_edges = {frozenset(edge) for edge in nx_graph.edges}
        assert nx_graph_edges == expected_nx_edges
        assert nx_graph.boundaries == expected_boundary
        assert set(nx_graph.nodes) == set(detector_records.keys()).union(
            expected_boundary
        )
        assert list(nx_graph.edge_records.values()) == list(
            hyper_graph.edge_records.values()
        )
        assert all(
            hyper_graph.detector_records[det] == nx_graph.detector_records[det]
            for det in hyper_graph.nodes
        )

    def test_cannot_lower_invalid_hypergraph_to_decoding_graph(self):
        hypergraph = decoding_hypergraph_with_hyperedges()
        with pytest.raises(
            ValueError,
            match=r"Cannot cast edge: \(0, 1, 2\) " r"of degree \d to decoding edge.",
        ):
            hypergraph.to_nx_decoding_graph()

    @pytest.mark.parametrize(
        ("hypergraph", "expected_parity_check_matrix"),
        [
            (DecodingHyperGraph([]), np.empty((0, 0), dtype=np.bool_)),
            (DecodingHyperGraph([(0,)]), np.array([[1]])),
            (
                DecodingHyperGraph([(0, 1, 2), (0, 2)]),
                np.array([[1, 1], [1, 0], [1, 1]]),
            ),
        ],
    )
    def test_hypergraph_to_parity_check_matrix_gives_expected_parity_check_matrix(
        self,
        hypergraph: DecodingHyperGraph,
        expected_parity_check_matrix: npt.NDArray[np.bool_],
    ):
        np.testing.assert_array_equal(
            hypergraph.to_parity_check_matrix(), expected_parity_check_matrix
        )

    def test_decoding_graph_can_turn_error_into_syndrome(self):
        decoding_graph = DecodingHyperGraph([(0, 1, 2)])
        assert decoding_graph.error_to_syndrome(
            [decoding_graph.edges[0]]
        ) == OrderedSyndrome({0, 1, 2})

    @pytest.mark.parametrize(
        ("edges", "logicals", "expected_relevant", "expected_trimmed"),
        [
            ([(0, 1), (1, 2)], [], set(), {0, 1, 2}),
            ([(0,), (1, 2)], [{(0,)}], {0}, {1, 2}),
            ([(0, 1), (1, 2), (3, 4)], [{(0, 1)}, {(0, 1)}], {0, 1, 2}, {3, 4}),
            (
                [(0, 1, 2), (10, 11), (20, 21, 22)],
                [{(0, 1, 2), (10, 11)}],
                {0, 1, 2, 10, 11},
                {20, 21, 22},
            ),
            (
                [(0, 1, 2), (3, 4, 5), (6, 7), (8, 9, 10)],
                [{(0, 1, 2)}, {(8, 9, 10)}],
                {0, 1, 2, 8, 9, 10},
                {3, 4, 5, 6, 7},
            ),
            (
                [(0, 1), (1, 2, 3), (3, 4), (10, 11, 12)],
                [{(1, 2, 3)}],
                {0, 1, 2, 3, 4},
                {10, 11, 12},
            ),
        ],
    )
    def test_get_relevant_nodes(
        self, edges, logicals, expected_relevant, expected_trimmed
    ):
        hyper_edges = [DecodingHyperEdge(e) for e in edges]
        hg = DecodingHyperGraph(hyper_edges)
        logicals = [
            {DecodingHyperEdge(le) for le in logical_group}
            for logical_group in logicals
        ]
        relevant = hg.get_relevant_nodes(logicals)
        assert relevant == expected_relevant
        assert (set(hg.nodes) - relevant) == expected_trimmed
