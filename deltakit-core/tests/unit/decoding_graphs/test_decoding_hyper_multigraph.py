# (c) Copyright Riverlane 2020-2025.
"""Tests for decoding hyper-multigraph datastructure."""

import math

import numpy as np
import pytest
from deltakit_core.decoding_graphs import (
    DecodingHyperEdge,
    DecodingHyperMultiGraph,
    DetectorRecord,
    EdgeRecord,
    OrderedSyndrome,
)


def decoding_hyper_multigraph_with_multi_hyperedges():
    edges = [
        DecodingHyperEdge((0, 1, 2)),
        DecodingHyperEdge((0, 1, 2)),
        DecodingHyperEdge((2, 3, 6)),
        DecodingHyperEdge((1, 4, 5)),
        DecodingHyperEdge((1, 4, 5)),
        DecodingHyperEdge((0, 6)),
    ]
    return DecodingHyperMultiGraph(edges)


def decoding_hyper_multigraph_with_hyperedges():
    edges = [
        DecodingHyperEdge((0, 1, 2)),
        DecodingHyperEdge((2, 3, 6)),
        DecodingHyperEdge((1, 4, 5)),
        DecodingHyperEdge((0, 6)),
    ]
    return DecodingHyperMultiGraph(edges)


def decoding_hyper_multigraph_with_multiedges():
    edges = [
        DecodingHyperEdge((0, 1)),
        DecodingHyperEdge((0, 1)),
        DecodingHyperEdge((2, 3)),
        DecodingHyperEdge((1, 4)),
        DecodingHyperEdge((1, 4)),
        DecodingHyperEdge((0, 4)),
    ]
    return DecodingHyperMultiGraph(edges)


def decoding_hyper_multigraph_with_edges():
    edges = [
        DecodingHyperEdge((0, 1)),
        DecodingHyperEdge((2, 3)),
        DecodingHyperEdge((1, 4)),
        DecodingHyperEdge((0, 4)),
    ]
    return DecodingHyperMultiGraph(edges)


def decoding_hyper_multigraph_no_edges():
    return DecodingHyperMultiGraph([])


class TestDecodingHyperMultiGraph:
    """Unit tests for DecodingHyperMultiGraph class."""

    @pytest.mark.parametrize(
        "graph, expected_edges",
        [
            (decoding_hyper_multigraph_no_edges(), set()),
            (
                decoding_hyper_multigraph_with_edges(),
                {
                    (DecodingHyperEdge((0, 1)), 0),
                    (DecodingHyperEdge((2, 3)), 0),
                    (DecodingHyperEdge((1, 4)), 0),
                    (DecodingHyperEdge((0, 4)), 0),
                },
            ),
            (
                decoding_hyper_multigraph_with_hyperedges(),
                {
                    (DecodingHyperEdge((0, 1, 2)), 0),
                    (DecodingHyperEdge((2, 3, 6)), 0),
                    (DecodingHyperEdge((1, 4, 5)), 0),
                    (DecodingHyperEdge((0, 6)), 0),
                },
            ),
            (
                decoding_hyper_multigraph_with_multiedges(),
                {
                    (DecodingHyperEdge((0, 1)), 0),
                    (DecodingHyperEdge((0, 1)), 1),
                    (DecodingHyperEdge((2, 3)), 0),
                    (DecodingHyperEdge((1, 4)), 0),
                    (DecodingHyperEdge((1, 4)), 1),
                    (DecodingHyperEdge((0, 4)), 0),
                },
            ),
            (
                decoding_hyper_multigraph_with_multi_hyperedges(),
                {
                    (DecodingHyperEdge((0, 1, 2)), 0),
                    (DecodingHyperEdge((0, 1, 2)), 1),
                    (DecodingHyperEdge((2, 3, 6)), 0),
                    (DecodingHyperEdge((1, 4, 5)), 0),
                    (DecodingHyperEdge((1, 4, 5)), 1),
                    (DecodingHyperEdge((0, 6)), 0),
                },
            ),
        ],
    )
    def test_hyper_multigraph_has_expected_edges(self, graph, expected_edges):
        assert set(graph.edges) == expected_edges

    @pytest.mark.parametrize(
        "edge_data",
        [
            [(0, 1, 2), (1, 2, 3), (0, 1, 2)],
            [
                DecodingHyperEdge((0, 1, 2)),
                DecodingHyperEdge((1, 2, 3)),
                DecodingHyperEdge((0, 1, 2)),
            ],
            (DecodingHyperEdge((0, 1, 2)), (1, 2, 3), (0, 1, 2)),
            ((0, 1, 2), (1, 2, 3), (DecodingHyperEdge((0, 1, 2)), EdgeRecord(0))),
        ],
    )
    def test_nx_multigraph_can_be_constructed_using_different_edge_data_formats(
        self, edge_data
    ):
        reference_edge_data = [
            (DecodingHyperEdge((0, 1, 2)), EdgeRecord(0)),
            (DecodingHyperEdge((1, 2, 3)), EdgeRecord(0)),
            (DecodingHyperEdge((0, 1, 2)), EdgeRecord(0)),
        ]
        reference_graph = DecodingHyperMultiGraph(reference_edge_data)
        graph = DecodingHyperMultiGraph(edge_data)
        assert set(graph.edges) == set(reference_graph.edges)
        assert graph.edge_records == reference_graph.edge_records

    @pytest.mark.parametrize(
        "graph, expected_nodes",
        [
            (decoding_hyper_multigraph_with_edges(), {0, 1, 2, 3, 4}),
            (decoding_hyper_multigraph_with_hyperedges(), {0, 1, 2, 3, 4, 5, 6}),
            (decoding_hyper_multigraph_with_multiedges(), {0, 1, 2, 3, 4}),
            (decoding_hyper_multigraph_with_multi_hyperedges(), {0, 1, 2, 3, 4, 5, 6}),
        ],
    )
    def test_hyper_multigraph_has_expected_nodes(self, graph, expected_nodes):
        assert set(graph.nodes) == expected_nodes

    @pytest.mark.parametrize(
        "edges, boundary_edges",
        [
            (
                [
                    DecodingHyperEdge((0, 1)),
                    DecodingHyperEdge((0, 1)),
                ],
                [],
            ),
            (
                [DecodingHyperEdge((0,)), DecodingHyperEdge((0, 1))],
                [(DecodingHyperEdge((0,)), 0)],
            ),
            (
                [
                    DecodingHyperEdge((0,)),
                    DecodingHyperEdge((0,)),
                    DecodingHyperEdge((1,)),
                ],
                [
                    (DecodingHyperEdge((0,)), 0),
                    (DecodingHyperEdge((0,)), 1),
                    (DecodingHyperEdge((1,)), 0),
                ],
            ),
        ],
    )
    def test_boundary_edges_returns_expected_edges(self, edges, boundary_edges):
        graph = DecodingHyperMultiGraph(edges)
        assert set(graph.boundary_edges) == set(boundary_edges)

    @pytest.mark.parametrize(
        "hypergraph",
        [
            (decoding_hyper_multigraph_with_edges()),
            (decoding_hyper_multigraph_with_hyperedges()),
            (decoding_hyper_multigraph_with_multiedges()),
            (decoding_hyper_multigraph_with_multi_hyperedges()),
        ],
    )
    def test_hyper_multigraph_has_no_boundary(self, hypergraph):
        assert hypergraph.boundaries == frozenset()
        for det in hypergraph.nodes:
            assert not hypergraph.detector_is_boundary(det)

    @pytest.mark.parametrize(
        "graph, detectors, expected_edges",
        [
            (
                decoding_hyper_multigraph_with_multi_hyperedges(),
                (0, 1, 2),
                {(DecodingHyperEdge((0, 1, 2)), 0), (DecodingHyperEdge((0, 1, 2)), 1)},
            ),
            (
                decoding_hyper_multigraph_with_edges(),
                (1, 4),
                {(DecodingHyperEdge((1, 4)), 0)},
            ),
        ],
    )
    def test_get_edges_returns_expected_edges(self, graph, detectors, expected_edges):
        assert set(graph.get_edges(*detectors)) == expected_edges

    @pytest.mark.parametrize(
        "detector, expected_edges",
        [
            (
                0,
                {
                    (DecodingHyperEdge((0, 1, 2)), 0),
                    (DecodingHyperEdge((0, 1, 2)), 1),
                    (DecodingHyperEdge((0, 6)), 0),
                },
            ),
            (6, {(DecodingHyperEdge((0, 6)), 0), (DecodingHyperEdge((2, 3, 6)), 0)}),
        ],
    )
    def test_incident_edges_returns_expected_edges(self, detector, expected_edges):
        graph = decoding_hyper_multigraph_with_multi_hyperedges()
        incident_edges = list(graph.incident_edges(detector))
        assert len(incident_edges) == len(expected_edges)
        assert set(incident_edges) == expected_edges

    @pytest.mark.parametrize(
        "detector, expected_nodes", [(0, {1, 2, 6}), (6, {0, 2, 3})]
    )
    def test_neighbours_returns_expected_edges(self, detector, expected_nodes):
        graph = decoding_hyper_multigraph_with_multi_hyperedges()
        assert set(graph.neighbors(detector)) == expected_nodes

    def test_hyper_multigraph_without_duplicate_edges_gives_expected_parity_check_matrix(
        self,
    ):
        hypergraph = DecodingHyperMultiGraph([(0, 1, 2), (0, 2)])
        np.testing.assert_array_equal(
            hypergraph.to_parity_check_matrix(), np.array([[1, 1], [1, 0], [1, 1]])
        )

    def test_hyper_multigraph_with_duplicate_edges_gives_parity_check_matrix_in_correct_order(
        self,
    ):
        hypergraph = DecodingHyperMultiGraph([(0, 1, 2), (0, 2), (0, 1, 2)])
        np.testing.assert_array_equal(
            hypergraph.to_parity_check_matrix(),
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
        )

    def test_multigraph_without_multi_edges_removes_multiple_edges(self):
        multigraph = DecodingHyperMultiGraph([(0, 1), (0, 2), (0, 1)])
        assert multigraph.with_multi_edges_merged().edges == [
            DecodingHyperEdge({0, 1}),
            DecodingHyperEdge({0, 2}),
        ]

    def test_multigraph_without_multi_edges_preserves_detector_records(self):
        multigraph = DecodingHyperMultiGraph(
            [(0, 1), (0, 2), (0, 1)],
            detector_records={
                0: DetectorRecord((0, 0)),
                1: DetectorRecord((0, 1)),
                2: DetectorRecord((1, 0)),
            },
        )
        assert (
            multigraph.with_multi_edges_merged().detector_records
            == multigraph.detector_records
        )

    def test_multigraph_without_multi_edges_calls_convolve_probabilities_by_default(
        self,
    ):
        """
        This test should be improved by mocking the call to the default edge
        combiner but I couldn't find how to do this properly with pytest-mock.
        """
        edge = DecodingHyperEdge({0, 1})
        multigraph = DecodingHyperMultiGraph(
            [(edge, EdgeRecord(0.01)), (edge, EdgeRecord(0.02))]
        )
        assert math.isclose(
            multigraph.with_multi_edges_merged().edge_records[edge].p_err,
            0.01 * (1 - 0.02) + 0.02 * (1 - 0.01),
        )

    def test_decoding_graph_can_turn_error_into_syndrome(self):
        decoding_graph = DecodingHyperMultiGraph([(0, 1, 2)])
        assert decoding_graph.error_to_syndrome(
            [decoding_graph.edges[0]]
        ) == OrderedSyndrome({0, 1, 2})
