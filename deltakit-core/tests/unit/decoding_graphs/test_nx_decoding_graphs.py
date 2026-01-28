# (c) Copyright Riverlane 2020-2025.
"""Tests for decoding graph datastructure."""

import math
import re
from itertools import chain, combinations, permutations, repeat

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest

from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DecodingHyperEdge,
    DetectorRecord,
    EdgeRecord,
    NXDecodingGraph,
    NXDecodingMultiGraph,
    OrderedSyndrome,
)
from deltakit_core.decoding_graphs._data_qubits import OrderedDecodingEdges


def edge_list_node_list_and_decoding_graph_no_nodes():
    edge_list = []
    decoding_graph = NXDecodingGraph.from_edge_list(edge_list)
    return edge_list, [], decoding_graph


def edge_list_node_list_and_decoding_multi_graph_no_nodes():
    edge_list = []
    decoding_graph = NXDecodingMultiGraph.from_edge_list(edge_list)
    return edge_list, [], decoding_graph


def edge_list_node_list_and_decoding_graph_no_edges():
    edge_list = []
    nodes = {i: DetectorRecord() for i in range(20)}
    decoding_graph = NXDecodingGraph.from_edge_list(edge_list, nodes)
    return edge_list, list(nodes.keys()), decoding_graph


def edge_list_node_list_and_decoding_multi_graph_no_edges():
    edge_list = []
    nodes = {i: DetectorRecord() for i in range(20)}
    decoding_graph = NXDecodingMultiGraph.from_edge_list(edge_list, nodes)
    return edge_list, list(nodes.keys()), decoding_graph


def edge_list_node_list_and_decoding_graph_complete():
    nodes = {i: DetectorRecord() for i in range(20)}
    edge_list = list(combinations(nodes, 2))
    decoding_graph = NXDecodingGraph.from_edge_list(edge_list, nodes)
    return edge_list, list(nodes.keys()), decoding_graph


def edge_list_node_list_and_decoding_multi_graph_all_multiedges():
    nodes = {i: DetectorRecord() for i in range(20)}
    edge_list = list(permutations(nodes, 2))
    decoding_graph = NXDecodingMultiGraph.from_edge_list(edge_list, nodes)
    return edge_list, list(nodes.keys()), decoding_graph


def edge_list_node_list_and_decoding_graph_cyclic():
    nodes = {i: DetectorRecord() for i in range(10)}
    edge_list = [(i, (i + 1) % 10) for i in range(10)]
    decoding_graph = NXDecodingGraph.from_edge_list(edge_list, nodes)
    return edge_list, list(nodes.keys()), decoding_graph


def edge_list_node_list_and_decoding_graph_layered():
    edge_list = [
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
    ]
    nodes = {i: DetectorRecord() for i in range(8)}
    return (
        edge_list,
        list(nodes.keys()),
        NXDecodingGraph.from_edge_list(edge_list, nodes),
    )


def edge_list_node_list_and_decoding_multi_graph_layered():
    edge_list = [
        (0, 1),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 5),
        (5, 6),
        (6, 7),
        (6, 7),
        (7, 4),
        (7, 4),
    ]
    nodes = {i: DetectorRecord() for i in range(8)}
    return (
        edge_list,
        list(nodes.keys()),
        NXDecodingMultiGraph.from_edge_list(edge_list, nodes),
    )


def edge_list_node_list_and_decoding_graph_layered_with_boundaries():
    edge_list = [
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
    ]
    nodes = {i: DetectorRecord() for i in range(8)}
    return (
        edge_list,
        list(nodes.keys()),
        NXDecodingGraph.from_edge_list(edge_list, nodes, boundaries=[2]),
    )


def edge_list_node_list_and_decoding_multi_graph_layered_with_boundaries():
    edge_list = [
        (0, 1),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 5),
        (5, 6),
        (6, 7),
        (6, 7),
        (7, 4),
        (7, 4),
    ]
    nodes = {i: DetectorRecord() for i in range(8)}
    return (
        edge_list,
        list(nodes.keys()),
        NXDecodingMultiGraph.from_edge_list(edge_list, nodes, boundaries=[2]),
    )


def two_line_of_6_decoding_graph_and_non_contiguous_logicals():
    edge_list = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
    ]
    logicals = [
        OrderedDecodingEdges(),
        OrderedDecodingEdges.from_syndrome_indices([(1, 2)]),
        OrderedDecodingEdges(),
        OrderedDecodingEdges.from_syndrome_indices([(8, 9)]),
    ]
    return NXDecodingGraph.from_edge_list(edge_list), logicals


def line_of_6_decoding_graph_and_logicals():
    edge_list = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    logicals = [
        OrderedDecodingEdges.from_syndrome_indices([(1, 2)]),
        OrderedDecodingEdges.from_syndrome_indices([(5, 6)]),
    ]
    return NXDecodingGraph.from_edge_list(edge_list), logicals


def line_of_6_decoding_multigraph_and_logicals():
    edge_list = [(1, 2), (3, 4), (4, 5), (5, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    logicals = [
        OrderedDecodingEdges.from_syndrome_indices([(1, 2)]),
        OrderedDecodingEdges.from_syndrome_indices([(5, 6)]),
    ]
    return NXDecodingMultiGraph.from_edge_list(edge_list), logicals


def line_of_6_decoding_graph_connected_by_boundary_and_logicals():
    edge_list = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    logicals = [
        OrderedDecodingEdges.from_syndrome_indices([(1, 2)]),
        OrderedDecodingEdges.from_syndrome_indices([(3, 4)]),
    ]
    return NXDecodingGraph.from_edge_list(edge_list, boundaries=[4]), logicals


def line_of_6_decoding_multigraph_connected_by_boundary_and_logicals():
    edge_list = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 2), (3, 4), (4, 5), (5, 6)]
    logicals = [
        OrderedDecodingEdges.from_syndrome_indices([(1, 2)]),
        OrderedDecodingEdges.from_syndrome_indices([(3, 4)]),
    ]
    return NXDecodingMultiGraph.from_edge_list(edge_list, boundaries=[4]), logicals


def disconnected_decoding_graph_and_logicals():
    edge_list = [(1, 2), (3, 4), (5, 6)]
    logicals = [
        OrderedDecodingEdges.from_syndrome_indices([(1, 2)]),
        OrderedDecodingEdges.from_syndrome_indices([(5, 6)]),
    ]
    return NXDecodingGraph.from_edge_list(edge_list), logicals


def disconnected_decoding_multigraph_and_logicals():
    edge_list = [(1, 2), (3, 4), (5, 6), (1, 2), (3, 4), (5, 6)]
    logicals = [
        OrderedDecodingEdges.from_syndrome_indices([(1, 2)]),
        OrderedDecodingEdges.from_syndrome_indices([(5, 6)]),
    ]
    return NXDecodingMultiGraph.from_edge_list(edge_list), logicals


def disconnected_cycle_decoding_multigraph_and_logicals():
    edge_list = [(1, 2), (2, 3), (3, 1), (4, 5), (1, 2), (2, 3), (3, 1), (4, 5)]
    logicals = [OrderedDecodingEdges.from_syndrome_indices([(1, 2), (2, 3)])]
    return NXDecodingMultiGraph.from_edge_list(edge_list), logicals


def decoding_graph_and_logicals_not_in_graph():
    edge_list = [(1, 2), (3, 4)]
    logicals = [OrderedDecodingEdges.from_syndrome_indices([(2, 3)])]
    return NXDecodingGraph.from_edge_list(edge_list), logicals


def decoding_multigraph_and_logicals_not_in_graph():
    edge_list = [(1, 2), (3, 4), (1, 2), (3, 4)]
    logicals = [OrderedDecodingEdges.from_syndrome_indices([(2, 3)])]
    return NXDecodingMultiGraph.from_edge_list(edge_list), logicals


def empty_decoding_graph_and_logicals():
    return NXDecodingGraph.from_edge_list([]), []


def empty_decoding_multigraph_and_logicals():
    return NXDecodingMultiGraph.from_edge_list([]), []


def decoding_graph_and_empty_logicals():
    edge_list = [(1, 2), (3, 4), (5, 6)]
    logicals = []
    return NXDecodingGraph.from_edge_list(edge_list), logicals


def decoding_multigraph_and_empty_logicals():
    edge_list = [(1, 2), (3, 4), (5, 6)]
    logicals = []
    return NXDecodingMultiGraph.from_edge_list(edge_list), logicals


def decoding_graph_no_boundaries():
    return NXDecodingGraph.from_edge_list([(i % 10, (i + 3) % 10) for i in range(21)])


def decoding_multi_graph_no_boundaries():
    return NXDecodingMultiGraph.from_edge_list(
        chain.from_iterable(repeat((i % 10, (i + 3) % 10), 2) for i in range(21))
    )


def decoding_graph_boundaries_16():
    return NXDecodingGraph.from_edge_list(
        [(i, (i + 1) % 17) for i in range(17)], boundaries=[16]
    )


def decoding_multi_graph_boundaries_16():
    return NXDecodingMultiGraph.from_edge_list(
        chain.from_iterable(repeat((i, (i + 1) % 17), 2) for i in range(17)),
        boundaries=[16],
    )


def decoding_graph_boundaries_0_4():
    return NXDecodingGraph.from_edge_list(
        [(i, (i + 1) % 10) for i in range(10)], boundaries=[0, 4]
    )


def decoding_multi_graph_boundaries_0_4():
    return NXDecodingMultiGraph.from_edge_list(
        chain.from_iterable(repeat((i, (i + 1) % 10), 2) for i in range(10)),
        boundaries=[0, 4],
    )


class TestAnyNXBasedGraph:
    """Generic tests for arbitrary NXDecodingGraphs and NXDecodingMultiGraphs,
    checking expected behaviour and edge cases.
    """

    @pytest.fixture(
        params=[
            edge_list_node_list_and_decoding_graph_no_nodes(),
            edge_list_node_list_and_decoding_multi_graph_no_nodes(),
            edge_list_node_list_and_decoding_graph_no_edges(),
            edge_list_node_list_and_decoding_multi_graph_no_edges(),
            edge_list_node_list_and_decoding_graph_complete(),
            edge_list_node_list_and_decoding_multi_graph_all_multiedges(),
            edge_list_node_list_and_decoding_graph_cyclic(),
            edge_list_node_list_and_decoding_graph_layered(),
            edge_list_node_list_and_decoding_multi_graph_layered(),
        ]
    )
    def edge_list_node_list_and_decoding_graph(self, request):
        return request.param

    @pytest.fixture(
        params=[
            line_of_6_decoding_graph_and_logicals(),
            line_of_6_decoding_multigraph_and_logicals(),
            disconnected_decoding_graph_and_logicals(),
            disconnected_decoding_multigraph_and_logicals(),
            disconnected_cycle_decoding_multigraph_and_logicals(),
            decoding_graph_and_logicals_not_in_graph(),
            decoding_multigraph_and_logicals_not_in_graph(),
            empty_decoding_graph_and_logicals(),
            empty_decoding_multigraph_and_logicals(),
            decoding_graph_and_empty_logicals(),
            decoding_multigraph_and_empty_logicals(),
            line_of_6_decoding_graph_connected_by_boundary_and_logicals(),
            line_of_6_decoding_multigraph_connected_by_boundary_and_logicals(),
            two_line_of_6_decoding_graph_and_non_contiguous_logicals(),
        ]
    )
    def decoding_graph_and_logicals(self, request):
        return request.param

    def test_all_nodes_present(self, edge_list_node_list_and_decoding_graph):
        _, node_list, decoding_graph = edge_list_node_list_and_decoding_graph
        assert set(decoding_graph.nodes) == set(node_list)

    @pytest.mark.parametrize(
        ("graph", "expected_boundaries"),
        [
            (decoding_graph_boundaries_0_4(), {0, 4}),
            (decoding_graph_no_boundaries(), set()),
            (decoding_graph_boundaries_16(), {16}),
            (decoding_multi_graph_boundaries_0_4(), {0, 4}),
            (decoding_multi_graph_no_boundaries(), set()),
            (decoding_multi_graph_boundaries_16(), {16}),
        ],
    )
    def test_graph_with_boundaries_returns_true_for_expected_nodes(
        self, graph, expected_boundaries
    ):
        returned_boundaries = set()
        for node in graph.nodes:
            if graph.detector_is_boundary(node):
                returned_boundaries.add(node)
        assert returned_boundaries == expected_boundaries

    @pytest.mark.parametrize(
        ("graph", "boundaries"),
        [
            (decoding_graph_boundaries_0_4(), [0, 4]),
            (decoding_multi_graph_boundaries_0_4(), [0, 4]),
            (decoding_graph_boundaries_16(), [16]),
            (decoding_multi_graph_boundaries_16(), [16]),
        ],
    )
    def test_boundary_view_does_not_contain_boundary_nodes(self, graph, boundaries):
        assert all(node not in graph.no_boundary_view for node in boundaries)

    @pytest.mark.parametrize(
        ("graph", "boundary_edges"),
        [
            (decoding_graph_boundaries_0_4(), [(0, 1), (3, 4)]),
            (
                decoding_multi_graph_boundaries_0_4(),
                [(0, 1, 0), (3, 4, 0), (0, 1, 1), (3, 4, 1)],
            ),
            (decoding_graph_boundaries_16(), [(15, 16)]),
            (decoding_multi_graph_boundaries_16(), [(15, 16, 0), (15, 16, 1)]),
        ],
    )
    def test_edges_connected_to_boundary_not_in_no_boundary_view(
        self, graph, boundary_edges
    ):
        assert all(edge not in graph.no_boundary_view.edges for edge in boundary_edges)

    @pytest.mark.parametrize(
        ("edges_nodes_graph", "syndrome_bit", "expected_neighbours"),
        [
            (edge_list_node_list_and_decoding_graph_layered(), 2, {0, 1, 3, 6}),
            (edge_list_node_list_and_decoding_multi_graph_layered(), 2, {0, 1, 3, 6}),
            (edge_list_node_list_and_decoding_graph_layered(), 0, {1, 2, 3, 4}),
            (edge_list_node_list_and_decoding_multi_graph_layered(), 0, {1, 2, 3, 4}),
        ],
    )
    def test_neighbour_nodes_found(
        self, edges_nodes_graph, syndrome_bit, expected_neighbours
    ):
        _, _, decoding_graph = edges_nodes_graph
        neighbours = set(decoding_graph.neighbors(syndrome_bit))
        assert neighbours == expected_neighbours

    @pytest.mark.parametrize(
        ("edges_nodes_graph", "endpoints", "expected_path"),
        [
            (
                edge_list_node_list_and_decoding_graph_layered(),
                (0, 6),
                {(0, 2), (2, 6)},
            ),
            (
                edge_list_node_list_and_decoding_multi_graph_layered(),
                (0, 6),
                {(0, 2), (2, 6)},
            ),
            (edge_list_node_list_and_decoding_graph_layered(), (1, 1), {}),
            (edge_list_node_list_and_decoding_multi_graph_layered(), (1, 1), {}),
        ],
    )
    def test_shortest_path_found(self, edges_nodes_graph, endpoints, expected_path):
        _, _, decoding_graph = edges_nodes_graph
        shortest_path = decoding_graph.shortest_path(endpoints[0], endpoints[1])
        assert set(shortest_path) == {DecodingEdge(u, v) for u, v in expected_path}

    @pytest.mark.parametrize(
        ("edges_nodes_graph", "endpoints", "expected_length"),
        [
            (edge_list_node_list_and_decoding_graph_layered(), (0, 6), 2),
            (edge_list_node_list_and_decoding_multi_graph_layered(), (0, 6), 2),
            (edge_list_node_list_and_decoding_graph_layered(), (1, 1), 0),
            (edge_list_node_list_and_decoding_multi_graph_layered(), (1, 1), 0),
        ],
    )
    def test_shortest_path_length_found(
        self, edges_nodes_graph, endpoints, expected_length
    ):
        _, _, decoding_graph = edges_nodes_graph
        path_length = decoding_graph.shortest_path_length(endpoints[0], endpoints[1])
        assert path_length == expected_length

    @pytest.mark.parametrize("graph_class", [NXDecodingGraph, NXDecodingMultiGraph])
    def test_graph_constructor_raises_error_for_invalid_edge_data(self, graph_class):
        edge_data = [(0, 1), (EdgeRecord(0.1), DecodingEdge(0, 3))]
        with pytest.raises(
            TypeError,
            match=f"Unsupported data type for edge {re.escape(str(edge_data[1]))}",
        ):
            graph_class.from_edge_list(edge_data)

    @pytest.mark.parametrize(
        ("edges_nodes_graph", "endpoints", "expected_path"),
        [
            (
                edge_list_node_list_and_decoding_graph_layered_with_boundaries(),
                (0, 6),
                {(0, 1), (5, 6), (1, 5)},
            ),
            (
                edge_list_node_list_and_decoding_multi_graph_layered_with_boundaries(),
                (0, 6),
                {(0, 1), (5, 6), (1, 5)},
            ),
            (
                edge_list_node_list_and_decoding_graph_layered_with_boundaries(),
                (1, 1),
                {},
            ),
            (
                edge_list_node_list_and_decoding_multi_graph_layered_with_boundaries(),
                (1, 1),
                {},
            ),
        ],
    )
    def test_shortest_path_no_boundaries_found(
        self, edges_nodes_graph, endpoints, expected_path
    ):
        _, _, decoding_graph = edges_nodes_graph
        shortest_path = decoding_graph.shortest_path_no_boundaries(
            endpoints[0], endpoints[1]
        )
        assert set(shortest_path) == {DecodingEdge(u, v) for u, v in expected_path}

    @pytest.mark.parametrize(
        ("edges_nodes_graph", "endpoints", "expected_length"),
        [
            (
                edge_list_node_list_and_decoding_graph_layered_with_boundaries(),
                (0, 6),
                3,
            ),
            (
                edge_list_node_list_and_decoding_multi_graph_layered_with_boundaries(),
                (0, 6),
                3,
            ),
            (
                edge_list_node_list_and_decoding_graph_layered_with_boundaries(),
                (1, 1),
                0,
            ),
            (
                edge_list_node_list_and_decoding_multi_graph_layered_with_boundaries(),
                (1, 1),
                0,
            ),
        ],
    )
    def test_shortest_path_length_no_boundaries_found(
        self, edges_nodes_graph, endpoints, expected_length
    ):
        _, _, decoding_graph = edges_nodes_graph
        path_length = decoding_graph.shortest_path_length_no_boundaries(
            endpoints[0], endpoints[1]
        )
        assert path_length == expected_length

    def test_boundaries_not_in_nodes_raises_error(self):
        with pytest.raises(
            ValueError,
            match=r"Boundaries \[17\] are not in nodes \[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,"
            r" 13, 14, 15, 16\].",
        ):
            NXDecodingGraph.from_edge_list(
                [(i, (i + 1) % 17) for i in range(17)], boundaries=[17]
            )

    def test_logicals_in_relevant_nodes(self, decoding_graph_and_logicals):
        graph, logicals = decoding_graph_and_logicals
        relevant_nodes = graph.get_relevant_nodes(logicals)
        for log in logicals:
            log_nodes = {node for edge in log for node in edge.vertices}
            assert all(node in relevant_nodes for node in log_nodes)

    def test_relevant_nodes_are_connected_components(self, decoding_graph_and_logicals):
        graph, logicals = decoding_graph_and_logicals
        relevant_nodes = graph.get_relevant_nodes(logicals)
        components = nx.connected_components(graph.no_boundary_view)
        for component in components:
            num_component_relevant_nodes = len(relevant_nodes.intersection(component))
            # check either entirety or none of component is relevant
            assert (
                num_component_relevant_nodes == len(component)
                or num_component_relevant_nodes == 0
            )

    def test_relevant_nodes_are_in_original_graph(self, decoding_graph_and_logicals):
        graph, logicals = decoding_graph_and_logicals
        relevant_nodes = graph.get_relevant_nodes(logicals)
        assert all(node in graph.nodes for node in relevant_nodes)

    def test_nodes_connected_by_boundary_are_not_relevant(self):
        graph_with_boundary, logicals = (
            line_of_6_decoding_graph_connected_by_boundary_and_logicals()
        )
        graph_no_boundary, _ = line_of_6_decoding_graph_and_logicals()
        # check strict subset
        assert graph_with_boundary.get_relevant_nodes(
            logicals
        ) < graph_no_boundary.get_relevant_nodes(logicals)


class TestNXGraphMutability:
    """Tests on `NXDecodingGraph` and `NXDecodingMultiGraph` to verify that appropriate
    data structures are (im)mutable.
    """

    @pytest.fixture(
        params=[decoding_graph_boundaries_0_4(), decoding_multi_graph_boundaries_0_4()]
    )
    def example_graph(self, request):
        return request.param

    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_nodes_cannot_be_added_to_immutable_graph(
        self, example_graph, immutable_graph_property
    ):
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            getattr(example_graph, immutable_graph_property).add_node(200)
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            getattr(example_graph, immutable_graph_property).add_nodes_from([200])

    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_edges_cannot_be_added_to_immutable_graph(
        self, example_graph, immutable_graph_property
    ):
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            getattr(example_graph, immutable_graph_property).add_edge(20, 30)
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            getattr(example_graph, immutable_graph_property).add_edges_from([(20, 30)])

    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_immutable_graph_cannot_be_cleared(
        self, example_graph, immutable_graph_property
    ):
        graph_view = getattr(example_graph, immutable_graph_property)
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            graph_view.clear()

    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_immutable_graph_nodes_cannot_be_removed(
        self, example_graph, immutable_graph_property
    ):
        graph_view = getattr(example_graph, immutable_graph_property)
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            graph_view.remove_node(2)
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            graph_view.remove_nodes_from([2])

    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_immutable_graph_node_properties_can_be_modified(
        self, example_graph, immutable_graph_property
    ):
        node_identifier = 2
        example_graph.graph.nodes[node_identifier]["node_property"] = 2
        graph_view = getattr(example_graph, immutable_graph_property)
        node_view = graph_view.nodes[node_identifier]
        node_view["node_property"] = 4
        assert node_view["node_property"] == 4

    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_immutable_graph_node_properties_can_be_added(
        self, example_graph, immutable_graph_property
    ):
        graph_view = getattr(example_graph, immutable_graph_property)
        node_view = graph_view.nodes[2]
        node_view["node_property"] = 4
        assert node_view["node_property"] == 4

    @pytest.mark.parametrize(
        ("graph", "edge_identifier"),
        [
            (decoding_graph_boundaries_0_4(), (1, 2)),
            (decoding_multi_graph_boundaries_0_4(), (1, 2, 0)),
        ],
    )
    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_immutable_graph_edges_cannot_be_cleared_or_removed(
        self, graph, edge_identifier, immutable_graph_property
    ):
        graph_view = getattr(graph, immutable_graph_property)
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            graph_view.clear_edges()
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            graph_view.remove_edge(*edge_identifier)
        with pytest.raises(nx.NetworkXError, match="Frozen graph can't be modified"):
            graph_view.remove_edges_from([edge_identifier])

    @pytest.mark.parametrize(
        ("graph", "edge_identifier"),
        [
            (decoding_graph_boundaries_0_4(), (1, 2)),
            (decoding_multi_graph_boundaries_0_4(), (1, 2, 0)),
        ],
    )
    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_immutable_graph_edge_properties_can_still_be_modified(
        self, graph, edge_identifier, immutable_graph_property
    ):
        graph.graph.edges[edge_identifier]["edge_property"] = 2
        graph_view = getattr(graph, immutable_graph_property)
        edge_view = graph_view.edges[edge_identifier]
        edge_view["edge_property"] = 4
        assert edge_view["edge_property"] == 4

    @pytest.mark.parametrize(
        ("graph", "edge_identifier"),
        [
            (decoding_graph_boundaries_0_4(), (1, 2)),
            (decoding_multi_graph_boundaries_0_4(), (1, 2, 0)),
        ],
    )
    @pytest.mark.parametrize("immutable_graph_property", ["no_boundary_view", "graph"])
    def test_immutable_graph_edge_properties_can_still_be_added(
        self, graph, edge_identifier, immutable_graph_property
    ):
        graph_view = getattr(graph, immutable_graph_property)
        edge_view = graph_view.edges[edge_identifier]
        edge_view["edge_property"] = 1
        assert edge_view["edge_property"] == 1


class TestNXDecodingGraph:
    """Unique test cases for NXDecodingGraph class"""

    @pytest.mark.parametrize(
        "edges_nodes_graph",
        [
            edge_list_node_list_and_decoding_graph_no_nodes(),
            edge_list_node_list_and_decoding_graph_no_edges(),
            edge_list_node_list_and_decoding_graph_complete(),
            edge_list_node_list_and_decoding_graph_cyclic(),
            edge_list_node_list_and_decoding_graph_layered(),
        ],
    )
    def test_all_edges_present_in_graph(self, edges_nodes_graph):
        edge_list, _, decoding_graph = edges_nodes_graph
        assert set(decoding_graph.edges) == {DecodingEdge(u, v) for u, v in edge_list}

    @pytest.mark.parametrize(
        "edge_data",
        [
            [(0, 1), (1, 2)],
            {DecodingEdge(0, 1), DecodingEdge(1, 2)},
            (DecodingEdge(0, 1), (1, 2)),
            [(0, 1), (DecodingEdge(1, 2), EdgeRecord(0))],
        ],
    )
    def test_nx_graph_can_be_constructed_using_different_edge_data_formats(
        self, edge_data
    ):
        reference_edge_data = [
            (DecodingEdge(0, 1), EdgeRecord(0)),
            (DecodingEdge(1, 2), EdgeRecord(0)),
        ]
        reference_graph = NXDecodingGraph.from_edge_list(reference_edge_data)
        graph = NXDecodingGraph.from_edge_list(edge_data)
        assert set(graph.edges) == set(reference_graph.edges)
        assert graph.edge_records == reference_graph.edge_records

    @pytest.mark.parametrize(
        ("graph", "expected_edges"),
        [
            (
                decoding_graph_boundaries_0_4(),
                {
                    DecodingEdge(0, 1),
                    DecodingEdge(5, 4),
                    DecodingEdge(4, 3),
                    DecodingEdge(0, 9),
                },
            ),
            (decoding_graph_no_boundaries(), set()),
            (
                decoding_graph_boundaries_16(),
                {DecodingEdge(15, 16), DecodingEdge(16, 0)},
            ),
        ],
    )
    def test_get_boundary_edges_returns_correct_edges(self, graph, expected_edges):
        assert set(graph.boundary_edges) == expected_edges

    @pytest.mark.parametrize(
        ("edges_nodes_graph", "syndrome_bit", "expected_edges"),
        [
            (
                edge_list_node_list_and_decoding_graph_layered(),
                1,
                {(1, 0), (1, 2), (1, 5)},
            ),
            (
                edge_list_node_list_and_decoding_graph_layered(),
                3,
                {(3, 2), (3, 0), (3, 7)},
            ),
        ],
    )
    def test_incident_edges_found(
        self, edges_nodes_graph, syndrome_bit, expected_edges
    ):
        _, _, decoding_graph = edges_nodes_graph
        incident_edges = {DecodingEdge(u, v) for u, v in expected_edges}
        assert set(decoding_graph.incident_edges(syndrome_bit)) == incident_edges

    @pytest.mark.parametrize(
        ("detectors", "expected_edge"),
        [
            ((0, 1), DecodingEdge(0, 1)),
            ((0, 2), DecodingEdge(0, 2)),
            ((2, 1), DecodingEdge(1, 2)),
        ],
    )
    def test_get_edge_gives_expected_edges(self, detectors, expected_edge):
        _, _, graph = edge_list_node_list_and_decoding_graph_layered()
        assert graph.get_edge(*detectors) == expected_edge

    @pytest.mark.parametrize(
        ("detectors", "expected_edge"),
        [
            ((0, 1), [DecodingEdge(0, 1)]),
            ((0, 2), [DecodingEdge(0, 2)]),
            ((2, 1), [DecodingEdge(1, 2)]),
        ],
    )
    def test_get_edgse_gives_expected_edges(self, detectors, expected_edge):
        _, _, graph = edge_list_node_list_and_decoding_graph_layered()
        assert graph.get_edges(*detectors) == expected_edge

    @pytest.mark.parametrize(
        ("graph", "edge_dets", "expected_record"),
        [
            (
                NXDecodingGraph.from_edge_list(
                    [
                        (DecodingEdge(0, 1), EdgeRecord()),
                        (DecodingEdge(1, 2), EdgeRecord(0.5)),
                    ]
                ),
                (1, 2),
                EdgeRecord(0.5),
            ),
            (
                NXDecodingGraph.from_edge_list(
                    [
                        (DecodingEdge(0, 1), EdgeRecord(0.5)),
                        (DecodingEdge(1, 2), EdgeRecord()),
                    ],
                    boundaries=[0],
                ),
                (0, 1),
                EdgeRecord(0.5),
            ),
            (
                NXDecodingGraph.from_edge_list(
                    [
                        (DecodingEdge(0, 1), EdgeRecord()),
                        (DecodingEdge(1, 2), EdgeRecord(edge_refinement=0.1)),
                    ],
                    boundaries=[0],
                ),
                (1, 2),
                EdgeRecord(edge_refinement=0.1),
            ),
        ],
    )
    def test_get_edge_record_from_edge(
        self, graph: NXDecodingGraph, edge_dets: tuple[int], expected_record: EdgeRecord
    ):
        assert graph.get_edge_record(*edge_dets).data == expected_record.data

    @pytest.mark.parametrize(
        ("nx_edges", "nx_boundaries", "expected_hyper_edges"),
        [
            (
                {(1, 3), (2, 7), (3, 7)},
                [7],
                {(1, 3), (2,), (3,)},
            ),
            ({(1, 2), (2, 3), (3, 1)}, [], {(1, 2), (2, 3), (3, 1)}),
            (set(), frozenset({}), set()),
        ],
    )
    def test_nx_graph_elevated_to_hypergraph_correctly_maps_nodes_and_edges(
        self, nx_edges, nx_boundaries, expected_hyper_edges
    ):
        nx_graph = NXDecodingGraph.from_edge_list(nx_edges, boundaries=nx_boundaries)
        decoding_hypergraph = nx_graph.to_decoding_hypergraph()
        assert {DecodingHyperEdge(edge) for edge in expected_hyper_edges} == set(
            decoding_hypergraph.edges
        )
        assert set(decoding_hypergraph.nodes) == set(nx_graph.nodes) - set(
            nx_boundaries
        )

    def test_nx_graph_with_isolated_node_maps_to_hypergraph(self):
        detector_records = {
            1: DetectorRecord(),
            2: DetectorRecord(),
            3: DetectorRecord(),
            4: DetectorRecord(),
        }
        edges = {(1, 2), (2, 4)}
        boundaries = [4]
        nx_graph = NXDecodingGraph.from_edge_list(
            edges, detector_records=detector_records, boundaries=boundaries
        )
        hypergraph = nx_graph.to_decoding_hypergraph()
        assert set(hypergraph.nodes) == {1, 2, 3}
        assert {frozenset(edge.vertices) for edge in hypergraph.edges} == {
            frozenset({1, 2}),
            frozenset(
                {
                    2,
                }
            ),
        }

    def test_graph_edge_records_are_independently_mutable(self):
        edge_list = [(0, 1), (1, 2), (2, 0)]
        decoding_graph = NXDecodingGraph.from_edge_list(edge_list)
        # modify the EdgeRecord for a select edge
        target_edge = DecodingEdge(0, 1)
        modified_edge_record = EdgeRecord(p_err=0.9)
        # verify that only one EdgeRecord is changed
        decoding_graph.edge_records[target_edge] = modified_edge_record
        other_edge_records = [
            edge_record
            for edge, edge_record in decoding_graph.edge_records.items()
            if edge != target_edge
        ]
        assert modified_edge_record not in other_edge_records
        # verify that inner NX graph EdgeRecords are also independent
        decoding_graph.graph.edges[0, 1].update(dict(modified_edge_record))
        other_edge_dicts = [
            decoding_graph.graph[u][v]
            for (u, v) in decoding_graph.edges
            if DecodingEdge(u, v) != target_edge
        ]
        assert dict(modified_edge_record) not in other_edge_dicts

    @pytest.mark.parametrize(
        ("graph", "expected_parity_check_matrix"),
        [
            (
                NXDecodingGraph.from_edge_list([(0, 1), (1, 2)]),
                np.array([[True, False], [True, True], [False, True]]),
            ),
            (
                NXDecodingGraph.from_edge_list([(0, 1), (0, 2)], boundaries={0}),
                np.array([[True, True], [True, False], [False, True]]),
            ),
        ],
    )
    def test_known_graph_gives_expected_parity_check_graph(
        self,
        graph: NXDecodingGraph,
        expected_parity_check_matrix: npt.NDArray[np.bool_],
    ):
        np.testing.assert_array_equal(
            graph.to_parity_check_matrix(), expected_parity_check_matrix
        )

    def test_decoding_graph_can_turn_error_into_syndrome(self):
        decoding_graph = NXDecodingGraph.from_edge_list([(0, 1)])
        assert decoding_graph.error_to_syndrome(
            [decoding_graph.edges[0]]
        ) == OrderedSyndrome({0, 1})

    def test_boundaries_are_not_in_syndrome_when_converting_from_error(self):
        decoding_graph = NXDecodingGraph.from_edge_list([(0, 1)], boundaries={0})
        assert decoding_graph.error_to_syndrome(
            [decoding_graph.edges[0]]
        ) == OrderedSyndrome({1})


class TestNXDecodingMultiGraph:
    """Unique tests for `NXDecodingMultiGraph`."""

    @pytest.mark.parametrize(
        "edges_nodes_graph",
        [
            edge_list_node_list_and_decoding_multi_graph_no_nodes(),
            edge_list_node_list_and_decoding_multi_graph_no_edges(),
            edge_list_node_list_and_decoding_multi_graph_all_multiedges(),
            edge_list_node_list_and_decoding_multi_graph_layered(),
        ],
    )
    def test_all_edges_present_in_multi_graph(self, edges_nodes_graph):
        edge_list, _, decoding_graph = edges_nodes_graph
        edges = sorted([tuple(sorted((u, v))) for (u, v), _ in decoding_graph.edges])
        assert edges == sorted([tuple(sorted((u, v))) for u, v in edge_list])

    @pytest.mark.parametrize(
        "edge_data",
        [
            [(0, 1), (1, 2), (0, 1)],
            [DecodingEdge(0, 1), DecodingEdge(1, 2), DecodingEdge(0, 1)],
            (DecodingEdge(0, 1), (1, 2), (0, 1)),
            [(0, 1), (1, 2), (DecodingEdge(0, 1), EdgeRecord(0))],
        ],
    )
    def test_nx_multigraph_can_be_constructed_using_different_edge_data_formats(
        self, edge_data
    ):
        reference_edge_data = [
            (DecodingEdge(0, 1), EdgeRecord(0)),
            (DecodingEdge(0, 1), EdgeRecord(0)),
            (DecodingEdge(1, 2), EdgeRecord(0)),
        ]
        reference_graph = NXDecodingMultiGraph.from_edge_list(reference_edge_data)
        graph = NXDecodingMultiGraph.from_edge_list(edge_data)
        assert graph.edges == reference_graph.edges
        assert graph.edge_records == reference_graph.edge_records

    @pytest.mark.parametrize(
        ("graph", "expected_edges"),
        [
            (
                decoding_multi_graph_boundaries_0_4(),
                [(0, 1), (4, 5), (3, 4), (0, 9), (0, 1), (4, 5), (3, 4), (0, 9)],
            ),
            (decoding_multi_graph_no_boundaries(), []),
            (
                decoding_multi_graph_boundaries_16(),
                [(15, 16), (0, 16), (15, 16), (0, 16)],
            ),
        ],
    )
    def test_boundary_edges_returns_correct_multi_edges(self, graph, expected_edges):
        boundary_edges = sorted(
            [tuple(sorted((u, v))) for (u, v), _ in graph.boundary_edges]
        )
        assert sorted(boundary_edges) == sorted(expected_edges)

    @pytest.mark.parametrize(
        ("syndrome_bit", "expected_edges"),
        [
            (1, [((0, 1), 0), ((1, 2), 0), ((1, 5), 0), ((0, 1), 1)]),
            (3, [((2, 3), 0), ((0, 3), 0), ((3, 7), 0), ((2, 3), 1)]),
            (0, [((0, 1), 0), ((0, 1), 1), ((0, 2), 0), ((0, 3), 0), ((0, 4), 0)]),
            (1, [((0, 1), 0), ((0, 1), 1), ((1, 2), 0), ((1, 5), 0)]),
            (2, [((0, 2), 0), ((1, 2), 0), ((2, 3), 0), ((2, 3), 1), ((2, 6), 0)]),
        ],
    )
    def test_incident_edges_found(self, syndrome_bit, expected_edges):
        _, _, decoding_graph = edge_list_node_list_and_decoding_multi_graph_layered()
        incident_edges = [
            (tuple(sorted((u, v))), idx)
            for (u, v), idx in decoding_graph.incident_edges(syndrome_bit)
        ]
        assert sorted(expected_edges) == sorted(incident_edges)

    @pytest.mark.parametrize(
        ("detectors", "expected_edges"),
        [
            ((0, 1), {(DecodingEdge(0, 1), 0), (DecodingEdge(0, 1), 1)}),
            ((0, 2), {(DecodingEdge(0, 2), 0)}),
            ((2, 1), {(DecodingEdge(1, 2), 0)}),
        ],
    )
    def test_multigraph_get_edges_gives_expected_edges(self, detectors, expected_edges):
        _, _, graph = edge_list_node_list_and_decoding_multi_graph_layered()
        assert set(graph.get_edges(*detectors)) == expected_edges

    def test_multigraph_edge_records_are_independently_mutable(self):
        edge_list = [(0, 1), (0, 1), (1, 2), (2, 0)]
        decoding_graph = NXDecodingMultiGraph.from_edge_list(edge_list)
        # modify the EdgeRecord for a select edge
        target_edge = (DecodingEdge(0, 1), 0)
        modified_edge_record = EdgeRecord(p_err=0.9)
        # verify that only one EdgeRecord is changed
        decoding_graph.edge_records[target_edge] = modified_edge_record
        other_edge_records = [
            edge_record
            for edge, edge_record in decoding_graph.edge_records.items()
            if edge != target_edge
        ]
        assert modified_edge_record not in other_edge_records
        # verify that inner NX graph EdgeRecords are also independent
        decoding_graph.graph.edges[0, 1, 0].update(dict(modified_edge_record))
        other_edge_dicts = [
            decoding_graph.graph[u][v][k]
            for (u, v), k in decoding_graph.edges
            if (DecodingEdge(u, v), k) != target_edge
        ]
        assert dict(modified_edge_record) not in other_edge_dicts

    @pytest.mark.parametrize(
        ("multigraph", "expected_parity_check_matrix"),
        [
            (
                NXDecodingMultiGraph.from_edge_list([(0, 1), (1, 2)]),
                np.array([[1, 0], [1, 1], [0, 1]]),
            ),
            (
                NXDecodingMultiGraph.from_edge_list([(0, 1), (0, 2)], boundaries={0}),
                np.array([[1, 1], [1, 0], [0, 1]]),
            ),
        ],
    )
    def test_multigraph_instance_without_multiedges_has_expected_parity_check_matrix(
        self,
        multigraph: NXDecodingMultiGraph,
        expected_parity_check_matrix: npt.NDArray[np.bool_],
    ):
        np.testing.assert_array_equal(
            multigraph.to_parity_check_matrix(), expected_parity_check_matrix
        )

    @pytest.mark.parametrize(
        ("multigraph", "expected_parity_check_matrix"),
        [
            (
                NXDecodingMultiGraph.from_edge_list([(0, 1), (0, 1)]),
                np.array([[1, 1], [1, 1]]),
            ),
            (
                NXDecodingMultiGraph.from_edge_list([(0, 2), (0, 1), (0, 2)]),
                np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]]),
            ),
            (
                NXDecodingMultiGraph.from_edge_list(
                    [(0, 2), (0, 1), (0, 2)], boundaries={2}
                ),
                np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]]),
            ),
        ],
    )
    def test_multigraph_instance_with_multiedges_has_expected_parity_check_matrix(
        self,
        multigraph: NXDecodingMultiGraph,
        expected_parity_check_matrix: npt.NDArray[np.bool_],
    ):
        np.testing.assert_array_equal(
            multigraph.to_parity_check_matrix(), expected_parity_check_matrix
        )

    def test_multigraph_without_multi_edges_removes_multiple_edges(self):
        multigraph = NXDecodingMultiGraph.from_edge_list([(0, 1), (0, 2), (0, 1)])
        assert multigraph.with_multi_edges_merged().edges == [
            DecodingEdge(0, 1),
            DecodingEdge(0, 2),
        ]

    def test_multigraph_without_multi_edges_preserves_detector_records(self):
        multigraph = NXDecodingMultiGraph.from_edge_list(
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

    def test_multigraph_without_multi_edge_preserves_boundaries(self):
        multigraph = NXDecodingMultiGraph.from_edge_list(
            [(0, 1), (0, 2), (0, 1)], boundaries=(0,)
        )
        assert multigraph.with_multi_edges_merged().boundaries == multigraph.boundaries

    def test_multigraph_without_multi_edges_calls_convolve_probabilities_by_default(
        self,
    ):
        """
        This test should be improved by mocking the call to the default edge
        combiner but I couldn't find how to do this properly with pytest-mock.
        """
        edge = DecodingEdge(0, 1)
        multigraph = NXDecodingMultiGraph.from_edge_list(
            [(edge, EdgeRecord(0.01)), (edge, EdgeRecord(0.02))]
        )
        assert math.isclose(
            multigraph.with_multi_edges_merged().edge_records[edge].p_err,
            0.01 * (1 - 0.02) + 0.02 * (1 - 0.01),
        )

    def test_decoding_multi_graph_can_turn_error_into_syndrome(self):
        decoding_graph = NXDecodingMultiGraph.from_edge_list([(0, 1)])
        assert decoding_graph.error_to_syndrome(
            [decoding_graph.edges[0]]
        ) == OrderedSyndrome({0, 1})

    def test_boundaries_are_not_in_syndrome_when_converting_from_error(self):
        decoding_graph = NXDecodingMultiGraph.from_edge_list([(0, 1)], boundaries={0})
        assert decoding_graph.error_to_syndrome(
            [decoding_graph.edges[0]]
        ) == OrderedSyndrome({1})
