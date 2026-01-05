# (c) Copyright Riverlane 2020-2025.
import math

import pytest
import stim

from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DecodingHyperEdge,
    OrderedDecodingEdges,
    OrderedSyndrome,
    dem_to_decoding_graph_and_logicals,
    dem_to_hypergraph_and_logicals,
)


def dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed_hyper():
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=1,
        distance=3,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.1,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=True)
    nodes = OrderedSyndrome(range(8))
    edges = {
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (0, 2),
        (2, 1),
        (1, 3),
        (0, 4),
        (2, 6),
        (1, 5),
        (3, 7),
        (4, 6),
        (6, 5),
        (5, 7),
    }
    logicals = [
        {
            DecodingHyperEdge([1]),
            DecodingHyperEdge([4]),
            DecodingHyperEdge([5]),
            DecodingHyperEdge([0]),
        }
    ]
    return dem, nodes, edges, logicals


def dem_nodes_edges_logials_repetition_code_4_round_decomposed_hyper():
    stim_circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=4,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=True)
    nodes = OrderedSyndrome(range(20))
    edges = {
        (8, 4),
        (0, 1),
        (1, 2),
        (2, 3),
        (11,),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
        (9, 5),
        (10, 6),
        (11, 7),
        (12, 8),
        (13, 9),
        (14, 10),
        (15, 11),
        (16, 12),
        (8, 9),
        (9, 10),
        (13, 14),
        (10, 11),
        (12,),
        (7,),
        (0,),
        (4,),
        (8,),
        (12, 13),
        (14, 15),
        (16, 17),
        (17, 18),
        (18, 19),
        (19,),
        (16,),
        (19, 15),
        (18, 14),
        (17, 13),
        (15,),
        (3,),
    }
    logicals = [
        {
            DecodingHyperEdge([19]),
            DecodingHyperEdge([7]),
            DecodingHyperEdge([15]),
            DecodingHyperEdge([11]),
            DecodingHyperEdge([3]),
        }
    ]
    return dem, nodes, edges, logicals


def dem_nodes_edges_logicals_RP_3x3_X_2_round_not_decomposed():
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=2,
        distance=3,
        before_round_data_depolarization=0.03,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=False)
    nodes = OrderedSyndrome(range(16))
    edges = {
        (11,),
        (6,),
        (9, 6),
        (10, 6, 7),
        (8, 9),
        (0, 2),
        (4, 5, 6),
        (10, 5),
        (4, 5),
        (4, 6),
        (2,),
        (10, 11),
        (9, 10, 5, 6),
        (10, 7),
        (1, 3),
        (3,),
        (1,),
        (9, 10, 11),
        (9, 11),
        (0,),
        (4,),
        (7,),
        (10,),
        (9,),
        (6, 7),
        (5, 8, 9),
        (5,),
        (8,),
        (8, 5),
        (1, 2),
    }
    logicals = [
        {
            DecodingHyperEdge([4]),
            DecodingHyperEdge([8, 9]),
            DecodingHyperEdge([0]),
            DecodingHyperEdge([4, 5]),
            DecodingHyperEdge([9]),
            DecodingHyperEdge([8, 9, 5]),
            DecodingHyperEdge([1]),
        }
    ]
    return dem, nodes, edges, logicals


def dem_nodes_edges_logicals_P_3x3_Z_2_round_not_decomposed():
    stim_circuit = stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        rounds=2,
        distance=3,
        before_round_data_depolarization=0.03,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=False)
    nodes = OrderedSyndrome(range(24))
    edges = {
        (0, 1),
        (0, 2),
        (0,),
        (1,),
        (1, 3),
        (2, 3),
        (2, 4),
        (2,),
        (3,),
        (3, 5),
        (4, 5),
        (4,),
        (5,),
        (6,),
        (6, 7),
        (6, 7, 9),
        (6, 8, 9, 11),
        (6, 8),
        (6, 11),
        (7,),
        (7, 9, 10, 12),
        (7, 10),
        (7, 12),
        (8, 9),
        (8, 11, 13),
        (8, 13),
        (8,),
        (9, 10),
        (9, 11, 12, 14),
        (9, 14),
        (9,),
        (10, 12, 15),
        (10, 15),
        (10,),
        (11,),
        (11, 12),
        (11, 13, 14, 16),
        (11, 16),
        (12,),
        (12, 14, 15, 17),
        (12, 17),
        (13,),
        (13, 14),
        (13, 16),
        (14,),
        (14, 15),
        (14, 16, 17),
        (15,),
        (15, 17),
        (16,),
        (16, 17),
        (17,),
    }
    logicals = [
        {
            DecodingHyperEdge([8, 6]),
            DecodingHyperEdge([9]),
            DecodingHyperEdge([2]),
            DecodingHyperEdge([0]),
            DecodingHyperEdge([9, 6, 7]),
            DecodingHyperEdge([4]),
            DecodingHyperEdge([8]),
            DecodingHyperEdge([10, 7]),
            DecodingHyperEdge([10]),
        }
    ]
    return dem, nodes, edges, logicals


def dem_nodes_edges_logicals_2_round_dist_3_colorcode_not_decomposed():
    stim_circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        rounds=2,
        distance=3,
        before_round_data_depolarization=0.03,
        before_measure_flip_probability=0.1,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=False)
    nodes = OrderedSyndrome(range(6))
    edges = {
        (0, 1, 2),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 3, 4),
        (0, 1),
        (0, 2),
        (0, 2, 3, 5),
        (0, 3),
        (0,),
        (1, 2),
        (1, 2, 4, 5),
        (1, 4),
        (1,),
        (2,),
        (2, 5),
        (3, 4, 5),
        (3, 4),
        (3, 5),
        (3,),
        (4, 5),
        (4,),
        (5,),
    }
    logicals = [
        {
            DecodingHyperEdge([0]),
            DecodingHyperEdge([3, 4]),
            DecodingHyperEdge([3]),
            DecodingHyperEdge([4]),
            DecodingHyperEdge([0, 1]),
            DecodingHyperEdge([1]),
        }
    ]
    return dem, nodes, edges, logicals


def dem_nodes_edges_logicals_repetition_code_4_round_decomposed():
    stim_circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=4,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=True)
    nodes = OrderedSyndrome(range(21))
    edge_indices = {
        (8, 4),
        (0, 1),
        (1, 2),
        (2, 3),
        (20, 11),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
        (9, 5),
        (10, 6),
        (11, 7),
        (12, 8),
        (13, 9),
        (14, 10),
        (15, 11),
        (16, 12),
        (8, 9),
        (9, 10),
        (13, 14),
        (10, 11),
        (20, 12),
        (20, 7),
        (20, 0),
        (20, 4),
        (20, 8),
        (12, 13),
        (14, 15),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (16, 20),
        (19, 15),
        (18, 14),
        (17, 13),
        (20, 15),
        (20, 3),
    }
    edges = OrderedDecodingEdges.from_syndrome_indices(edge_indices)
    logical_edges = [(19, 20), (20, 7), (20, 15), (11, 20), (3, 20)]
    logicals = [set(OrderedDecodingEdges.from_syndrome_indices(logical_edges))]
    return dem, nodes, edges, logicals


@pytest.fixture(scope="module")
def dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed():
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=1,
        distance=3,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.1,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=True)
    nodes = OrderedSyndrome(range(9))
    edge_indices = {
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
        (8, 6),
        (8, 7),
        (0, 2),
        (2, 1),
        (1, 3),
        (0, 4),
        (2, 6),
        (1, 5),
        (3, 7),
        (4, 6),
        (6, 5),
        (5, 7),
    }
    edges = OrderedDecodingEdges.from_syndrome_indices(edge_indices)
    logical_edges = [(8, 1), (8, 4), (8, 5), (0, 8)]
    logicals = [set(OrderedDecodingEdges.from_syndrome_indices(logical_edges))]
    return dem, nodes, edges, logicals


def dem_nodes_edges_logicals_RP_3x3_Z_2_round_decomposed():
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=2,
        distance=3,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.1,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=True)
    nodes = OrderedSyndrome(range(17))
    edge_indices = {
        (0, 16),
        (1, 16),
        (2, 16),
        (3, 16),
        (4, 16),
        (5, 16),
        (6, 16),
        (7, 16),
        (8, 16),
        (9, 16),
        (10, 16),
        (11, 16),
        (12, 16),
        (13, 16),
        (14, 16),
        (15, 16),
        (9, 11),
        (0, 8),
        (2, 10),
        (0, 1),
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 7),
        (5, 13),
        (7, 15),
        (12, 13),
        (13, 14),
        (14, 15),
        (5, 13),
        (10, 14),
        (4, 6),
        (10, 5),
        (10, 7),
        (8, 5),
        (8, 12),
        (9, 6),
    }
    edges = OrderedDecodingEdges.from_syndrome_indices(edge_indices)
    logical_edges = [(16, 15), (16, 7), (16, 1), (16, 3), (16, 13), (16, 5)]
    logicals = [set(OrderedDecodingEdges.from_syndrome_indices(logical_edges))]
    return dem, nodes, edges, logicals


@pytest.fixture(
    params=[dem_to_decoding_graph_and_logicals, dem_to_hypergraph_and_logicals]
)
def dem_to_graph_method(request):
    return request.param


def test_there_are_no_multi_edges_in_the_graph(dem_to_graph_method):
    dem = stim.DetectorErrorModel("\n".join(["error(0.01) D0 D1", "error(0.02) D0 D1"]))
    graph, _ = dem_to_graph_method(dem)
    assert len(graph.edges) == len({edge.vertices for edge in graph.edges})


@pytest.mark.parametrize(
    "dem",
    [
        stim.DetectorErrorModel(
            "\n".join(["error(0.01) D0 D1 L0", "error(0.02) D0 D1 L0"])
        ),
        stim.DetectorErrorModel(
            "\n".join(
                ["error(0.01) D0 D1 L0", "error(0.06) D0 D1", "error(0.02) D0 D1 L0"]
            )
        ),
        stim.DetectorErrorModel(
            "\n".join(
                [
                    "error(0.01) D0 D1 L0",
                    "error(0.03) D0 D1",
                    "error(0.02) D0 D1 L0",
                    "error(0.04) D0 D1",
                ]
            )
        ),
        stim.DetectorErrorModel(
            "\n".join(
                [
                    "error(0.01) D0 D1 L1",
                    "error(0.02) D0 D1 L0",
                    "error(0.03) D0 D1 L1",
                ]
            )
        ),
    ],
)
def test_there_are_no_multi_edges_in_the_logicals(dem, dem_to_graph_method):
    _, logicals = dem_to_graph_method(dem)
    for logical in logicals:
        assert len(logical) == len({edge.vertices for edge in logical})


@pytest.mark.parametrize(
    ("dem", "expected_weight"),
    [
        (stim.DetectorErrorModel("error(0.1) D0 D1"), 2.1972245773362196),
        (
            stim.DetectorErrorModel(
                "\n".join(["error(0.1) D0 D1", "error(0.2) D0 D1"])
            ),
            1.0459685551826876,
        ),
    ],
)
def test_edge_weights_correctly_calculated(
    dem_to_graph_method, dem: stim.DetectorErrorModel, expected_weight: float
):
    graph, _ = dem_to_graph_method(dem)
    assert math.isclose(graph.edge_records[graph.edges[0]].weight, expected_weight)


def test_dem_with_coordinateless_detectors_adds_nodes_to_graph(dem_to_graph_method):
    graph, _ = dem_to_graph_method(stim.DetectorErrorModel("detector D0"))
    assert 0 in graph.detector_records


def test_dem_with_single_coordinate_detector_has_correct_detector_record(
    dem_to_graph_method,
):
    graph, _ = dem_to_graph_method(stim.DetectorErrorModel("detector(3) D0"))
    assert graph.detector_records[0].time == 3


class TestDemToDecodingHyperGraph:
    """Tests for .dem -> hypergraph conversion. Includes test cases using
    .dem files both with and without hyperedges.
    """

    @pytest.fixture(
        params=[
            dem_nodes_edges_logials_repetition_code_4_round_decomposed_hyper(),
            dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed_hyper(),
            dem_nodes_edges_logicals_RP_3x3_X_2_round_not_decomposed(),
            dem_nodes_edges_logicals_P_3x3_Z_2_round_not_decomposed(),
            dem_nodes_edges_logicals_2_round_dist_3_colorcode_not_decomposed(),
        ]
    )
    def dem_nodes_edges(self, request):
        return request.param

    def test_nx_function_returns_expected_logicals(self, dem_nodes_edges):
        dem, _, _, expected_logicals = dem_nodes_edges
        _, logicals = dem_to_hypergraph_and_logicals(dem)
        assert logicals == expected_logicals

    def test_hypergraph_has_expected_nodes(self, dem_nodes_edges):
        dem, nodes, _, _ = dem_nodes_edges
        hypergraph, _ = dem_to_hypergraph_and_logicals(dem)
        assert set(hypergraph.nodes) == set(nodes)

    def test_hypergraph_has_expected_edges(self, dem_nodes_edges):
        dem, _, expected_edges, _ = dem_nodes_edges
        hypergraph, _ = dem_to_hypergraph_and_logicals(dem)
        int_edges = {frozenset(idx for idx in edge) for edge in hypergraph.edges}
        expected_int_edges = {frozenset(idx for idx in edge) for edge in expected_edges}
        assert int_edges == expected_int_edges

    def test_untracked_logicals_cause_warning(self):
        dem = stim.DetectorErrorModel(
            """
            error(0.2) D0
            detector(0, 4, 0) D0
            logical_observable L1
            """
        )
        with pytest.warns(
            Warning, match="Isolated logical observables L1 declared in DEM file."
        ):
            dem_to_hypergraph_and_logicals(dem)

    @pytest.mark.skip(
        reason="Still considering what the weight of such edges should be"
    )
    def test_edges_with_zero_p_err_are_added_to_the_graph(self):
        dem = stim.DetectorErrorModel("error(0) D0 D1")
        graph, _ = dem_to_hypergraph_and_logicals(dem)
        assert DecodingHyperEdge({0, 1}, p_err=0) in graph.edges

    @pytest.mark.parametrize(
        ("dem", "expected_logicals"),
        [
            (
                stim.DetectorErrorModel(
                    "\n".join(["error(0.1) D0 L0", "error(0.1) D1 L1"])
                ),
                [{DecodingHyperEdge({0})}, {DecodingHyperEdge({1})}],
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 D1 L0 L1"),
                [{DecodingHyperEdge({0, 1})}, {DecodingHyperEdge({0, 1})}],
            ),
        ],
    )
    def test_dem_with_multiple_logicals_give_expected_logicals(
        self,
        dem: stim.DetectorErrorModel,
        expected_logicals: list[set[DecodingHyperEdge]],
    ):
        _, logicals = dem_to_hypergraph_and_logicals(dem)
        assert logicals == expected_logicals


class TestDemToNXGraph:
    """Test conversion from stim detector error model to NXDecodingGraph via
    the `dem_to_decoding_graph_and_logicals` function.
    """

    @pytest.fixture(
        params=[
            dem_nodes_edges_logicals_repetition_code_4_round_decomposed(),
            "dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed",
            dem_nodes_edges_logicals_RP_3x3_Z_2_round_decomposed(),
        ]
    )
    def dem_nodes_edges_and_logicals(self, request: pytest.FixtureRequest):
        param = request.param
        if isinstance(param, str):
            param = request.getfixturevalue(param)
        return param

    def test_nx_function_returns_expected_logicals(self, dem_nodes_edges_and_logicals):
        dem, _, _, expected_logicals = dem_nodes_edges_and_logicals
        _, logicals = dem_to_decoding_graph_and_logicals(dem)
        assert logicals == expected_logicals

    def test_nx_graph_has_expected_nodes(self, dem_nodes_edges_and_logicals):
        dem, expected_nodes, _, _ = dem_nodes_edges_and_logicals
        nx_graph, _ = dem_to_decoding_graph_and_logicals(dem)
        assert set(nx_graph.nodes) == set(expected_nodes)

    def test_nx_graph_has_expected_edges(self, dem_nodes_edges_and_logicals):
        dem, _, expected_edges, _ = dem_nodes_edges_and_logicals
        nx_graph, _ = dem_to_decoding_graph_and_logicals(dem)
        assert {
            DecodingEdge(edge.first, edge.second) for edge in nx_graph.edges
        } == set(expected_edges)

    @pytest.mark.parametrize(
        "nongraphlike_edge",
        [
            stim.DetectorErrorModel("error(0.1) D1 D2 ^ D3 D4 D5"),
            stim.DetectorErrorModel("error(0.1) D0 D1 D2 D5"),
            stim.DetectorErrorModel("error(0.1) D0 D1 D2"),
        ],
    )
    def test_error_is_raised_if_edge_has_no_graphlike_degree(self, nongraphlike_edge):
        with pytest.raises(
            ValueError,
            match=r"Edge of degree \d+ cannot be "
            "converted to decoding edge.",
        ):
            dem_to_decoding_graph_and_logicals(nongraphlike_edge)

    @pytest.mark.skip(
        reason="Still considering what the weight of such edges should be"
    )
    def test_edges_with_zero_p_err_are_added_to_the_graph(self):
        dem = stim.DetectorErrorModel("error(0) D0 D1")
        graph, _ = dem_to_decoding_graph_and_logicals(dem)
        assert DecodingEdge({0, 1}, p_err=0) in graph.edges

    @pytest.mark.parametrize(
        ("dem", "expected_logicals"),
        [
            (
                stim.DetectorErrorModel(
                    "\n".join(["error(0.1) D0 L0", "error(0.1) D1 L1"])
                ),
                [{DecodingEdge(0, 2)}, {DecodingEdge(1, 2)}],
            ),
            (
                stim.DetectorErrorModel("error(0.01) D0 D1 L0 L1"),
                [{DecodingEdge(0, 1)}, {DecodingEdge(0, 1)}],
            ),
            # (
            #     stim.DetectorErrorModel("error(0.01) D3 D1 L1 ^ L2"),
            #     [set(), {DecodingEdge(3, 1)}, set()]
            # )
        ],
    )
    def test_dem_with_multiple_logicals_give_expected_logicals(
        self, dem: stim.DetectorErrorModel, expected_logicals: list[set[DecodingEdge]]
    ):
        _, logicals = dem_to_decoding_graph_and_logicals(dem)
        assert logicals == expected_logicals

    @pytest.mark.parametrize(
        "dem",
        [
            stim.DetectorErrorModel("error(0.01) L2"),
            stim.DetectorErrorModel("error(0.01) D0 D1 ^ D2 D3 L0 ^ L2"),
            stim.DetectorErrorModel("error(0.01) D0 D3 ^ L3 ^ D0 L2"),
        ],
    )
    def test_warning_raised_for_degree_0_edge(self, dem):
        msg = "Degree 0 edge has been skipped over in graph creation."
        with pytest.warns(UserWarning, match=msg):
            dem_to_decoding_graph_and_logicals(dem)


class TestExampleRPlanar3x3x1DemToDecodingGraph:
    """Integration test to verify that the RPlanar 3x3x1
    stim detector error model is correctly converted to a QECF NXDecodingGraph object.
    """

    @pytest.fixture(scope="class")
    def example_decoding_graph(
        self, dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed
    ):
        dem, _, _, _ = dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed
        nx_graph, _ = dem_to_decoding_graph_and_logicals(dem)
        return nx_graph

    @pytest.fixture(scope="class")
    def example_logical(self, dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed):
        dem, _, _, _ = dem_nodes_edges_logicals_RP_3x3_X_1_round_decomposed
        _, logical = dem_to_decoding_graph_and_logicals(dem)
        return logical

    def test_there_is_only_one_boundary_node(self, example_decoding_graph):
        assert (
            len(
                [
                    node
                    for node in example_decoding_graph.nodes
                    if example_decoding_graph.detector_is_boundary(node)
                ]
            )
            == 1
        )

    def test_boundary_node_fully_connected(self, example_decoding_graph):
        boundary = next(
            node
            for node in example_decoding_graph.nodes
            if example_decoding_graph.detector_is_boundary(node)
        )
        assert (
            len(list(example_decoding_graph.neighbors(boundary)))
            == len(example_decoding_graph.nodes) - 1
        )

    def test_all_spatial_coordinates_are_not_none(self, example_decoding_graph):
        assert all(
            node.spatial_coord is not None
            for node in example_decoding_graph.detector_records.values()
        )

    def test_number_of_logicals_is_1(self, example_logical):
        assert len(example_logical) == 1

    def test_logicals_correctly_stored(self, example_logical):
        expected_logicals = {(0, 8), (1, 8), (4, 8), (5, 8)}
        obtained_logicals = set()
        for u, v in example_logical[0]:
            obtained_logicals.add(tuple(sorted((u, v))))
        assert expected_logicals == obtained_logicals
