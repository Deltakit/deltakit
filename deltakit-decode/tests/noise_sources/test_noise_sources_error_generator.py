# (c) Copyright Riverlane 2020-2025.

import deltakit_circuit as sp
import pytest
import stim
from deltakit_core.decoding_graphs import (DecodingEdge, DecodingHyperEdge,
                                           DecodingHyperGraph, EdgeRecord,
                                           HyperMultiGraph, NXDecodingGraph,
                                           OrderedDecodingEdges,
                                           OrderedSyndrome,
                                           dem_to_decoding_graph_and_logicals,
                                           dem_to_hypergraph_and_logicals)
from deltakit_decode.utils import parse_stim_circuit
from deltakit_decode.noise_sources import (EdgeProbabilityMatchingNoise,
                                           ExhaustiveMatchingNoise,
                                           ExhaustiveWeightedMatchingNoise,
                                           FixedWeightMatchingNoise,
                                           SampleStimNoise,
                                           UniformMatchingNoise)
from deltakit_decode.noise_sources._generic_noise_sources import _NoiseModel
from pytest_lazy_fixtures import lf


@pytest.fixture(scope="module")
def manual_decoding_graph():
    edge_vertices = [
        (0, 1), (0, 2), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 5), (2, 6), (3, 7), (4, 5),
        (5, 6), (6, 7), (7, 4)
    ]
    return NXDecodingGraph.from_edge_list(
        [(DecodingEdge(u, v), EdgeRecord(0.5)) for u, v in edge_vertices])


@pytest.fixture(scope="module")
def stim_decoding_graph():
    stim_circuit = stim.Circuit.generated("surface_code:unrotated_memory_z",
                                          rounds=2, distance=3,
                                          before_round_data_depolarization=0.01,
                                          before_measure_flip_probability=0.01)
    graph, _, _ = parse_stim_circuit(stim_circuit, lexical_detectors=False)
    return graph


@pytest.fixture(scope="module")
def manual_decoding_hypergraph():
    edges = [
        (DecodingHyperEdge((0, 1, 2)), EdgeRecord(0.5)),
        (DecodingHyperEdge((2, 3, 6)), EdgeRecord(0.5)),
        (DecodingHyperEdge((1, 4, 5)), EdgeRecord(0.5)),
        (DecodingHyperEdge((6, 5, 4)), EdgeRecord(0.5)),
        (DecodingHyperEdge((0, 6, 5)), EdgeRecord(0.5))]
    return DecodingHyperGraph(edges)


@pytest.fixture(scope="module")
def stim_decoding_hypergraph():
    stim_circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z",
                                          distance=5, rounds=2,
                                          before_round_data_depolarization=0.02,
                                          after_clifford_depolarization=0.02,
                                          before_measure_flip_probability=0.02)
    stim_decoding_hypergraph, _ = dem_to_hypergraph_and_logicals(
        stim_circuit.detector_error_model(decompose_errors=False))
    return stim_decoding_hypergraph


class TestExhaustiveMatchingNoise:

    @pytest.mark.parametrize("graph, expected_errors", [
        (
            lf("manual_decoding_graph"),
            {frozenset({0, 1}), frozenset({0, 2})}
        ),
        (
            lf("stim_decoding_graph"),
            {frozenset({0, 18}), frozenset({1, 18})}
        ),
        (
            lf("manual_decoding_hypergraph"),
            {frozenset({2, 3, 6}), frozenset({0, 1, 2})}
        ),
        (
            lf("stim_decoding_hypergraph"),
            {frozenset({0, 2}), frozenset({0, 3})}
        )
    ])
    def test_sampled_errors_match_expected_errors(self, graph, expected_errors):
        noise_model = ExhaustiveMatchingNoise(2)
        noise_sample = next(noise_model.error_generator(graph, 1))
        error = {edge.vertices for edge in noise_sample}
        assert error == expected_errors


class TestExhaustiveWeightedMatchingNoise:

    @pytest.fixture
    def noise_model_and_decoding_graph(self):
        spatial_d, rounds = (3, 1)
        stim_circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z",
                                              distance=spatial_d, rounds=rounds,
                                              before_round_data_depolarization=0.02,
                                              after_clifford_depolarization=0.02,
                                              before_measure_flip_probability=0.02)

        dem = stim_circuit.detector_error_model(decompose_errors=True)
        decoding_graph, logicals = dem_to_decoding_graph_and_logicals(dem)
        relevant_nodes = decoding_graph.get_relevant_nodes(logicals)
        irrelevant_nodes = set(decoding_graph.nodes) - relevant_nodes
        stim_circuit = sp.trim_detectors(stim_circuit, irrelevant_nodes)
        dem = stim_circuit.detector_error_model(decompose_errors=True)
        decoding_graph, logicals = dem_to_decoding_graph_and_logicals(dem)

        times = {
            decoding_graph.detector_records[logical.second].time for logical in logicals[0]}
        exhaustion_ceiling = min(
            [sum(decoding_graph.edge_records[edge].weight for edge in logicals[0]
                 if decoding_graph.detector_records[edge.second].time == time)
             for time in times]) / 2

        return (ExhaustiveWeightedMatchingNoise(exhaustion_ceiling), decoding_graph)

    def test_pruning_noise_model_deduces_correct_num_of_weights_that_can_be_combined_within_the_exhaustion_ceiling(self, noise_model_and_decoding_graph):
        noise_model, decoding_graph = noise_model_and_decoding_graph
        _, max_edges_within_ceiling = noise_model.prune_edges(decoding_graph)
        assert max_edges_within_ceiling == 1

    def test_correct_sequence_size_for_noise_model_sequence(self, noise_model_and_decoding_graph):
        noise_model, decoding_graph = noise_model_and_decoding_graph
        assert noise_model.sequence_size(decoding_graph) == 4

    def test_no_errors_above_ceiling_are_sampled(self, manual_decoding_hypergraph):
        noise_model = ExhaustiveWeightedMatchingNoise(0.01)
        noise_sample = set(noise_model.error_generator(manual_decoding_hypergraph))
        assert not noise_sample

    def test_correct_errors_are_generated(self, noise_model_and_decoding_graph):
        noise_model, decoding_graph = noise_model_and_decoding_graph
        expected_edges = frozenset([
            DecodingEdge(8, 5),
            DecodingEdge(8, 6),
            DecodingEdge(8, 1),
            DecodingEdge(8, 2)
        ])
        generated_edges = []
        for ordered_decoding_edges in noise_model.error_generator(decoding_graph):
            generated_edges += list(ordered_decoding_edges)
        assert expected_edges == frozenset(generated_edges)


class TestFixedWeightMatchingNoise:

    @pytest.mark.parametrize("graph, expected_errors", [
        (
            lf("manual_decoding_graph"),
            {frozenset({4, 5}), frozenset({2, 3})}
        ),
        (
            lf("stim_decoding_graph"),
            {frozenset({2, 4}), frozenset({6, 7})}
        ),
        (
            lf("manual_decoding_hypergraph"),
            {frozenset({4, 5, 6}), frozenset({1, 4, 5})}
        ),
        (
            lf("stim_decoding_hypergraph"),
            {frozenset({17, 22}), frozenset({16, 25, 21})}
        )
    ])
    def test_sampled_errors_match_expected_errors(self, graph, expected_errors):
        noise_model = FixedWeightMatchingNoise(2)
        noise_sample = next(noise_model.error_generator(graph, 1))
        error = {edge.vertices for edge in noise_sample}
        assert error == expected_errors


class TestSampleStimNoise:

    @pytest.mark.parametrize("stim_circuit, expected_sample", [
        (
            stim.Circuit("""
                X_ERROR(1) 0 1 2
                M 0 1 2
                DETECTOR rec[-1]
                DETECTOR rec[-2]
                DETECTOR rec[-3]
                OBSERVABLE_INCLUDE(0) rec[-1]
                """),
            (OrderedSyndrome([0, 1, 2]), (True,))
        ),
        (
            stim.Circuit("""
                X_ERROR(1) 3 4
                M 3 4
                DETECTOR rec[-1]
                DETECTOR rec[-2]
                """),
            (OrderedSyndrome([0, 1]), ())
        ),
        (
            stim.Circuit("""
                X_ERROR(1) 1
                MZ 0 1
                OBSERVABLE_INCLUDE(1) rec[-1]
                OBSERVABLE_INCLUDE(0) rec[-2]
                """),
            (OrderedSyndrome(), (False, True))
        )
    ])
    def test_deterministic_stim_circuit_samples(
            self, stim_circuit, expected_sample):
        noise_model = SampleStimNoise()
        gen = noise_model.error_generator(stim_circuit)
        assert next(gen) == expected_sample

    def test_addition_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            SampleStimNoise() + UniformMatchingNoise(0.01)


class TestUniformMatchingNoise:

    @pytest.mark.parametrize("noise_model, graph, expected_errors", [
        (
            UniformMatchingNoise(0.2),
            lf("manual_decoding_graph"),
            {frozenset({4, 5}), frozenset({0, 3})}
        ),
        (
            UniformMatchingNoise(0.1),
            lf("stim_decoding_graph"),
            {frozenset({10, 11}), frozenset({9, 18}), frozenset({10, 7})}
        ),
        (
            UniformMatchingNoise(0.5),
            lf("manual_decoding_hypergraph"),
            {frozenset({0, 5, 6}), frozenset({1, 4, 5})}
        ),
        (
            UniformMatchingNoise(0.01),
            lf("stim_decoding_hypergraph"),
            {frozenset({25, 30, 31}), frozenset({26, 7}), frozenset({15, 13, 6, 14})}
        )
    ])
    def test_sampled_errors_match_expected_errors(self, noise_model, graph, expected_errors):
        noise_sample = next(noise_model.error_generator(graph, 1))
        error = {edge.vertices for edge in noise_sample}
        assert error == expected_errors


class TestEdgeProbabilityMatchingNoise:
    @pytest.mark.parametrize("decoding_graph, expected_errors", [
        (
            lf("stim_decoding_graph"),
            OrderedDecodingEdges()
        ),
        (
            lf("stim_decoding_hypergraph"),
            OrderedDecodingEdges([DecodingHyperEdge({6}),
                                  DecodingHyperEdge({26, 7}),
                                  DecodingHyperEdge({16}),
                                  DecodingHyperEdge({35})])
        ),
        (
            lf("manual_decoding_graph"),
            OrderedDecodingEdges([DecodingEdge(0, 3),
                                  DecodingEdge(1, 2),
                                  DecodingEdge(1, 5),
                                  DecodingEdge(2, 6),
                                  DecodingEdge(4, 5),
                                  DecodingEdge(6, 7)])
        ),
        (
            lf("manual_decoding_hypergraph"),
            OrderedDecodingEdges([DecodingHyperEdge({1, 4, 5}),
                                  DecodingHyperEdge({0, 5, 6})])
        )

    ])
    def test_sampled_errors_match_expected_errors(
        self,
        decoding_graph: HyperMultiGraph,
        expected_errors: OrderedDecodingEdges
    ):
        noise_model = EdgeProbabilityMatchingNoise()
        noise_sample = next(noise_model.error_generator(decoding_graph, 1))
        assert noise_sample == expected_errors

    @pytest.mark.parametrize("rhs", [
        EdgeProbabilityMatchingNoise(),
        UniformMatchingNoise(0.1),
        ExhaustiveMatchingNoise(5),
        ExhaustiveWeightedMatchingNoise(0.1)
    ])
    def test_error_is_raised_when_trying_to_add_to_another_noise(self, rhs: _NoiseModel):
        lhs = EdgeProbabilityMatchingNoise()
        with pytest.raises(NotImplementedError,
                           match=r"Cannot add edge probability noise to any other noise."):
            lhs + rhs
