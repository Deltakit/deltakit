# (c) Copyright Riverlane 2020-2025.
from itertools import tee

import deltakit_circuit as sp
import networkx as nx
import pytest
import stim
from deltakit_decode.utils._graph_circuit_helpers import parse_stim_circuit


def stim_circuit_rep_5x4():
    return stim.Circuit.generated("repetition_code:memory",
                                  rounds=4,
                                  distance=5,
                                  before_round_data_depolarization=0.1,
                                  before_measure_flip_probability=0.1)


def stim_circuit_rplanar_3x3x3():
    return stim.Circuit.generated('surface_code:rotated_memory_x',
                                  rounds=3,
                                  distance=3,
                                  before_round_data_depolarization=0.1,
                                  before_measure_flip_probability=0.1)


def stim_circuit_planar_5x5x2():
    return stim.Circuit.generated('surface_code:unrotated_memory_z',
                                  rounds=2,
                                  distance=5,
                                  before_round_data_depolarization=0.1,
                                  before_measure_flip_probability=0.1)


class TestParseStimCircuit:

    @pytest.fixture(params=[
        stim_circuit_rep_5x4(),
        stim_circuit_rplanar_3x3x3(),
        stim_circuit_planar_5x5x2()
    ], scope="class")
    def stim_circuit(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def original_graph_trimmed_graph_logicals(self, stim_circuit):
        trimmed_graph, logicals, _ = parse_stim_circuit(stim_circuit, trim_circuit=True)
        original_graph, _, _ = parse_stim_circuit(stim_circuit, trim_circuit=False)
        return original_graph, trimmed_graph, logicals

    def test_trimmed_stim_circuit_has_same_number_of_detectors_as_its_corresponding_trimmed_graph(self, stim_circuit):
        trimmed_graph, _, trimmed_stim_circuit = parse_stim_circuit(stim_circuit)
        assert trimmed_stim_circuit.num_detectors == len(
            trimmed_graph.nodes) - len(trimmed_graph.boundaries)

    def test_trimmed_stim_circuit_has_same_number_of_observables_as_its_corresponding_trimmed_graph(self, stim_circuit):
        _, trimmed_logicals, trimmed_stim_circuit = parse_stim_circuit(stim_circuit)
        assert trimmed_stim_circuit.num_observables == len(trimmed_logicals)

    def test_logicals_are_reachable_in_trimmed_stim_graph(self,
                                                          original_graph_trimmed_graph_logicals):
        _, trimmed_graph, logicals = original_graph_trimmed_graph_logicals
        non_boundary_nodes = (
            node for node in trimmed_graph.nodes if not trimmed_graph.detector_is_boundary(node))
        for node in non_boundary_nodes:
            assert any(any(nx.has_path(trimmed_graph.no_boundary_view, node, logical_a)
                           for logical_a, _ in logical if not trimmed_graph.detector_is_boundary(logical_a))
                       or
                       any(nx.has_path(trimmed_graph.no_boundary_view, node, logical_b)
                           for _, logical_b in logical if not trimmed_graph.detector_is_boundary(logical_b))
                       for logical in logicals)

    def test_logicals_edges_are_still_in_trimmed_graph(self,
                                                       original_graph_trimmed_graph_logicals):
        _, trimmed_graph, logicals = original_graph_trimmed_graph_logicals
        assert all(edge in trimmed_graph.edges for logical in logicals for edge in logical)

    def test_trimmed_graph_has_no_more_nodes_than_origin(self,
                                                         original_graph_trimmed_graph_logicals):
        original_graph, trimmed_graph, _ = original_graph_trimmed_graph_logicals
        assert len(trimmed_graph.nodes) <= len(original_graph.nodes)

    def test_trimmed_graph_has_no_more_edges_than_origin(self,
                                                         original_graph_trimmed_graph_logicals):
        original_graph, trimmed_graph, _ = original_graph_trimmed_graph_logicals
        assert len(trimmed_graph.edges) <= len(original_graph.edges)

    def test_detector_order_is_unchanged_without_lexical_detectors(self, stim_circuit):
        _, _, stim_circuit_out = parse_stim_circuit(
            stim_circuit, trim_circuit=False, lexical_detectors=False)
        circuit_in = sp.Circuit.from_stim_circuit(stim_circuit)
        circuit_out = sp.Circuit.from_stim_circuit(stim_circuit_out)
        assert len(circuit_in.detectors()) == len(circuit_out.detectors())
        assert circuit_in.detectors() == circuit_out.detectors()

    def test_detectors_are_more_ordered_when_lexical_flag_set(self, stim_circuit):
        _, _, stim_circuit_out = parse_stim_circuit(
            stim_circuit, trim_circuit=False, lexical_detectors=True)
        circuit_in = sp.Circuit.from_stim_circuit(stim_circuit)
        circuit_out = sp.Circuit.from_stim_circuit(stim_circuit_out)
        assert len(circuit_in.detectors()) == len(circuit_out.detectors())

        firsts_out, seconds_out = tee(circuit_out.detectors())
        firsts_in, seconds_in = tee(circuit_in.detectors())
        next(seconds_out), next(seconds_in)
        output_orderings = sum(a.coordinate <= b.coordinate for a,
                               b in zip(firsts_out, seconds_out))
        input_orderings = sum(a.coordinate <= b.coordinate for a,
                              b in zip(firsts_in, seconds_in))

        assert output_orderings >= input_orderings
