# (c) Copyright Riverlane 2020-2025.
import pytest
import stim
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    NXCode,
    OrderedDecodingEdges,
    OrderedSyndrome,
)

from deltakit_decode.utils import VisDecodingGraph3D, parse_stim_circuit


@pytest.fixture(scope="module")
def code():
    circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z",
                                     distance=5, rounds=5,
                                     before_round_data_depolarization=0.001,
                                     after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001)
    graph, logicals, _ = parse_stim_circuit(circuit, trim_circuit=True)
    return NXCode(graph, logicals)


@pytest.mark.parametrize("categorise_edges", [True, False])
def test_decoding_graph_visualiser_does_not_error(code, categorise_edges):
    syndrome = OrderedSyndrome([0, 3])
    error_edges = OrderedDecodingEdges([DecodingEdge(0, 3)])
    VisDecodingGraph3D(code.graph).plot_3d(
        syndrome=syndrome, correction_edges=error_edges, error_edges=error_edges,
        logicals=code.logicals, categorise_edges=categorise_edges, show=False)
