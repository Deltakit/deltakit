# (c) Copyright Riverlane 2020-2025.
"""
Sub-package for defining decoding graphs and related data types; these structures are
used by many decoders to define their understanding of the code and noise model being
decoded.
"""

from deltakit_core.decoding_graphs._data_qubits import (
    DecodingEdge,
    DecodingHyperEdge,
    EdgeRecord,
    EdgeT,
    OrderedDecodingEdges,
    errors_to_syndrome,
)
from deltakit_core.decoding_graphs._decoding_graph import (
    AnyEdgeT,
    DecodingCode,
    DecodingHyperGraph,
    DecodingHyperMultiGraph,
    HyperLogicals,
    HyperMultiGraph,
    NXCode,
    NXDecodingGraph,
    NXDecodingMultiGraph,
    NXLogicals,
)
from deltakit_core.decoding_graphs._decoding_graph_tools import (
    compute_graph_distance,
    compute_graph_distance_for_logical,
    filter_to_data_edges,
    filter_to_measure_edges,
    graph_to_json,
    has_contiguous_nodes,
    hypergraph_to_weighted_edge_list,
    inverse_logical_at_boundary,
    is_single_connected_component,
    single_boundary_is_last_node,
    unweight_graph,
    worst_case_num_detectors,
)
from deltakit_core.decoding_graphs._dem_parsing import (
    DemParser,
    DetectorCounter,
    DetectorRecorder,
    LogicalsInEdges,
    dem_to_decoding_graph_and_logicals,
    dem_to_hypergraph_and_logicals,
    observable_warning,
)
from deltakit_core.decoding_graphs._explained_dem_parsing import (
    extract_logicals,
    parse_explained_dem,
)
from deltakit_core.decoding_graphs._hypergraph_annotations_tools import (
    annotate_edges_with_window_ids,
    get_unique_logical_window_ids,
    get_unique_window_ids,
    get_window_id_transfer_graph,
    separate_edges_per_window_id,
    separate_logicals_by_window_id,
)
from deltakit_core.decoding_graphs._hypergraph_decomposition import decompositions
from deltakit_core.decoding_graphs._syndromes import (
    Bitstring,
    DetectorRecord,
    FixedWidthBitstring,
    OrderedSyndrome,
    get_round_words,
)
from deltakit_core.decoding_graphs._weighted_graphs import (
    change_graph_error_probabilities,
    vector_weights,
)
from deltakit_core.decoding_graphs._windowing_utils import (
    connect_dangling_to_boundary_hypergraph,
    expand_nodes_to_time_span,
    induce_subhypergraph,
    nodes_within_radius,
    relabel_hypergraph_nodes_contiguously,
    weighted_nodes_within_radius,
)

# List only public members in `__all__`.
__all__ = [
    "AnyEdgeT",
    "Bitstring",
    "DecodingCode",
    "DecodingEdge",
    "DecodingHyperEdge",
    "DecodingHyperGraph",
    "DecodingHyperMultiGraph",
    "DemParser",
    "DetectorCounter",
    "DetectorRecord",
    "DetectorRecorder",
    "EdgeRecord",
    "EdgeT",
    "FixedWidthBitstring",
    "HyperLogicals",
    "HyperMultiGraph",
    "LogicalsInEdges",
    "NXCode",
    "NXDecodingGraph",
    "NXDecodingMultiGraph",
    "NXLogicals",
    "OrderedDecodingEdges",
    "OrderedSyndrome",
    "annotate_edges_with_window_ids",
    "change_graph_error_probabilities",
    "compute_graph_distance",
    "compute_graph_distance_for_logical",
    "connect_dangling_to_boundary_hypergraph",
    "decompositions",
    "dem_to_decoding_graph_and_logicals",
    "dem_to_hypergraph_and_logicals",
    "errors_to_syndrome",
    "expand_nodes_to_time_span",
    "extract_logicals",
    "filter_to_data_edges",
    "filter_to_measure_edges",
    "get_round_words",
    "get_unique_logical_window_ids",
    "get_unique_window_ids",
    "get_window_id_transfer_graph",
    "graph_to_json",
    "has_contiguous_nodes",
    "hypergraph_to_weighted_edge_list",
    "induce_subhypergraph",
    "inverse_logical_at_boundary",
    "is_single_connected_component",
    "nodes_within_radius",
    "observable_warning",
    "parse_explained_dem",
    "relabel_hypergraph_nodes_contiguously",
    "separate_edges_per_window_id",
    "separate_logicals_by_window_id",
    "single_boundary_is_last_node",
    "unweight_graph",
    "vector_weights",
    "weighted_nodes_within_radius",
    "worst_case_num_detectors",
]
