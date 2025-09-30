# (c) Copyright Riverlane 2020-2025.
"""Module containing useful functions to create and interact with decoding graphs."""

import decimal
import json
import math
from typing import (
    AbstractSet,
    FrozenSet,
    Iterable,
    List,
    Set,
    Tuple,
    Union,
    no_type_check,
)

import networkx as nx
import numpy as np
from deltakit_core.decoding_graphs import (
    AnyEdgeT,
    DecodingEdge,
    DecodingHyperGraph,
    DecodingHyperMultiGraph,
    DetectorRecord,
    EdgeRecord,
    HyperLogicals,
    HyperMultiGraph,
    NXDecodingGraph,
    NXDecodingMultiGraph,
    NXLogicals,
)


def filter_to_data_edges(graph: NXDecodingGraph) -> List[DecodingEdge]:
    """Given some decoding edges, filter to those that directly correspond to a data
    qubit.
    """
    return [edge for edge in graph.edges if edge.is_spacelike(graph.detector_records)]


def filter_to_measure_edges(graph: NXDecodingGraph) -> List[DecodingEdge]:
    """Given some decoding edges, filter to those that directly correspond to a
    measurement event, i.e. they only move in the time axis.
    """
    return [edge for edge in graph.edges if edge.is_timelike(graph.detector_records)]


def hypergraph_to_weighted_edge_list(
    hypergraph: DecodingHyperGraph,
) -> List[Tuple[FrozenSet[int], float]]:
    """Return a weighted edge list representation of a decoding hypergraph.

    A format that is useful for integration with other libraries for example in
    Belief Propagation decoders.

    Returns
    -------
    List[Tuple[frozenset[int], float]]]
        List of edges as a set of nodes and error probabilities.
        Ex. `[({0, 3, 5}, 0.05)]` represents a hyperedge with nodes
        {0, 3, 5} and error probability 0.05.
    """
    return [
        (edge.vertices, hypergraph.edge_records[edge].p_err)
        for edge in hypergraph.edges
    ]


def has_contiguous_nodes(graph: HyperMultiGraph, start: int = 0) -> bool:
    """Returns whether the nodes of the graph have contiguous indices with
    respect to the start index.

    Parameters
    ----------
    graph : HyperMultiGraph
        Any graph.
    start : int, optional
        Which index to start counting nodes from at, by default 0.

    Returns
    -------
    bool
        Whether all nodes in the graph are zero indexed and contiguous.
    """
    return all(index == node for index, node in enumerate(graph.nodes, start))


def single_boundary_is_last_node(graph: HyperMultiGraph) -> bool:
    """
    Parameters
    ----------
    graph : HyperMultiGraph
        Any graph

    Returns
    -------
    bool
        True if graph has a single boundary and that boundary is the last node where
        nodes are ordered wrt their index.
    """
    return (
        len(graph.boundaries) == 1 and next(iter(graph.boundaries)) == graph.nodes[-1]
    )


def is_single_connected_component(graph: NXDecodingGraph) -> bool:
    """
    Determines whether a graph consists of a single connected component
    excluding the boundary nodes of the graph.

    Parameters
    ----------
    graph : NXDecodingGraph
        A NetworkX decoding graph

    Returns
    -------
    bool
        True if the graph contains a single connected component False otherwise.
    """
    return nx.number_connected_components(graph.no_boundary_view) == 1


@no_type_check
def graph_to_json(
    decoding_graph: HyperMultiGraph[AnyEdgeT],
    logicals: Iterable[Set[AnyEdgeT]],
    full: bool = False,
) -> str:
    """Represent some graph as an edge list in JSON. Logicals are also included.
    Fields are:

    - edges: List of sorted tuples for all edges
    - edge_weights: list of floats for edge weights
    - logicals: List of sorted tuples for logical edges
    - boundary: Decoding graph boundary. None if it doesn't exist.

    If full flag is True, we additionally serialize:

    - detector_records: dict containing detector metadata
    - edge_record: dict containing edge metadata

    Parameters
    ----------
    decoding_graph :  HyperMultiGraph[AnyEdgeT]
        Graph to represent in JSON. Should have no more than one boundary.
    logicals : Iterable[Set[AnyEdgeT]]
        Logicals to represent in JSON.
    full : bool
        Whether to save full graph information that includes
        detector and edge records.

    Returns
    -------
    str
        JSON formatted string representing the essential graph and logical info
    """
    if len(decoding_graph.boundaries) > 1:
        raise ValueError("JSON graph representation supports maximum one boundary")

    if isinstance(decoding_graph, (DecodingHyperMultiGraph, NXDecodingMultiGraph)):
        edges = [(sorted(edge), edge_id) for edge, edge_id in decoding_graph.edges]
        logicals = [
            [(sorted(edge), edge_id) for edge, edge_id in logical]
            for logical in logicals
        ]
    else:
        edges = [sorted(edge) for edge in decoding_graph.edges]
        logicals = [[sorted(edge) for edge in logical] for logical in logicals]

    edge_weights = [
        decoding_graph.edge_records[edge].weight for edge in decoding_graph.edges
    ]
    boundary = (
        next(iter(decoding_graph.boundaries)) if decoding_graph.boundaries else None
    )

    graph_as_dict = {
        "edges": edges,
        "edge_weights": edge_weights,
        "logicals": logicals,
        "boundary": boundary,
    }
    if full:
        detector_records = {
            key: dict(value) for key, value in decoding_graph.detector_records.items()
        }
        edge_records = []
        for edge in edges:
            record = dict(decoding_graph.edge_records[DecodingEdge(*edge)])
            record.pop("decoding_edge", None)
            edge_records.append(record)
        graph_as_dict["detector_records"] = detector_records
        graph_as_dict["edge_records"] = edge_records

    return json.dumps(graph_as_dict)


def nx_graph_from_json(
    json_str: str,
) -> Tuple[NXDecodingGraph, List[FrozenSet[DecodingEdge]]]:
    """Loads the graph from json string given by graph_to_json.

    Parameters
    ----------
    json_str : str
        JSON string.

    Returns
    -------
    Tuple[NXDecodingGraph, List[FrozenSet[DecodingEdge]]]
        Reconstructed decoding graph and logicals.
    """
    graph_as_dict = json.loads(json_str)
    if (boundary_node := graph_as_dict["boundary"]) is not None:
        boundary = frozenset([boundary_node])
    else:
        boundary = frozenset()
    # collect detector records if defined
    detector_records = None
    if (detector_records_dict := graph_as_dict.get("detector_records")) is not None:
        detector_records = {
            int(key): DetectorRecord.from_dict(value)
            for key, value in detector_records_dict.items()
        }
    # collect edge data if defined
    edge_data: Iterable[Union[DecodingEdge, Tuple[DecodingEdge, EdgeRecord]]]
    if (edge_records_dict := graph_as_dict.get("edge_records")) is not None:
        edge_records = [EdgeRecord.from_dict(record) for record in edge_records_dict]
        edge_data = [
            (DecodingEdge(*edge), record)
            for edge, record in zip(graph_as_dict["edges"], edge_records, strict=False)
        ]
    else:
        edge_data = [DecodingEdge(*edge) for edge in graph_as_dict["edges"]]
    graph = NXDecodingGraph.from_edge_list(edge_data, detector_records, boundary)
    logicals = [
        frozenset(DecodingEdge(*edge) for edge in logical)
        for logical in graph_as_dict["logicals"]
    ]
    return graph, logicals


def inverse_logical_at_boundary(
    decoding_graph: NXDecodingGraph, logical: Set[DecodingEdge]
) -> Set[DecodingEdge]:
    """Given a decoding graph and a logical, that is assumed to be along the
    boundary, return another logical constructed of all the edges incident to the
    boundary that are not in the given logical.

    On a surface code for example, if we gave the logical as the edges along one
    side of the surface, this function would return the logical made of the edges
    along the opposite side of the surface.

    Parameters
    ----------
    decoding_graph : NXDecodingGraph
        Decoding graph the logicals are defined over.
    logical : Set[DecodingEdge]
        The logical to get the boundary inverse of.
        If this given logical is not consisting of only edges along the boundary, then
        an exception will be raised.

    Returns
    -------
    Set[DecodingEdge]
        A logical along the boundary, the inverse of the given logical.

    Raises
    ------
    ValueError
        Exception raised if the the logical is not entirely along the boundary of the
        graph. This includes if the logical has an edge not in the graph.
    """
    all_edges_to_boundaries: Set[DecodingEdge] = set()
    for boundary in decoding_graph.boundaries:
        all_edges_to_boundaries.update(decoding_graph.incident_edges(boundary))

    inverse_logical_edges = {
        edge for edge in all_edges_to_boundaries if edge not in logical
    }
    if len(inverse_logical_edges | logical) != len(all_edges_to_boundaries):
        raise ValueError(
            "The logical given was not entirely along the boundary, "
            "so the inverse logical is undefined."
        )
    return inverse_logical_edges


def is_logical_along_boundary(
    decoding_graph: HyperMultiGraph, logicals: HyperLogicals
) -> bool:
    """Given a decoding graph and logicals, check if the logicals are along the boundary

    Parameters
    ----------
    decoding_graph : HyperMultiGraph
        The decoding graph that the logicals are defined over.
    logicals : HyperLogicals
        The logicals to check.

    Returns
    -------
    bool
        True if the logicals are along the boundary.
    """
    return all(
        (
            any(
                incident_syndrome_bit in decoding_graph.boundaries
                for incident_syndrome_bit in tuple(edge.vertices)
            )
            for logical in logicals
            for edge in logical
        )
    )


def worst_case_num_detectors(
    decoding_graph: DecodingHyperGraph, target_logical_error: float
) -> int:
    """For a given `decoding_graph` representing error mechanisms, and a
    `target_logical_error`, return the largest number of detection events we would
    expect to observe on the `decoding_graph`.

    Parameters
    ----------
    decoding_graph : DecodingHyperGraph
        Decoding hyper graph, where each edge is an error mechanism with some
        probability of occurrence `p_err`.
    target_logical_error : float

    Returns
    -------
    int
        Largest number of detection events we expect to observe.
    """

    def poisson_pmf(expected, k: int):
        expected = decimal.Decimal(expected)  # For increased precision
        return (expected**k) * decimal.Decimal(-expected).exp() / math.factorial(k)

    expected_num_detectors = sum(
        edge_record.p_err * len(edge)
        for edge, edge_record in decoding_graph.edge_records.items()
    )

    if expected_num_detectors == 0:
        return 0

    cumulative_p = 0
    for num_detectors in range(
        len(decoding_graph.nodes) - len(decoding_graph.boundaries) + 1
    ):
        cumulative_p += poisson_pmf(expected_num_detectors, num_detectors)

        if (1 - cumulative_p) <= target_logical_error:
            break

    return num_detectors


def compute_graph_distance_for_logical(
    decoding_graph: NXDecodingGraph,
    logical: AbstractSet[DecodingEdge],
    weighted: bool = False,
) -> Union[int, float]:
    """Computes the distance of the decoding graph given a logical.
    This is done by calculating the shortest path
    between nodes of the logical edges in the graph where all edges of
    that logical have been removed.
    If weighted, returns the length of the shortest non-trivial loop.

    Note: this method is not fully general and will not work well for all
    definitions of the logical! In particular, it fails if the smallest
    weight logical operator has a support on the logical greater than 1.
    I.e. if non-trivial logical loop of edges crosses the logical more
    than once. However, this should be satisfied for most surface codes
    with usual definition of logicals.
    It is likely possible to do this more generally on a matching graph
    by trying to find a shortest path from every point to itself (using
    a modified Dijkstra) that contains the odd number of logical crossings.

    Parameters
    ----------
    decoding_graph : NXDecodingGraph
        Decoding graph to calculate the distance on.
    logical : AbstractSet[DecodingEdge]
        Reference logical.
    weighted : bool
        Whether to calculate the distance in terms of number of edges
        or the weight.
    Returns
    -------
    Union[int, float]
        Distance of the logical. If weighted, returns the number of edges
        (int), otherwise returns the length of the shortest non-trivial loop (float)
    """
    decoding_graph_no_logicals = nx.subgraph_view(
        decoding_graph.graph, filter_edge=lambda u, v: DecodingEdge(u, v) not in logical
    )
    weight = "weight" if weighted else None
    # get the minimum weight of a logical loop by taking the minimal
    # path around a logical edge and adding the weight of that edge
    if weighted:
        edge_weights = np.array(
            [decoding_graph.edge_records[edge].weight for edge in logical]
        )
    else:
        edge_weights = np.ones(len(logical), dtype=int)
    path_lengths = np.array(
        [
            nx.dijkstra_path_length(
                decoding_graph_no_logicals, edge.first, edge.second, weight=weight
            )
            for edge in logical
        ]
    )
    return (path_lengths + edge_weights).min().item()


def compute_graph_distance(
    decoding_graph: NXDecodingGraph, logicals: NXLogicals, weighted: bool = False
) -> Union[int, float]:
    """Calculates the minimum distance for all logicals.
    See `compute_graph_distance_for_logical`.
    Decoding graph to calculate the distance on.

    logicals : NXLogicals
        Reference logicals.
    weighted : bool
        Whether to calculate the distance in terms of number of edges
        or the weight.

    Note: this method is not fully general and will not work well for all
    definitions of the logical! In particular, it fails if the smallest
    weight logical operator has a support on the logical greater than 1.
    I.e. if non-trivial logical loop of edges crosses the logical more
    than once. However, this should be satisfied for most surface codes
    with usual definition of logicals.
    It is likely possible to do this more generally on a matching graph
    by trying to find a shortest path from every point to itself (using
    a modified Dijkstra) that contains the odd number of logical crossings.

    Returns
    -------
    Union[int, float]
        Decoding graph distance. Returns float if weighted is True.
    """
    return min(
        compute_graph_distance_for_logical(decoding_graph, logical, weighted=weighted)
        for logical in logicals
    )


def unweight_graph(decoding_graph: NXDecodingGraph):
    """Sets all weights in the given graph to 1 and error probabilities to 1/(1+e)
    so that they correspond to the weights.

    Parameters
    ----------
    decoding_graph : NXDecodingGraph
        Original decoding to unweight
    """
    for edge in decoding_graph.edges:
        decoding_graph.edge_records[edge].p_err = 1.0 / (1.0 + math.e)
        # set the weight as well just to be sure there are no floating point errors
        decoding_graph.edge_records[edge]["weight"] = 1.0
