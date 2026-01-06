# (c) Copyright Riverlane 2020-2025.

import math
from collections.abc import Callable
from itertools import combinations

import stim
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DecodingHyperEdge,
    DecodingHyperGraph,
    EdgeRecord,
    HyperMultiGraph,
    NXDecodingGraph,
    dem_to_decoding_graph_and_logicals,
    dem_to_hypergraph_and_logicals,
)

from deltakit_decode.utils._derivation_tools import (
    _calculate_edge_prob_with_higher_degrees,
    _d1_formula,
    _n1_formula,
    _n2_formula,
)

PijData = dict[frozenset[int], float]


def _calculate_pij_degree_2(
    key: frozenset[int],
    exp_values: PijData,
) -> float:
    """Calculate Pij values for degree 2 edges as per the formulae defined
    in P19 from https://arxiv.org/pdf/2102.06132.pdf

    Parameters
    ----------
    key : FrozenSet[int]
        The edge we're calculating the Pij for.
    exp_values : PijData
        The expectation values derived from experimental data, to be
        used to calculate the Pij value

    Returns
    -------
    float
        Pij value for the given key/edge.
    """
    xi_key, xj_key = (frozenset((x,)) for x in key)

    numerator = float(_n1_formula(exp_values, [xi_key, xj_key]))

    denominator = _d1_formula(exp_values, [key])

    return 0.5 - 0.5 * math.sqrt(numerator / denominator)


def _calculate_pij_degree_3(
    key: frozenset[int],
    exp_values: PijData,
) -> float:
    """Calculate Pij values for degree 3 hyperedges.

    Parameters
    ----------
    key : FrozenSet[int]
        The edge we're calculating the Pij for.
    exp_values : PijData
        The expectation values derived from experimental data, to be
        used to calculate the Pij value

    Returns
    -------
    float
        Pij value for the given key/edge.
    """
    key_combinations = [frozenset(x) for x in combinations(key, 2)]
    i, j, k = (frozenset((x,)) for x in key)

    n1 = _n1_formula(exp_values, [i, j, k])

    n2 = _n2_formula(exp_values, key, key_combinations)

    d1 = _d1_formula(exp_values, key_combinations)

    inner_b = math.pow(((n1 * n2) / d1), 0.25)

    return 0.5 * (1 - inner_b)


def _calculate_pij_degree_4(
    key: frozenset[int],
    exp_values: PijData,
) -> float:
    """Calculate Pij values for degree 4 hyperedges.

    Parameters
    ----------
    key : FrozenSet[int]
        The edge we're calculating the Pij for.
    exp_values : PijData
        The expectation values derived from experimental data, to be
        used to calculate the Pij value

    Returns
    -------
    float
        Pij value for the given key/edge.
    """
    w2_nodes = [frozenset(x) for x in combinations(key, 2)]
    w3_nodes = [frozenset(x) for x in combinations(key, 3)]
    i, j, k, m = (frozenset((x,)) for x in key)

    n1 = _n1_formula(exp_values, [i, j, k, m])

    n2 = math.prod(_n2_formula(exp_values, n3, [frozenset(x)
                                                 for x in combinations(n3, 2)])
                    for n3 in w3_nodes)

    d1 = _d1_formula(exp_values, w2_nodes)

    d2 = 1\
        - (2 * math.fsum(exp_values.get(n, 0.0) for n in [i, j, k, m])) \
        + (4 * math.fsum(exp_values.get(n2, 0.0) for n2 in w2_nodes)) \
        - (8 * math.fsum(exp_values.get(n3, 0.0) for n3 in w3_nodes)) \
        + (16 * exp_values.get(key, 0.0))

    inner_b = math.pow((n1 * n2) / (d1 * d2), 1 / 8)
    return 0.5 * (1 - inner_b)


def _get_calculate_pij_degree_callable(
    max_degree: int
) -> Callable[[frozenset[int], PijData], float]:
    match max_degree:
        case 2:
            return _calculate_pij_degree_2
        case 3:
            return _calculate_pij_degree_3
        case 4:
            return _calculate_pij_degree_4
        case _:
            msg = f"{max_degree} is not a valid degree, must be between 2-4 inclusive."
            raise NotImplementedError(msg)

def calculate_pij_values(exp_values: PijData,
                         graph: NXDecodingGraph | DecodingHyperGraph | None = None,
                         min_prob: float = -math.inf,
                         max_degree: int = 2,
                         noise_floor_graph: NXDecodingGraph | DecodingHyperGraph | None = None
                         ) -> PijData:
    """Calculates the Pij values given the <Xi> <Xj> etc values.
    At most degree 4 edges are supported.

    NB: If an edge does not turn up in the expectation data, no
        pij value will be calculated for it. E.g, if the edge (3,5)
        is possible in your circuit, but never appears in the samples,
        then there will be no entry in the Pij matrix for this edge.

    Parameters
    ----------
    exp_values : PijData
        Dictionary with keys being Tuples of ints,
        describing edges in the error graph.
        Values are floats describing the expectation
        value for the detector lighting up obtained
        from experimental data.
    graph : Optional[Union[NXDecodingGraph, DecodingHyperGraph]], optional
        Optionally pass a (hyper)graph that will then be used to
        inform which edges of the graph to remove.
        Comparison made against stim's decoder graph, as in
        the Google paper.
    min_prob : float
        Minimum probability with which to limit the calculated
        Pij values. By default -math.inf to allow all values.
        In Google's paper, they set this value to their T1/T2 times.
        Overridden by noise_floor_graph, or only used for edge comparisons
        with edges that are not contained in both the Pij and noise_floor_graph.
    max_degree : int
        Maximum degree of possible (hyper)edges. Any value over 4 will
        be ignored as we do not support computing hyperedges with
        degree > 4.
        Default value is 2 to consider simple graphs rather than hypergraphs.
    noise_floor_graph : Optional[Union[NXDecodingGraph, DecodingHyperGraph]], optional
        (Hyper)graph against which we can compare edges and floor those edges
        if they fall below the value of this noise floor graph. If no graph is given,
        the values will instead be compared against min_prob.
        Noise floor graph likely generated from a noise model.
        Default value is None.

    Returns
    -------
    PijData
        Dictionary with keys being Tuples of ints,
        describing edges in the error graph.
        Values are floats describing the Pij value
        for that particular edge.
    """
    pij_func = _get_calculate_pij_degree_callable(max_degree)
    noise_floor_edges: PijData = {}

    if noise_floor_graph:
        for edge_rec, props in noise_floor_graph.edge_records.items():
            if len(edge_rec) <= max_degree:
                noise_floor_edges[
                    edge_rec.vertices - noise_floor_graph.boundaries] = props.p_err

    edges_to_calc = list(exp_values)
    if len(edges_to_calc) == 0:
        return {}

    if graph:
        edges_to_calc = [e.vertices
                         for e in graph.edges if len(e) <= max_degree]
        if isinstance(graph, NXDecodingGraph):
            boundary = graph.boundaries
            edges_to_calc = [e - boundary for e in edges_to_calc]

    pij_data: PijData = {}
    edges_for_recalc: list[frozenset[int]] = []
    for key in edges_to_calc:
        if len(key) < max_degree:
            # any edges less than max degree need to be re-calculated
            # taking into account any higher degree edges that affect them
            edges_for_recalc.append(key)
            if len(key) == 1:
                pij_data[key] = exp_values[key]
            elif len(key) == 2:
                pij_data[key] = _calculate_pij_degree_2(
                    key, exp_values)
            elif len(key) == 3:
                pij_data[key] = _calculate_pij_degree_3(
                    key, exp_values
                )
            continue

        pij_val = pij_func(key, exp_values)

        pij_data[key] = max(noise_floor_edges.get(key, min_prob), pij_val)

    # calculate probs by taking into account higher degree edges
    # we work top down, adjusting highest degree first
    for edge in sorted(edges_for_recalc, reverse=True, key=len):
        # turn into set for subset calculations.
        # doing this is cheaper than the equivalent tuple checks
        # by a factor of around 8
        adjusted_pi = _calculate_edge_prob_with_higher_degrees(
            edge, pij_data, min_prob)
        pij_data[edge] = max(noise_floor_edges.get(edge, min_prob), adjusted_pi)
    return pij_data


def create_dem_from_pij(
    pij_data: PijData,
    graph: NXDecodingGraph | DecodingHyperGraph,
    logicals: list[set[DecodingHyperEdge]]
) -> stim.DetectorErrorModel:
    """Create a detector error model from a Pij probabilities data set
    and accompanying stim Circuit.

    Parameters
    ----------
    pij_data : PijData
        Pij probabilities for edges between detectors.
    graph : Union[NXDecodingGraph, DecodingHyperGraph]
        Accompanying (hyper)graph for the circuit from which boundaries
        and detector coordinates are derived.
    logicals : List[Set[DecodingHyperEdge]]
        List of set of (hyper)edges that affect any logicals that exist.

    Returns
    -------
    stim.DetectorErrorModel
        stim.DetectorErrorModel for given pij_data and circuit.
    """
    boundaries = graph.boundaries
    d_graph: HyperMultiGraph
    if isinstance(graph, NXDecodingGraph):
        boundary = graph.boundaries
        edge_data = [(DecodingEdge(*edge), EdgeRecord(p_err=pij))
                     if len(edge) > 1
                     else (DecodingEdge(*edge, *boundary), EdgeRecord(p_err=pij))
                     for edge, pij in pij_data.items()]
        d_graph = NXDecodingGraph.from_edge_list(edge_data, graph.detector_records)
    elif isinstance(graph, DecodingHyperGraph):
        d_graph = DecodingHyperGraph([(DecodingHyperEdge(edge), EdgeRecord(p_err=pij))
                                     for edge, pij in pij_data.items()],
                                     graph.detector_records)
    else:
        d_graph = DecodingHyperGraph([], {})

    output = []
    for edge in d_graph.edges:
        detectors = " ".join([f"D{i}" for i in sorted(edge.vertices - boundaries)])
        logical_string = ""
        for i, edge_set in enumerate(logicals):
            # change when edge records are in
            for l_edge in edge_set:
                if edge == l_edge:
                    logical_string += f" L{i}"
                    break
        out_str = f"error({d_graph.edge_records[edge].p_err}) {detectors}" + \
            logical_string
        output.append(out_str + "\n")

    # add detector coords to end
    detector_coords = {k: [*list(v["spatial_coord"]), v["time"]]
                       for k, v in graph.detector_records.items()}
    for b in graph.boundaries:
        detector_coords.pop(b, None)
    for detector, d_coord in detector_coords.items():
        output.append(
            f"detector({int(d_coord[0])}, {int(d_coord[1])}, "
            f"{int(d_coord[2])}) D{int(detector)}\n")

    # append any unobserved logicals
    for i, l_set_len in enumerate([len(l_set) for l_set in logicals]):
        if l_set_len == 0:
            output.append(f"logical_observable L{i}\n")

    return stim.DetectorErrorModel("".join(output))


def pijs_edge_diff(
    pij1: PijData,
    pij2: PijData,
) -> tuple[set, set]:
    """Compare two Pij dicts to see if they contain the same edges.

    Parameters
    ----------
    pij1 : PijData
        The first Pij for comparison.

    pij2 : PijData
        The second Pij for comparison

    Returns
    -------
    Tuple[Set, Set]
        True if Pijs contain same edges, False otherwise.
        Tuple of the difference from pij1 to pij2 and pij2 to pij1,
        respectively.
    """
    set_pij1 = set(pij1.keys())
    set_pij2 = set(pij2.keys())
    return (set_pij1.difference(set_pij2), set_pij2.difference(set_pij1))


def pij_edges_max_diff(
    pij1: PijData,
    pij2: PijData,
) -> float:
    """Compare two Pij matrices containing identical edges w.r.t nodes
    and return the maximum difference between similar edges' probabilities.

    Parameters
    ----------
    pij1 : PijData
        The first Pij for comparison.

    pij2 : PijData
        The second Pij for comparison

    Returns
    -------
    float
        Maximum difference between similar edges of the two Pijs.
    """
    first_diff, second_diff = pijs_edge_diff(pij1, pij2)
    if len(first_diff) > 0 or len(second_diff) > 0:
        msg = (
            "Pij matrices do not contain identical edges so cannot compare for maximum "
            f"difference. Diff: {first_diff, second_diff}"
        )
        raise ValueError(msg)
    return max((abs(prob - pij2[edge]) for edge, prob in pij1.items()), default=0.0)


def pij_and_dem_edge_diff(
    dem: stim.DetectorErrorModel,
    pij: PijData,
    is_hypergraph: bool = False,
) -> tuple[set, set]:
    """Given a stim.DetectorErrorModel and Pij dict,
    compare the two to see if they contain the same edges
    w.r.t nodes in the edges. Return a tuple of sets
    containing the difference for the dem and Pij respectively.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        DEM to compare with Pij.

    pij : PijData
        Pij to compare with DEM.

    is_hypergraph : bool
        Specifies whether DEM/Pij describe a hypergraph or not.
        Default value is False.

    Returns
    -------
    Tuple[Set, Set]
        Tuple containing a set for the dem and pij that contain the
        elements contained in the dem/pij but not contained in the
        pij/dem respectively.
    """
    graph: HyperMultiGraph
    graph_edges: set[frozenset[int]]
    if is_hypergraph:
        graph, _ = dem_to_hypergraph_and_logicals(dem)
        graph_edges = {e.vertices for e in graph.edges}
    else:
        graph, _ = dem_to_decoding_graph_and_logicals(dem)
        boundary = next(iter(graph.boundaries), -1)
        graph_edges = {edge.vertices - {boundary} for edge in graph.edges}

    pij_edges = set(pij.keys())
    return (graph_edges.difference(pij_edges),
            pij_edges.difference(graph_edges))


def dem_and_pij_edges_max_diff(
    dem: stim.DetectorErrorModel,
    pij: PijData,
    is_hypergraph: bool = False,
) -> float:
    """Given a stim.DetectorErrorModel and Pij matrix,
    compare the two to find the maximum difference in
    probability between similar edges.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        DEM to compare with Pij.

    pij : PijData
        Pij to compare with DEM.

    is_hypergraph : bool
        Specifies whether DEM/Pij describe a hypergraph or not.
        Default value is False.

    Returns
    -------
    float
        float value representing the maximum difference in
        probability between similar edges between DEM and Pij.
    """
    first_diff, second_diff = pij_and_dem_edge_diff(
        dem, pij, is_hypergraph=is_hypergraph)
    if len(first_diff) != 0 or len(second_diff) != 0:
        msg = (
            "Pij matrices do not contain identical edges so cannot compare for maximum "
            f"difference. Diff: {first_diff, second_diff}"
        )
        raise ValueError(msg)
    graph: HyperMultiGraph
    graph_edges: PijData
    if is_hypergraph:
        graph, _ = dem_to_hypergraph_and_logicals(dem)
        graph_edges = {e.vertices: graph.edge_records[e].p_err for e in graph.edges}
    else:
        graph, _ = dem_to_decoding_graph_and_logicals(dem)
        boundary = next(iter(graph.boundaries), -1)
        # graph.edges is not guaranteed to give sorted tuple (smallest, largest)
        graph_edges = {x.vertices - {boundary}: graph.edge_records[x].p_err
                       for x in graph.edges}
    return max((abs(prob - pij[edge])
                for edge, prob in graph_edges.items()), default=0.0)
