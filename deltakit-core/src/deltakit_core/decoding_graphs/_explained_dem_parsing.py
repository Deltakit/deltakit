# (c) Copyright Riverlane 2020-2025.
from collections import defaultdict
from typing import List, Set

import stim
from deltakit_core.decoding_graphs import (
    AnyEdgeT,
    DecodingHyperEdge,
    DecodingHyperMultiGraph,
    DetectorRecord,
    EdgeRecord,
    HyperMultiGraph,
)


def depolarising_as_independent(probability: float, num_qubits: int) -> float:
    """A depoloarising noise channel is equivalent to applying each of its
    component Pauli channels independently with a carefully chosen
    probability.  This function computes that probability.

    Parameters
    ----------
    probability : float
        The probability of the depolarising error occurring.
    num_qubits : int
        The number of qubits the depolarising channel acts on.

    Returns
    -------
    float
        The equivalent probability of applying Pauli errors independently.
    """
    # Need to convert a probability of applying X, Y or Z into a probability
    # of applying X, Y, Z or I.
    pauli_combinations = 4**num_qubits
    mixing_probability = (pauli_combinations - 1) / pauli_combinations
    if probability > mixing_probability:
        raise ValueError(
            "Depolarising probability cannot be above the mixing "
            f"probability which is {mixing_probability}"
        )
    p_with_i = probability / mixing_probability
    exponent = 1 / 2 ** (2 * num_qubits - 1)
    return float((1 - (1 - p_with_i) ** exponent) / 2)


def noise_probability(noise_channel: stim.CircuitTargetsInsideInstruction) -> float:
    """Calculate the independent probability of a Pauli error happening from a
    given stim noise gate.

    Parameters
    ----------
    noise_channel : stim.CircuitTargetsInsideInstruction
        The stim noise channel

    Returns
    -------
    float
        The independent probability of a Pauli error.

    Raises
    ------
    TypeError
        If the gate type is not a noise channel or is a noise channel that
        cannot be expressed as independent error.
    """
    if noise_channel.gate == "DEPOLARIZE1":
        return depolarising_as_independent(noise_channel.args[0], 1)
    if noise_channel.gate == "DEPOLARIZE2":
        return depolarising_as_independent(noise_channel.args[0], 2)
    if noise_channel.gate in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        return float(noise_channel.args[0])
    raise TypeError(f"Unsupported gate type: {noise_channel.gate}")


def parse_explained_dem(
    explained_dem: List[stim.ExplainedError],
) -> DecodingHyperMultiGraph:
    """Parse an explained DEM into a hyper-multi-graph. Information about
    which logicals an edge affects is stored in the edge record of that edge

    Parameters
    ----------
    explained_dem : List[stim.ExplainedError]
        The explained DEM which comes from the
        `stim.Circuit.explain_detector_error_model_errors` method.

    Returns
    -------
    DecodingHyperMultiGraph
        A HyperMultiGraph object which represents all possible errors in the
        explained DEM.

    Raises
    ------
    TypeError
        If there is a target separator (^) in the explained DEM since we don't
        expect this character to appear there.
    """
    detector_records = {}
    edges = []
    for error in explained_dem:
        edge_vertices = set()
        logicals_affected = []
        for dem_term in error.dem_error_terms:
            target = dem_term.dem_target
            if target.is_separator():
                raise TypeError("Target separators should not be in the explained DEM.")
            if target.is_relative_detector_id():
                edge_vertices.add(target.val)
                *spatial_coords, time = dem_term.coords
                detector_records[target.val] = DetectorRecord(
                    tuple(spatial_coords), time=time
                )
            elif target.is_logical_observable_id():
                logicals_affected.append(target.val)

        edge = DecodingHyperEdge(edge_vertices)
        for error_location in error.circuit_error_locations:
            p_err = noise_probability(error_location.instruction_targets)
            edges.append(
                (edge, EdgeRecord(p_err=p_err, logicals_affected=logicals_affected))
            )

    return DecodingHyperMultiGraph(edges, detector_records=detector_records)


def extract_logicals(graph: HyperMultiGraph[AnyEdgeT]) -> List[Set[AnyEdgeT]]:
    """Extract which edges affect which logical observables from a graph by
    looking through the edge records.

    Parameters
    ----------
    graph : HyperMultiGraph[AnyEdgeT]
        The graph containing edges records annotated with which edges affect
        which logicals.

    Returns
    -------
    List[Set[AnyEdgeT]]
        A list where each index represents a logical and the set of edges in
        at each index are the edges in the graph which affect that logical.
    """
    max_logical = 0
    logicals_dict = defaultdict(set)
    for edge, edge_record in graph.edge_records.items():
        for logical in edge_record.get("logicals_affected", []):
            max_logical = max(max_logical, logical)
            logicals_dict[logical].add(edge)

    # Logicals are zero indexed so we must plus one
    logicals: List[Set[AnyEdgeT]] = [set() for _ in range(max_logical + 1)]
    for logical, edges in logicals_dict.items():
        logicals[logical] = edges

    return logicals
