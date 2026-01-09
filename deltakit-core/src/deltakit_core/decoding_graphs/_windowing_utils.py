# (c) Copyright Riverlane 2020-2025.
"""Utility functions for windowing."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from deltakit_core.decoding_graphs._data_qubits import EdgeRecord
from deltakit_core.decoding_graphs._decoding_graph import (
    DecodingHyperEdge,
    DecodingHyperGraph,
)


def nodes_within_radius(
    hypergraph: DecodingHyperGraph, start_nodes: Iterable[int], radius: int
) -> set[int]:
    """Compute nodes within a given unweighted graph distance from any start node.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        The global decoding hypergraph.
    start_nodes : Iterable[int]
        Nodes from which to grow the region.
    radius : int
        Number of unweighted adjacency steps to grow.

    Returns
    -------
    Set[int]
        Set of detector node ids within the specified hop distance of any start node.

    """
    if radius < 0:
        msg = "Radius must be non-negative"
        raise ValueError(msg)
    visited: set[int] = set(start_nodes)
    if radius == 0:
        return visited
    frontier: deque[tuple[int, int]] = deque((n, 0) for n in visited)
    while frontier:
        node, dist = frontier.popleft()
        if dist == radius:
            continue
        for neighbour in hypergraph.neighbors(node):
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append((neighbour, dist + 1))
    return visited


def weighted_nodes_within_radius(
    hypergraph: DecodingHyperGraph, start_nodes: Iterable[int], radius: float
) -> set[int]:
    """Compute nodes within a weighted distance budget from any start node.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        The global decoding hypergraph.
    start_nodes : Iterable[int]
        Seed nodes to begin weighted growth.
    radius : float
        Maximum accumulated edge weight distance.

    Returns
    -------
    Set[int]
        Nodes reachable with total path weight <= ``radius``.
    """
    msg = "Weighted growth is not yet implemented."
    raise NotImplementedError(msg)


def induce_subhypergraph(
    hypergraph: DecodingHyperGraph, nodes: set[int]
) -> DecodingHyperGraph:
    """Form the induced sub-hypergraph on the provided set of nodes. Keeps only
    hyperedges whose vertices are fully contained in ``nodes`` and restricts detector
    records accordingly.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        Source decoding hypergraph.
    nodes : Set[int]
        Set of hyergraph nodes to induce subgraph from.

    Returns
    -------
    DecodingHyperGraph
        New hypergraph containing only edges fully within ``nodes`` and detector records
        for those nodes.
    """
    edge_data: list[tuple[DecodingHyperEdge, EdgeRecord]] = []
    for edge, record in hypergraph.edge_records.items():
        if edge.vertices.issubset(nodes):
            edge_data.append((edge, record))
    det_records = {n: hypergraph.detector_records[n] for n in nodes}
    return DecodingHyperGraph(edge_data, detector_records=det_records)


def connect_dangling_to_boundary_hypergraph(
    supergraph: DecodingHyperGraph, subgraph: DecodingHyperGraph
) -> DecodingHyperGraph:
    """Fold edges leaving the subgraph into unary boundary edges per node.

    For each node in ``subgraph``, consider all edges in ``supergraph`` that include the
    node and at least one vertex outside ``subgraph``. These excluded edges are folded
    into a single unary hyperedge on that node with ``p_err`` equal to the combined
    probability of their probabilities. If a unary edge already exists on the node in
    the subgraph, its probability is combined with the folded probability.

    Parameters
    ----------
    supergraph : DecodingHyperGraph
        The original global hypergraph.
    subgraph : DecodingHyperGraph
        The induced subgraph to which boundary edges will be added.

    Returns
    -------
    DecodingHyperGraph
        A new hypergraph with updated/added unary boundary edges and the same detector
        records as ``subgraph``.
    """

    def _combine_probabilities(prob1: float, prob2: float) -> float:
        """Helper to combine two edge probabilities."""
        return prob1 * (1 - prob2) + (1 - prob1) * prob2

    nodes = set(subgraph.nodes)
    folded_records = dict(subgraph.edge_records)

    for node in nodes:
        folded_p = 0.0
        for edge in supergraph.incident_edges(node):
            if not edge.vertices.issubset(nodes):
                folded_p = _combine_probabilities(
                    folded_p, supergraph.edge_records[edge].p_err
                )
        if folded_p == 0.0:
            continue
        unary_edge = DecodingHyperEdge((node,))
        if unary_edge in folded_records:
            existing_p = folded_records[unary_edge].p_err
            folded_records[unary_edge] = EdgeRecord(
                p_err=_combine_probabilities(existing_p, folded_p)
            )
        else:
            folded_records[unary_edge] = EdgeRecord(p_err=folded_p)

    edge_data = list(folded_records.items())
    return DecodingHyperGraph(edge_data, detector_records=subgraph.detector_records)


def relabel_hypergraph_nodes_contiguously(
    hypergraph: DecodingHyperGraph,
) -> tuple[DecodingHyperGraph, dict[int, int]]:
    """Return a new hypergraph with nodes relabelled to a contiguous 0..N-1 range.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        The hypergraph whose node labels may be sparse
        relative to global indices.

    Returns
    -------
    Tuple[DecodingHyperGraph, Dict[int, int]]
        A tuple of (relabelled_hypergraph, global_to_window_detector_map) where
        the mapping gives global node id -> window id.
    """
    node_mapping: dict[int, int] = {
        old: i for i, old in enumerate(sorted(hypergraph.nodes))
    }

    remapped_edge_data: list[tuple[DecodingHyperEdge, EdgeRecord]] = []
    for edge, record in hypergraph.edge_records.items():
        remapped_edge = DecodingHyperEdge(tuple(node_mapping[n] for n in edge))
        remapped_edge_data.append((remapped_edge, record))

    remapped_detector_records = {
        node_mapping[n]: rec
        for n, rec in hypergraph.detector_records.items()
        if n in node_mapping
    }

    relabelled = DecodingHyperGraph(
        remapped_edge_data, detector_records=remapped_detector_records
    )
    return relabelled, node_mapping


def expand_nodes_to_time_span(
    hypergraph: DecodingHyperGraph, nodes: set[int]
) -> set[int]:
    """Expand ``nodes`` so that for every time value represented among ``nodes`` all
    detectors in the hypergraph with that time are included.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        Hypergraph providing detector records.
    nodes : Set[int]
        Initial grown node set in hypergraph.

    Returns
    -------
    Set[int]
        Expanded node set containing full rounds.
    """
    if not nodes:
        return set()
    times = [hypergraph.detector_records[n].time for n in nodes]
    t_min, t_max = min(times), max(times)
    return {
        n
        for n, rec in hypergraph.detector_records.items()
        if t_min <= rec.time <= t_max
    }
