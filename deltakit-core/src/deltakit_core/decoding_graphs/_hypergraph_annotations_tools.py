# (c) Copyright Riverlane 2020-2025.
"""Module for utilities on annotating hypergraphs with window ids."""

from collections import defaultdict
from collections.abc import Callable
from itertools import combinations

from networkx import DiGraph
from networkx.algorithms import simple_cycles

from deltakit_core.decoding_graphs._data_qubits import DecodingHyperEdge
from deltakit_core.decoding_graphs._decoding_graph import (
    DecodingHyperGraph,
    HyperLogicals,
)

CoordinateToWindowId = Callable[[tuple[float, ...]], int]


def annotate_edges_with_window_ids(
    hypergraph: DecodingHyperGraph,
    coordinate_to_window_id: CoordinateToWindowId,
    clear_edges: bool = True,
) -> None:
    """Annotate edges with window_ids derived from node coordinates.

    For each detector, compute a window_id using the provided
    `coordinate_to_window_id` function (based on the node's full_coord).
    Each edge's `window_id` is set to the minimum window_id of its adjacent
    nodes. Optionally clears any existing edge `window_id` annotations first.

    Using the minimum detector enforces that windows flow from lower to higher
    `window ids`, which produces an acyclic transfer graph when deriving inter-window
    dependencies.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        Hypergraph whose edges will be annotated.
    coordinate_to_window_id : Callable[[tuple[float, ...]], int]
        Function mapping a detector's full_coord (including time) to a window id.
    clear_edges : bool
        If True, clear existing edge window_ids before annotating. Default True.
    """
    if clear_edges:
        for edge_record in hypergraph.edge_records.values():
            edge_record.pop("window_id", None)

    node_to_window_id = {
        det_id: coordinate_to_window_id(det_record.full_coord)
        for det_id, det_record in hypergraph.detector_records.items()
    }

    for edge, edge_record in hypergraph.edge_records.items():
        edge_record["window_id"] = min(node_to_window_id[det] for det in edge)


def get_window_id_transfer_graph(hypergraph: DecodingHyperGraph) -> DiGraph:
    """Given a decoding hypergraph with `window_id` fields in edge records,
    construct a transfer dependency graph. Transfers are unidirectional
    and always go from lower to higher `window_id`.

    Two window_ids are dependent if there are adjacent edges (sharing the same node)
    with those window_ids.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        Global hypergraph where all edges have an associated
        `window_id`.

    Returns
    -------
    DiGraph
        Directed transfer hypergraph whose nodes are the unique
        window_ids of the given hypergraph.
    """
    transfer_graph: DiGraph = DiGraph()
    for node in hypergraph.nodes:
        shared_window_ids = {
            hypergraph.edge_records[edge]["window_id"]
            for edge in hypergraph.incident_edges(node)
        }
        for lbl1, lbl2 in combinations(shared_window_ids, 2):
            src, sink = (lbl2, lbl1) if lbl2 < lbl1 else (lbl1, lbl2)
            transfer_graph.add_edge(src, sink)
    # sanity check that there are no cycles
    if len(list(simple_cycles(transfer_graph))) != 0:
        msg = "Transfer hypergraph cannot have cycles!"
        raise ValueError(msg)
    return transfer_graph


def get_unique_window_ids(hypergraph: DecodingHyperGraph) -> set[int]:
    """Return the set of unique edge `window_id`s in the hypergraph.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        Global hypergraph where all edges have an associated `window_id`.

    Returns
    -------
    set[int]
        set of unique edge `window_id`s.
    """
    return {
        edge_record["window_id"] for edge_record in hypergraph.edge_records.values()
    }


def get_unique_logical_window_ids(
    hypergraph: DecodingHyperGraph, logicals: HyperLogicals
) -> set[int]:
    """Gets the unique edge `window_id`s across the provided logicals (edge sets).

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        Global hypergraph where all logical edges have an associated window_id.
    logicals : HyperLogicals
        Global logicals to get the window_ids for.

    Returns
    -------
    set[int]
        Unique edge `window_id`s present among logical edges.
    """
    return {
        hypergraph.edge_records[edge]["window_id"]
        for logical in logicals
        for edge in logical
    }


def separate_edges_per_window_id(
    graph: DecodingHyperGraph,
    enforce_window_ids: bool = True,
) -> dict[int, list[DecodingHyperEdge]]:
    """Separates labelled graph edges into lists keyed by ``window_id``.

    Parameters
    ----------
    graph : DecodingHyperGraph
        Hypergraph whose edges are (partially) annotated with ``window_id``.
    enforce_window_ids : bool, optional
        If True, raises ``ValueError`` if any edge lacks a ``window_id`` label.

    Returns
    -------
    dict[int, list[DecodingHyperEdge]]
        Mapping from window id to list of edges with that label.
    """
    windows_edges: dict[int, list[DecodingHyperEdge]] = defaultdict(list)
    unlabelled: list[DecodingHyperEdge] = []
    for edge, edge_record in graph.edge_records.items():
        if (label := edge_record.get("window_id")) is not None:
            windows_edges[label].append(edge)
        else:
            unlabelled.append(edge)
    if enforce_window_ids and unlabelled:
        msg = f"Found {len(unlabelled)} unlabelled edges when enforce_window_ids=True"
        raise ValueError(msg)
    return windows_edges


def separate_logicals_by_window_id(
    hypergraph: DecodingHyperGraph, logicals: HyperLogicals
) -> dict[int, HyperLogicals]:
    """Separate logicals by edge `window_id`.

    Returns a dictionary mapping window_id to a list of logicals (each logical is a
    frozenset of edges) that have that window_id. Keys are in ascending order.

    Parameters
    ----------
    hypergraph : DecodingHyperGraph
        Global hypergraph where all logical edges have an associated window_id.
    logicals : HyperLogicals
        Global logicals.

    Returns
    -------
    dict[int, list[list[DecodingHyperEdge]]]
        dictionary from logical window_ids to the list of lists of edges containing
        logicals that have that window_id.
    """
    num_logicals = len(logicals)
    all_window_ids = get_unique_logical_window_ids(hypergraph, logicals)
    logicals_by_window_id: dict[int, list[list[DecodingHyperEdge]]] = {
        lbl: [[] for _ in range(num_logicals)] for lbl in sorted(all_window_ids)
    }
    for i, logical in enumerate(logicals):
        for edge in logical:
            window_id = hypergraph.edge_records[edge]["window_id"]
            logicals_by_window_id[window_id][i].append(edge)
    return {
        key: [frozenset(log) for log in logicals]
        for key, logicals in logicals_by_window_id.items()
    }
