# (c) Copyright Riverlane 2020-2025.
"""Module for functions for weighted decoding graphs."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from deltakit_core.decoding_graphs._decoding_graph import (
    DecodingEdge,
    EdgeRecord,
    NXDecodingGraph,
)


def change_graph_error_probabilities(
    graph: NXDecodingGraph, new_p_errors: Sequence[float]
) -> NXDecodingGraph:
    """Make a new graph from the given graph where the error probabilities on the edges
    are changed to the new error probabilities. The sequence of new error probabilities
    should correspond to the sequence of edges in the given graph.

    Parameters
    ----------
    graph : NXDecodingGraph
        The graph to use to with the old error probabilities.
    new_p_errors : Sequence[float]
        The new error probabilities to apply to the graph.

    Returns
    -------
    NXDecodingGraph
        A new graph instance with the same vertices, edges and boundaries as
        the input graph but with an updated error proability on each edge.

    Raises
    ------
    ValueError
        If the length of the new error probabilities and edges do not match.
    """
    if len(graph.edges) != len(new_p_errors):
        msg = (
            "There should be an equal number of new error probabilities to edges in "
            "the graph."
        )
        raise ValueError(msg)
    return NXDecodingGraph.from_edge_list(
        [
            (DecodingEdge(u, v), EdgeRecord(p_err=p_err))
            for (u, v), p_err in zip(graph.edges, new_p_errors, strict=False)
        ],
        detector_records=graph.detector_records,
        boundaries=graph.boundaries,
    )


def vector_weights(
    graph: NXDecodingGraph,
) -> defaultdict[tuple[float, ...], list[float]]:
    """Treat the edges of a graph as vectors and return a mapping of
    normalised vectors to a list of weights on those vectors.

    Parameters
    ----------
    graph : NXDecodingGraph
        The edges on this graph represent vectors.

    Returns
    -------
    DefaultDict[Tuple[float, ...], List[float]]
    """
    vectors = defaultdict(list)
    for edge in graph.edges:
        u, v = edge
        edge_record = graph.edge_records[edge]
        if (u_coord := graph.detector_records[u].full_coord) == (0,):
            msg = f"{u} does not have proper coordinates."
            raise ValueError(msg)
        if (v_coord := graph.detector_records[v].full_coord) == (0,):
            msg = f"{v} does not have proper coordinates."
            raise ValueError(msg)
        if u in graph.boundaries or v in graph.boundaries:
            vectors[(0.0,) * len(u_coord)].append(edge_record.weight)
        else:
            vector = np.subtract(v_coord, u_coord)
            # Take the convention that the first non-zero element of the
            # vector must be positive.
            vector *= np.sign(vector[vector != 0][0])
            vectors[tuple(vector.tolist())].append(edge_record.weight)
    return vectors
