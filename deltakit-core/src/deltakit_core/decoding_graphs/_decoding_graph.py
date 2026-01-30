# (c) Copyright Riverlane 2020-2025.
"""Datastructures for decoding graphs."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from collections.abc import (
    Set as AbstractSet,
)
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, tee
from typing import Any, ClassVar, Generic, TypeGuard, TypeVar, cast

import networkx as nx
import numpy as np
import numpy.typing as npt

from deltakit_core.decoding_graphs._data_qubits import (
    DecodingEdge,
    DecodingHyperEdge,
    EdgeRecord,
)
from deltakit_core.decoding_graphs._syndromes import DetectorRecord, OrderedSyndrome


def _is_decoding_edge_and_edge_record(
    data: DecodingEdge | tuple[int, ...] | tuple[DecodingEdge, EdgeRecord],
) -> TypeGuard[tuple[DecodingEdge, EdgeRecord]]:
    return (
        not isinstance(data, DecodingEdge)
        and isinstance(data[0], DecodingEdge)
        and isinstance(data[-1], EdgeRecord)
    )


def _is_decoding_hyper_edge_and_edge_record(
    data: DecodingHyperEdge | tuple[int, ...] | tuple[DecodingHyperEdge, EdgeRecord],
) -> TypeGuard[tuple[DecodingHyperEdge, EdgeRecord]]:
    return (
        not isinstance(data, DecodingHyperEdge)
        and isinstance(data[0], DecodingHyperEdge)
        and isinstance(data[-1], EdgeRecord)
    )


def _is_tuple_of_ints(
    data: DecodingHyperEdge | tuple[int, ...] | tuple[DecodingHyperEdge, EdgeRecord],
) -> TypeGuard[tuple[int, ...]]:
    return not isinstance(data, DecodingHyperEdge) and all(
        isinstance(v, int) for v in data
    )


DecodingGraphT = TypeVar("DecodingGraphT")
AnyEdgeT = TypeVar("AnyEdgeT")


def convolve_probabilities(edge_records: Iterable[EdgeRecord]) -> EdgeRecord:
    """Computes the probability that an odd number of independent events
    with the given probabilities occur.

    Parameters
    ----------
    edge_records : Iterable[EdgeRecord]
        All of the edge records associated with a particular edge.

    Returns
    -------
    EdgeRecord
        A single edge record which is the product of the individual
        probabilities of each edge record.
    """
    combined_probabilities = math.prod(1 - 2 * record.p_err for record in edge_records)
    return EdgeRecord((1 - combined_probabilities) / 2)


class HyperMultiGraph(ABC, Generic[AnyEdgeT]):
    """Class for abstract immutable multigraph with hyperedges. Any set of
    vertices can be connected by one or more edges in the multi-hypergraph,
    forming the basis of all other graph objects."""

    @property
    @abstractmethod
    def edges(self) -> Sequence[AnyEdgeT]:
        """Return all edges of this graph."""

    @property
    @abstractmethod
    def nodes(self) -> Sequence[int]:
        """Return all nodes of this graph."""

    @property
    @abstractmethod
    def boundary_edges(self) -> Iterable[AnyEdgeT]:
        """Fetch the edges that are touching the boundary of the graph."""

    @abstractmethod
    def get_edges(self, *detectors: int) -> Iterable[AnyEdgeT]:
        """Generator for the edges defined by the given set of detectors."""

    @abstractmethod
    def incident_edges(self, detector: int) -> Iterator[AnyEdgeT]:
        """Iterator for the incident edges connected to a given detector."""

    @abstractmethod
    def neighbors(self, detector: int) -> Iterator[int]:
        """Iterator for the neighbours of a given detector."""

    @property
    @abstractmethod
    def edge_records(self) -> Mapping[AnyEdgeT, EdgeRecord]:
        """Return mapping of edge records, keyed by edge."""

    @property
    @abstractmethod
    def detector_records(self) -> Mapping[int, DetectorRecord]:
        """Return mapping of detector records, keyed by detector."""

    @property
    def boundaries(self) -> frozenset[int]:
        """Return all detectors in this graph that are labelled as boundaries."""
        return frozenset()

    @abstractmethod
    def detector_is_boundary(self, detector: int) -> bool:
        """Return True if given detector is a boundary, False otherwise."""

    @abstractmethod
    def to_parity_check_matrix(self) -> npt.NDArray[np.uint8]:
        """Convert the hypergraph to a parity check matrix  of size
        `(len(nodes), len(edges))`. Each column represents an edge and
        the non-zero entries in that column represent detectors in the edge.

        The nodes in the graph must be contiguous to construct the matrix.
        """

    @abstractmethod
    def error_to_syndrome(self, edges: Iterable[AnyEdgeT]) -> OrderedSyndrome:
        """Assume an error has happened on the given edges and return the
        corresponding syndrome.

        Parameters
        ----------
        edges : Iterable[AnyEdgeT]
            Edges which had errors.

        Returns
        -------
        OrderedSyndrome
        """


class DecodingHyperMultiGraph(HyperMultiGraph[tuple[DecodingHyperEdge, int]]):
    """Representation of a decoding multigraph with hyper-edges.

    Parameters
    ----------
    multi_hyper_edges: Iterable[Tuple[DecodingHyperEdge, int]]
        Edges that form this graph.
    detector_records : Optional[Dict[int, DetectorRecord]], optional
        Optional mapping of detector indices to detector records, by default None.
    """

    def __init__(
        self,
        edge_data: Iterable[
            DecodingHyperEdge | tuple[int, ...] | tuple[DecodingHyperEdge, EdgeRecord]
        ],
        detector_records: dict[int, DetectorRecord] | None = None,
    ):
        edge_records: dict[tuple[DecodingHyperEdge, int], EdgeRecord] = {}
        edges: list[tuple[DecodingHyperEdge, int]] = []
        edge_count: dict[DecodingHyperEdge, int] = defaultdict(int)

        for data in edge_data:
            if isinstance(data, DecodingHyperEdge):
                edge = data
                edge_record = EdgeRecord()
            elif _is_decoding_hyper_edge_and_edge_record(data):
                edge, edge_record = data
            elif _is_tuple_of_ints(data):
                edge = DecodingHyperEdge(data)
                edge_record = EdgeRecord()
            else:
                msg = f"Invalid edge data {data}"
                raise ValueError(msg)
            edge_id = edge_count[edge]
            edge_count[edge] += 1
            edge_records[(edge, edge_id)] = edge_record
            edges.append((edge, edge_id))

        self._edges = edges
        if len(set(edges)) != len(edges):
            msg = "Each edge must have a unique integer identifier"
            raise ValueError(msg)
        self._detector_records = {} if detector_records is None else detector_records
        self._edge_records = edge_records

    @cached_property
    def _nodes_in_edges(self) -> set[int]:
        return {node for edge, _ in self._edges for node in edge}

    @property
    def detector_records(self) -> dict[int, DetectorRecord]:
        return self._detector_records

    @property
    def edge_records(self) -> dict[tuple[DecodingHyperEdge, int], EdgeRecord]:
        return self._edge_records

    @property
    def edges(self) -> list[tuple[DecodingHyperEdge, int]]:
        return self._edges

    @cached_property
    def nodes(self) -> list[int]:
        return sorted(self._nodes_in_edges.union(set(self._detector_records.keys())))

    @cached_property
    def boundary_edges(self) -> list[tuple[DecodingHyperEdge, int]]:
        return [(edge, edge_id) for edge, edge_id in self.edges if len(edge) == 1]

    @cached_property
    def _nodes_to_full_edge(
        self,
    ) -> defaultdict[frozenset[int], list[tuple[DecodingHyperEdge, int]]]:
        nodes_to_full_edge = defaultdict(list)
        for edge, edge_id in self.edges:
            nodes_to_full_edge[edge.vertices].append((edge, edge_id))
        return nodes_to_full_edge

    @cached_property
    def _node_to_incident_edges(
        self,
    ) -> defaultdict[int, list[tuple[DecodingHyperEdge, int]]]:
        detector_to_edges = defaultdict(list)
        for edge, edge_id in self.edges:
            for node in edge.vertices:
                detector_to_edges[node].append((edge, edge_id))
        return detector_to_edges

    def get_edges(self, *detectors: int) -> list[tuple[DecodingHyperEdge, int]]:
        """Fetch the edges connecting a sequence of detectors."""
        return self._nodes_to_full_edge[frozenset(detectors)]

    def incident_edges(self, detector: int) -> Iterator[tuple[DecodingHyperEdge, int]]:
        yield from self._node_to_incident_edges[detector]

    def neighbors(self, detector: int) -> Iterator[int]:
        incident_edges = self.incident_edges(detector)
        neighbour_nodes = {
            node
            for edge, _ in incident_edges
            for node in edge.vertices
            if node != detector
        }
        yield from neighbour_nodes

    def detector_is_boundary(self, detector: int) -> bool:  # noqa: ARG002
        return False

    def to_parity_check_matrix(self) -> npt.NDArray[np.uint8]:
        check_matrix = np.zeros((len(self.nodes), len(self.edges)), dtype=np.uint8)
        for edge_index, (edge, _) in enumerate(self.edges):
            check_matrix[tuple(edge), (edge_index,)] = True
        return check_matrix

    def with_multi_edges_merged(
        self,
        combine_edge_records: Callable[
            [Iterable[EdgeRecord]], EdgeRecord
        ] = convolve_probabilities,
    ) -> DecodingHyperGraph:
        """Create a `DecodingHyperGraph` instance with the same nodes but with
        only single edges, merging the multi-edge edge records to a single edge
        record.

        Parameters
        ----------
        combine_edge_records : Callable[[Iterable[EdgeRecord]], EdgeRecord], optional
            A function which combines all the edge records for a given edge,
            by default convolve_probabilities which applies the generalised
            `p * (1 - q) + q * (1 - p)`.

        Returns
        -------
        DecodingHyperGraph
            A hypergraph with the same properties as this graph but with all
            multi-edges merged to a single edge.
        """
        grouped_edge_records = defaultdict(list)
        for (edge, _), edge_record in self.edge_records.items():
            grouped_edge_records[edge].append(edge_record)
        return DecodingHyperGraph(
            edge_data=(
                (edge, combine_edge_records(records))
                for edge, records in grouped_edge_records.items()
            ),
            detector_records=self.detector_records,
        )

    def error_to_syndrome(
        self, edges: Iterable[tuple[DecodingHyperEdge, int]]
    ) -> OrderedSyndrome:
        return OrderedSyndrome(
            symptom for symptom in chain.from_iterable(edge for edge, _ in edges)
        )


HyperLogicals = Sequence[AbstractSet[DecodingHyperEdge]]


class DecodingHyperGraph(HyperMultiGraph[DecodingHyperEdge]):
    """Representation of a decoding hypergraph, built to handle Stim-based noise
    sources. Each hyperedge in the graph has a weight between 0 and 1 (inclusive).

    Parameters
    ----------
    hyper_edges : Iterable[DecodingHyperEdge]
        Edges that form this graph.
    detector_records : Optional[Dict[int, DetectorRecord]], optional
        Optional mapping of detector indices to detector records, by default None.
    edge_records : Optional[Dict[DecodingHyperEdge, EdgeRecord]], optional
        Optional mapping of edges to edge records, by default None.
    """

    def __init__(
        self,
        edge_data: Iterable[
            DecodingHyperEdge | tuple[int, ...] | tuple[DecodingHyperEdge, EdgeRecord]
        ],
        detector_records: dict[int, DetectorRecord] | None = None,
    ):
        edge_records: dict[DecodingHyperEdge, EdgeRecord] = {}
        edges: list[DecodingHyperEdge] = []
        for data in edge_data:
            if _is_decoding_hyper_edge_and_edge_record(data):
                edge, edge_record = data
                edge_records[edge] = edge_record
            elif isinstance(data, DecodingHyperEdge):
                edge = data
                edge_records[edge] = EdgeRecord()
            elif _is_tuple_of_ints(data):
                edge = DecodingHyperEdge(data)
                edge_records[edge] = EdgeRecord()
            else:
                msg = f"Invalid edge data {data}"
                raise ValueError(msg)
            edges.append(edge)

        self._edges = edges
        if len(set(edges)) != len(edges):
            msg = "Each edge must have a unique integer identifier"
            raise ValueError(msg)
        self._detector_records = {} if detector_records is None else detector_records
        self._edge_records = edge_records
        self._nodes_in_detector_records = set(self._detector_records.keys())

    @cached_property
    def _nodes_in_edges(self) -> set[int]:
        return set(chain.from_iterable(self._edges))

    @property
    def detector_records(self) -> dict[int, DetectorRecord]:
        return self._detector_records

    @property
    def edge_records(self) -> dict[DecodingHyperEdge, EdgeRecord]:
        return self._edge_records

    @cached_property
    def _node_to_incident_edges(self) -> defaultdict[int, list[DecodingHyperEdge]]:
        detector_to_edges = defaultdict(list)
        for edge in self.edges:
            for node in edge.vertices:
                detector_to_edges[node].append(edge)
        return detector_to_edges

    @cached_property
    def edges(self) -> list[DecodingHyperEdge]:
        return self._edges

    @cached_property
    def nodes(self) -> list[int]:
        return sorted(self._nodes_in_edges.union(self._nodes_in_detector_records))

    @cached_property
    def boundary_edges(self) -> list[DecodingHyperEdge]:
        return [edge for edge in self.edges if len(edge) == 1]

    def get_edges(self, *detectors: int) -> list[DecodingHyperEdge]:
        return (
            [edge]
            if (edge := DecodingHyperEdge(detectors)) in self.edge_records
            else []
        )

    def get_edge(self, *detectors: int) -> DecodingHyperEdge:
        """Returns the unique edge in the graph corresponding to the given set
        of detector indices.

        Raises
        ------
        ValueError
            If no edge in the graph corresponds to the given detectors.
        """
        if (edge := DecodingHyperEdge(detectors)) in self.edges:
            return edge
        msg = f"Detectors {detectors} do not belong to any edge in the graph."
        raise ValueError(msg)

    def get_edge_record(self, *detectors: int) -> EdgeRecord:
        """Fetch the edge record for a sequence of detectors."""
        return self.edge_records[self.get_edge(*detectors)]

    def incident_edges(self, detector: int) -> Iterator[DecodingHyperEdge]:
        yield from self._node_to_incident_edges[detector]

    def neighbors(self, detector: int) -> Iterator[int]:
        incident_edges = self.incident_edges(detector)
        neighbour_nodes = {
            node
            for edge in incident_edges
            for node in edge.vertices
            if node != detector
        }
        yield from neighbour_nodes

    def to_nx_decoding_graph(self) -> NXDecodingGraph:
        """Lower hypergraph to NXDecodingGraph, on the condition that the
        hypergraph does not contain any hyperedges.
        """
        if not self.nodes:
            return NXDecodingGraph.from_edge_list([])
        possible_boundary = max(self.nodes) + 1
        normal_edges: list[tuple[DecodingEdge, EdgeRecord]] = [
            (edge.to_decoding_edge(possible_boundary), self.edge_records[edge])
            for edge in self._edges
        ]
        return NXDecodingGraph.from_edge_list(
            normal_edges,
            self._detector_records,
            boundaries=(
                [possible_boundary]
                if any(possible_boundary in edge for edge, _ in normal_edges)
                else []
            ),
        )

    def to_parity_check_matrix(self) -> npt.NDArray[np.uint8]:
        check_matrix = np.zeros((len(self.nodes), len(self.edges)), dtype=np.uint8)
        for edge_index, edge in enumerate(self.edges):
            check_matrix[tuple(edge), (edge_index,)] = True
        return check_matrix

    def detector_is_boundary(self, detector: int) -> bool:  # noqa: ARG002
        return False

    def error_to_syndrome(self, edges: Iterable[DecodingHyperEdge]) -> OrderedSyndrome:
        return OrderedSyndrome(symptom for symptom in chain.from_iterable(edges))

    def get_relevant_nodes(
        self, logicals: Iterable[set[DecodingHyperEdge]]
    ) -> set[int]:
        """Return nodes that have a hyperedge with a path to logical.

        Hyperedges to boundaries are ignored if applicable.
        """
        union_find: nx.utils.UnionFind[int] = nx.utils.UnionFind()
        for edge in self.edges:
            union_find.union(*edge)

        relevant_nodes: set[int] = set()
        components: list[set[int]] = list(union_find.to_sets())
        for logical in logicals:
            relevant_nodes.update(chain.from_iterable(logical))
            for component in components:
                if any(component.intersection(edge) for edge in logical):
                    relevant_nodes.update(component)
        return relevant_nodes


@dataclass(frozen=True)
class DecodingCode:
    """Convenience class encapsulating the annotated hypergraph and logicals."""

    hypergraph: DecodingHyperGraph
    logicals: HyperLogicals

    def to_nx_code(self) -> NXCode:
        """Lower hypergraph to graph, if there are no hyper edges."""
        graph = self.hypergraph.to_nx_decoding_graph()
        return NXCode(
            graph=graph,
            logicals=[
                {
                    edge.to_decoding_edge(next(iter(graph.boundaries)))
                    for edge in edge_set
                }
                for edge_set in self.logicals
            ],
        )


class _QECNX(nx.Graph):
    """Custom nx graph subclass that uses special dictionary classes for attributes."""

    node_attr_dict_factory = DetectorRecord
    edge_attr_dict_factory = EdgeRecord


class _QECNXMG(nx.MultiGraph):
    """Custom nx.MultiGraph subclass that uses special dictionary classes for
    attributes."""

    node_attr_dict_factory = DetectorRecord
    edge_attr_dict_factory = EdgeRecord


NXGraphT = TypeVar("NXGraphT", bound=nx.Graph)


class NXGraph(HyperMultiGraph[AnyEdgeT], Generic[NXGraphT, AnyEdgeT]):
    """Class for all NetworkX-based graph methods that are shared between
    NXDecodingGraph and NXDecodingMultiGraph
    """

    def __init__(self, graph: NXGraphT, boundaries: Iterable[int] = frozenset()):
        self._graph: NXGraphT = nx.freeze(graph)
        self._boundaries = frozenset(boundaries)
        if any(boundary not in self.nodes for boundary in self.boundaries):
            msg = f"Boundaries {boundaries} are not in nodes {self.nodes}."
            raise ValueError(msg)
        max_index = max(self.nodes, default=0)
        self._boundaries_lookup = tuple(
            i in self.boundaries for i in range(max_index + 1)
        )

    def __getstate__(self) -> dict[str, Any]:
        inner_state = self.__dict__.copy()
        if "no_boundary_view" in inner_state:
            del inner_state["no_boundary_view"]
        return inner_state

    @property
    def graph(self) -> NXGraphT:
        """Get an immutable view of the whole NetworkX graph."""
        return self._graph

    @cached_property
    def no_boundary_view(self) -> NXGraphT:
        """Get an immutable view of the NetworkX graph without boundaries."""
        subgraph_view = nx.subgraph_view(
            self.graph, filter_node=lambda node: node not in self.boundaries
        )
        return cast(NXGraphT, subgraph_view)

    @cached_property
    def nodes(self) -> list[int]:
        return sorted(self.graph.nodes)

    @property
    def boundaries(self) -> frozenset[int]:
        """Return all nodes of this graph that are labelled as boundaries."""
        return self._boundaries

    @cached_property
    def detector_records(self) -> dict[int, DetectorRecord]:
        return dict(self.graph.nodes.data(data=True))

    @property
    def adj(self) -> nx.coreviews.AdjacencyView[int, int, dict[str, Any]]:
        """
        Graph adjacency object holding the neighbours of each node.
        The returned object is a mappable that holds for each node
        a dictionary
        containing all of its neighbour nodes and the associated edge
        metadata (EdgeRecords).
        """
        return self._graph.adj

    def neighbors(self, detector: int) -> Iterator[int]:
        yield from self._graph[detector]

    def detector_is_boundary(self, detector: int) -> bool:
        """Return True if given detector is a boundary, False otherwise."""
        return self._boundaries_lookup[detector]

    def get_relevant_nodes(self, logicals: Iterable[set[DecodingEdge]]) -> set[int]:
        """Return the nodes that have an edge with a path to logical that
        is not via the boundary"""
        components = list(nx.connected_components(self.no_boundary_view))
        relevant_nodes: set[int] = set()
        for logical in logicals:
            relevant_nodes.update(chain.from_iterable(logical))
            for component in components:
                if any(a in component or b in component for a, b in logical):
                    relevant_nodes.update(component)
        return relevant_nodes

    def shortest_path(self, origin: int, destination: int) -> list[DecodingEdge]:
        """Find the shortest path between two syndrome bits, as a sequence of decoding
        edges.
        """
        # pylint: disable=too-many-function-args
        # in Python 3.10 use pairwise here
        firsts, seconds = tee(nx.shortest_path(self._graph, origin, destination))
        next(seconds)
        return [DecodingEdge(u, v) for u, v in zip(firsts, seconds, strict=False)]

    def shortest_path_length(self, origin: int, destination: int) -> float:
        """Find the length of the shortest path between two syndrome bits."""
        return nx.shortest_path_length(self._graph, origin, destination)

    def shortest_path_no_boundaries(
        self, origin: int, destination: int
    ) -> list[DecodingEdge]:
        """Find the shortest path between two syndrome bits without going via any
        boundaries, as a sequence of decoding edges.
        If origin or destination are boundaries, an exception will be raised.
        """
        # pylint: disable=too-many-function-args
        # in Python 3.10 use pairwise here
        firsts, seconds = tee(
            nx.shortest_path(self.no_boundary_view, origin, destination)
        )
        next(seconds)
        return [DecodingEdge(*edge) for edge in zip(firsts, seconds, strict=False)]

    def shortest_path_length_no_boundaries(
        self, origin: int, destination: int
    ) -> float:
        """Find the length of the shortest path between two syndrome bits without going
        via any boundaries.
        If origin or destination are boundaries, an exception will be raised.
        """
        return nx.shortest_path_length(self.no_boundary_view, origin, destination)


NXLogicals = Sequence[AbstractSet[DecodingEdge]]


@dataclass
class NXCode:
    """Convenience class encapsulating
    the graph and logicals."""

    graph: NXDecodingGraph
    logicals: NXLogicals


class NXDecodingMultiGraph(NXGraph[_QECNXMG, tuple[DecodingEdge, int]]):
    """Implementation of a decoding multigraph using NetworkX and QEC syndrome
    and edge objects. All edges in this graph are size 2, and a pair of syndromes
    can be connected by multiple edges (a multi-edge). As with the non-multiedge
    decoding graph, a fictitious boundary vertex exists to match isolated vertices.

    Parameters
    ----------
    graph : _QECNXMG
        NetworkX MultiGraph to wrap.
    boundaries : Optional[Iterable[int]], optional
        Optional indices of detectors that should be treated as a boundary, by default
        frozenset().
    """

    base_graph_class: ClassVar = _QECNXMG

    def __init__(self, graph: _QECNXMG, boundaries: Iterable[int] = frozenset()):
        super().__init__(graph, boundaries)

    @classmethod
    def from_edge_list(
        cls,
        edge_data: Iterable[
            DecodingEdge | tuple[int, int] | tuple[DecodingEdge, EdgeRecord]
        ],
        detector_records: dict[int, DetectorRecord] | None = None,
        boundaries: Iterable[int] = frozenset(),
    ) -> NXDecodingMultiGraph:
        """Create a graph, where connectivity is defined by an edge list.
        Other information about the graph is given via an optional list of nodes,
        boundaries and detector records.

        Parameters
        ----------
        edge_data : Iterable[Union[DecodingEdge, Tuple[int, int], Tuple[DecodingEdge, EdgeRecord]]],
            Collection of decoding edges to include in the graph, as an iterable of
            edges or as a Tuple of edges and their associated EdgeRecords.

        detector_records : Optional[Dict[int, DetectorRecord]], optional
            Optional specification of graph vertices given as a dictionary
            of detector index to detector record.

        edge_records : Optional[Dict[Tuple[DecodingEdge, int] | DecodingEdge, EdgeRecord]] = None, optional
            Optional specification of graph edges given as a dictionary
            of DecodingEdge to edge record.

        boundaries : Optional[Iterable[int]], optional
            Collection of nodes that are classified as a boundary vertex.

        Returns
        -------
        NXDecodingMultiGraph
            Constructed decoding multigraph.
        """  # noqa: E501
        nx_graph = cls.base_graph_class()
        detector_records = {} if detector_records is None else detector_records
        edge_records: dict[tuple[DecodingEdge, int], EdgeRecord] = {}
        nx_graph.add_nodes_from(detector_records.items())

        for data in edge_data:
            if isinstance(data, DecodingEdge):
                edge = data
                edge_record = EdgeRecord()
            elif _is_tuple_of_ints(data):
                edge = DecodingEdge(*data)
                edge_record = EdgeRecord()
            elif _is_decoding_edge_and_edge_record(data):
                edge, edge_record = data
            else:
                msg = f"Unsupported data type for edge {data}"
                raise TypeError(msg)
            u, v = edge
            k = nx_graph.number_of_edges(u, v)
            edge_records[(edge, k)] = edge_record
            nx_graph.add_edge(u, v, k, **edge_record)

        return cls(nx_graph, boundaries)

    @cached_property
    def edges(self) -> list[tuple[DecodingEdge, int]]:
        return [(DecodingEdge(u, v), edge_id) for u, v, edge_id in self.graph.edges]

    @cached_property
    def boundary_edges(self) -> set[tuple[DecodingEdge, int]]:
        return set(
            chain.from_iterable(
                self.incident_edges(boundary) for boundary in self.boundaries
            )
        )

    @cached_property
    def edge_records(self) -> dict[tuple[DecodingEdge, int], EdgeRecord]:
        return {
            (DecodingEdge(u, v), k): record
            for (u, v, k), record in self._graph.edges.items()
        }

    def get_edges(self, *detectors: int) -> Iterator[tuple[DecodingEdge, int]]:
        edge_data = self.graph.get_edge_data(*detectors)
        for edge_id in edge_data:  # type: ignore[union-attr]
            yield (DecodingEdge(*detectors), int(edge_id))

    def incident_edges(self, detector: int) -> Iterator[tuple[DecodingEdge, int]]:
        unique_detectors = {tuple(edge) for edge in self.graph.edges(detector)}
        for detectors in unique_detectors:
            yield from self.get_edges(*detectors)

    def to_parity_check_matrix(self) -> npt.NDArray[np.uint8]:
        check_matrix = np.zeros((len(self.nodes), len(self.edges)), dtype=np.uint8)
        for edge_index, (edge, _) in enumerate(self.edges):
            check_matrix[tuple(edge), (edge_index,)] = True
        return check_matrix

    def with_multi_edges_merged(
        self,
        combine_edge_records: Callable[
            [Iterable[EdgeRecord]], EdgeRecord
        ] = convolve_probabilities,
    ) -> NXDecodingGraph:
        """Create an `NXDecodingGraph` instance with the same nodes but with
        only single edges, merging the multi-edge edge records to a single edge
        record.

        Parameters
        ----------
        combine_edge_records : Callable[[ Iterable[EdgeRecord]], EdgeRecord], optional
            A function which combines all the edge records for a given edge,
            by default convolve_probabilities which applies the generalised
            `p * (1 - q) + q * (1 - p)`.

        Returns
        -------
        NXDecodingGraph
            A graph with the same properties as this graph but with all
            multi-edges combined to a single edge.
        """
        grouped_edge_records = defaultdict(list)
        for (edge, _), edge_record in self.edge_records.items():
            grouped_edge_records[edge].append(edge_record)
        new_graph = NXDecodingGraph(_QECNX(self._graph), self.boundaries)
        for edge, edge_records in grouped_edge_records.items():
            new_graph.edge_records[edge] = combine_edge_records(edge_records)
        return new_graph

    def error_to_syndrome(
        self, edges: Iterable[tuple[DecodingEdge, int]]
    ) -> OrderedSyndrome:
        return OrderedSyndrome(
            symptom
            for symptom in chain.from_iterable(edge for edge, _ in edges)
            if symptom not in self.boundaries
        )


class NXDecodingGraph(NXGraph[_QECNX, DecodingEdge]):
    """Implementation of a decoding graph using NetworkX and the QEC syndrome
    and edge objects. This works much like the hypergraph, but all edges are size 2.
    To account for boundaries, which in a hypergraph exist as size 1 edges, extra nodes
    may be added. These are virtual detectors that do not physically exist when
    running the syndrome extraction circuit.

    Parameters
    ----------
    graph : _QECNX
        NetworkX graph to wrap.
    boundaries : Optional[Iterable[int]], optional
        Optional indices of detectors that should be treated as a boundary, by default
        None.
    """

    base_graph_class: ClassVar = _QECNX

    def __init__(self, graph: _QECNX, boundaries: Iterable[int] = frozenset()):
        super().__init__(graph, boundaries)

    @classmethod
    def from_edge_list(
        cls,
        edge_data: Iterable[
            DecodingEdge | tuple[int, int] | tuple[DecodingEdge, EdgeRecord]
        ],
        detector_records: dict[int, DetectorRecord] | None = None,
        boundaries: Iterable[int] = frozenset(),
    ) -> NXDecodingGraph:
        """Create a graph, where connectivity is defined by an edge list.
        Other information about the graph is given via an optional list of nodes,
        boundaries and detector records.

        Parameters
        ----------
        edge_data : Iterable[Union[DecodingEdge, Tuple[int, int], Tuple[DecodingEdge, EdgeRecord]]],
            Collection of decoding edges to include in the graph, as an iterable of
            edges or as a Tuple of edges and their associated EdgeRecords.

        detector_records : Optional[Dict[int, DetectorRecord]], optional
            Optional specification of graph vertices given as a dictionary
            of detector index to detector record.

        boundaries : Optional[Iterable[int]], optional
            Collection of nodes that are classified as a boundary vertex.

        Returns
        -------
        NXDecodingGraph
            Constructed decoding graph.
        """  # noqa: E501
        nx_graph = cls.base_graph_class()
        detector_records = {} if detector_records is None else detector_records
        edge_records: dict[DecodingEdge, EdgeRecord] = {}
        nx_graph.add_nodes_from(detector_records.items())

        for data in edge_data:
            if isinstance(data, DecodingEdge):
                edge = data
                edge_records[edge] = EdgeRecord()
            elif isinstance(data, tuple) and isinstance(data[1], int):
                edge = DecodingEdge(*data)
                edge_records[edge] = EdgeRecord()
            elif isinstance(data, tuple) and isinstance(data[1], EdgeRecord):
                edge, edge_record = data
                edge_records[edge] = edge_record
            else:
                msg = f"Unsupported data type for edge {data}"
                raise TypeError(msg)
            nx_graph.add_edge(*edge, **edge_records[edge])

        return cls(nx_graph, boundaries)

    def to_decoding_hypergraph(self) -> DecodingHyperGraph:
        """Elevate decoding graph to hypergraph."""
        edge_data = [
            (DecodingHyperEdge(set(edge) - self.boundaries), self.edge_records[edge])
            for edge in self.edges
        ]
        det_records_copy = {
            k: v for k, v in self.detector_records.items() if k not in self.boundaries
        }
        return DecodingHyperGraph(edge_data, detector_records=det_records_copy)

    def get_edges(self, *detectors: int) -> list[DecodingEdge]:
        return [edge] if (edge := DecodingEdge(*detectors)) in self.edges else []

    def get_edge(self, *detectors: int) -> DecodingEdge:
        """Returns the unique edge in the graph corresponding to the given set
        of detector indices.

        Raises
        ------
        ValueError
            If no edge in the graph corresponds to the given detectors.
        """
        if (edge := DecodingEdge(*detectors)) in self.edge_records:
            return edge
        msg = f"Detectors {detectors} do not belong to any edge in the graph."
        raise ValueError(msg)

    def get_edge_record(self, *detectors: int) -> EdgeRecord:
        """Given a set of detectors that define an edge in the graph,
        return its correspoiding `EdgeRecord`.
        """
        return self.edge_records[DecodingEdge(*detectors)]

    @cached_property
    def edges(self) -> list[DecodingEdge]:
        return [DecodingEdge(u, v) for u, v in self._graph.edges]

    @cached_property
    def boundary_edges(self) -> set[DecodingEdge]:
        return set(
            chain.from_iterable(
                self.incident_edges(boundary) for boundary in self.boundaries
            )
        )

    @cached_property
    def edge_records(self) -> dict[DecodingEdge, EdgeRecord]:
        return {
            DecodingEdge(u, v): record for (u, v), record in self._graph.edges.items()
        }

    def incident_edges(self, detector: int) -> Iterator[DecodingEdge]:
        for edge in self._graph.edges(detector):
            yield DecodingEdge(*edge)

    def to_parity_check_matrix(self) -> npt.NDArray[np.uint8]:
        check_matrix = np.zeros((len(self.nodes), len(self.edges)), dtype=np.uint8)
        for edge_index, edge in enumerate(self.edges):
            check_matrix[tuple(edge), (edge_index,)] = True
        return check_matrix

    def error_to_syndrome(self, edges: Iterable[DecodingEdge]) -> OrderedSyndrome:
        return OrderedSyndrome(
            symptom
            for symptom in chain.from_iterable(edges)
            if symptom not in self.boundaries
        )
