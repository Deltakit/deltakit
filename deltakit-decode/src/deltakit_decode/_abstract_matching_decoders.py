# (c) Copyright Riverlane 2020-2025.
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from functools import cached_property
from itertools import chain, islice
from typing import Generic, Protocol, TypeAlias, TypeVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DecodingHyperEdge,
    HyperLogicals,
    HyperMultiGraph,
    NXDecodingGraph,
    OrderedDecodingEdges,
    OrderedSyndrome,
)

from deltakit_decode.utils import make_logger

Matching: TypeAlias = list[tuple[int, int]]

GraphT = TypeVar("GraphT", bound=HyperMultiGraph)


class DecoderProtocol(Protocol):
    """Protocol class for decoders that can be implemented explicitly using inheritance
    or implicitly using duck typing. Combine with other protocols to create intersection
    types to aid the type checker.
    """

    @abstractmethod
    def decode_to_full_correction(self, syndrome: OrderedSyndrome
                                  ) -> AbstractSet[DecodingHyperEdge]:
        """Decode a given syndrome and return the full correction as a collection of
        decoding edges. Edges are un-projected, i.e. the same physical qubit can be
        referenced multiple times at different time-steps.

        Parameters
        ----------
        syndrome : OrderedSyndrome
            Syndrome to decode, as a collection of syndrome bits.

        Returns
        -------
        OrderedDecodingEdges
            Full correction as a collection of decoding edges.
        """

    @abstractmethod
    def decode_to_logical_flip(self, syndrome: OrderedSyndrome) -> tuple[bool, ...]:
        """Decode a given syndrome and return a flip boolean for each logical.

        Parameters
        ----------
        syndrome : OrderedSyndrome
            Syndrome to decode, as a collection of syndrome bits.

        Returns
        -------
        Tuple[bool, ...]
            List of logical flips as booleans. True if the homology class is 1 (flipped),
            False if the homology class is 0 (not flipped).
        """

    @abstractmethod
    def decode_batch_to_full_correction(
        self, syndrome_batch: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Decodes a batch of syndrome bitstrings to full correction.

        Parameters
        ----------
        syndrome_batch : np.ndarray
            Syndrome  to decode. 2D Array of shape
            (number of shots, number of syndromes). Each element is a 1
            or 0.


        Returns
        -------
        np.ndarray
            2D Array indicating full corrections of shape
            (number of shots, number of edges). Each element is a 1
            or 0.
        """

    @abstractmethod
    def decode_batch_to_logical_flip(
        self, syndrome_batch: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Decodes a batch of syndrome bitstrings to logical flips.

        Parameters
        ----------
        syndrome_batch : np.ndarray
            Syndrome  to decode. 2D Array of shape
            (number of shots, number of syndromes). Each element is a 1
            or 0.


        Returns
        -------
        np.ndarray
            2D Array indicating logical flips of shape
            (number of shots, number of logicals). Each element is a 1
            or 0.
        """


class GraphDecoder(DecoderProtocol, ABC, Generic[GraphT]):
    """Abstract class for decoders that work given an input decoding graph, by solving
    a problem similar to matching.

    Parameters
    ----------
    decoding_graph : GraphT
        Decoding graph to decode over.
    logicals : Sequence[AbstractSet[DecodingHyperEdge]]
        Sequence of reference logicals to keep track of, represented by collections of
        un-projected decoding edges.
    lvl : int, optional
        Logging level to use, by default logging.ERROR.
    """

    def __init__(self,
                 decoding_graph: GraphT,
                 logicals: HyperLogicals,
                 lvl: int = logging.ERROR):
        if not logicals:
            warnings.warn("No logicals are given.", stacklevel=2)
        if not all(logicals):
            warnings.warn("A logical was given with no activators.", stacklevel=2)
        if not all(logical_edge in decoding_graph.edges
                   for logical_edge in chain.from_iterable(logicals)):
            msg = (
                f"Logicals {logicals} are not entirely within {decoding_graph.edges}."
            )
            raise ValueError(msg)

        self.decoding_graph = decoding_graph
        self.logicals = logicals
        self.log = make_logger(lvl, self.__class__.__name__)

    def decode_to_logical_flip(self, syndrome: OrderedSyndrome) -> tuple[bool, ...]:
        result = self.decode_to_full_correction(syndrome)
        return tuple(len(result & logical) % 2 == 1 for logical in self.logicals)

    @cached_property
    def logicals_edge_list(self) -> list[list[int]]:
        "Get list of edges for each logical"
        return [
            [idx for idx, edge in enumerate(self.decoding_graph.edges) if edge in log]
            for log in self.logicals
        ]

    @abstractmethod
    def decode_to_full_correction(
        self, syndrome: OrderedSyndrome
    ) -> OrderedDecodingEdges:
        """Decode to full correction returning OrderedDecodingEdges object"""

    def decode_batch_to_full_correction(
        self, syndrome_batch: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        return np.asarray(
            [
                np.asarray(
                    self.decode_to_full_correction(
                        OrderedSyndrome.from_bitstring(s)
                    ).as_bitstring(self.decoding_graph.edges),
                    dtype=np.uint8,
                )
                for s in syndrome_batch
            ]
        )

    def decode_batch_to_logical_flip(
        self, syndrome_batch: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        return np.asarray(
            [
                np.asarray(
                    self.decode_to_logical_flip(OrderedSyndrome.from_bitstring(s)),
                    dtype=np.uint8,
                )
                for s in syndrome_batch
            ]
        )


class MatchingDecoder(GraphDecoder[NXDecodingGraph]):
    """Abstract class to represent matching decoders. These are decoders that define
    `decode_to_matching`, which takes the syndrome and returns a matching. A matching is
    a list of pairs of syndrome bits. The full correction is then a mod 2 combination of
    all the paths between each pair in the matching.
    """

    @abstractmethod
    def decode_to_matching(self, syndrome: OrderedSyndrome
                           ) -> Matching:
        """Decode a given syndrome and return a matching.

        Parameters
        ----------
        syndrome : OrderedSyndrome
            Syndrome to decode, as a collection of syndrome bits.

        Returns
        -------
        Matching
            List of pairs of syndrome bits, as a matching.
        """

    def decode_to_full_correction(self, syndrome: OrderedSyndrome
                                  ) -> OrderedDecodingEdges:
        matching = self.decode_to_matching(syndrome)
        correction_paths: list[Sequence[DecodingEdge]] = []
        for origin, destination in matching:
            if (self.decoding_graph.detector_is_boundary(origin) or
                    self.decoding_graph.detector_is_boundary(destination)):
                correction_paths.append(
                    self.decoding_graph.shortest_path(origin, destination))
            else:
                correction_paths.append(
                    self.decoding_graph.shortest_path_no_boundaries(origin, destination))

        return OrderedDecodingEdges(chain.from_iterable(correction_paths))


class ClusteringDecoder(MatchingDecoder):
    """Abstract class to represent clustering decoders. These are decoders that define
    `decode_to_clustering`, which can in turn be used to define a matching by randomly
    pairing syndromes from the same cluster.

    Additionally, when the decoding graph is cluster homology accelerated, the clustering
    can be used to give a logical flip without going via the full correction.
    """

    @abstractmethod
    def decode_to_clustering(self, syndrome: OrderedSyndrome
                             ) -> set[frozenset[int]]:
        """Decode a given syndrome and return a clustering.

        Parameters
        ----------
        syndrome : OrderedSyndrome
            Syndrome to decode, as a collection of syndrome bits.

        Returns
        -------
        Set[OrderedSyndrome]
            The clustering, given as a set of syndrome sets.
        """

    @cached_property
    def trivial_match_enabled(self) -> bool:
        """Return True if the graph and logicals define a scenario where matching is
        trivial. This is when for each edge in a logical (a,b), there is no path from
        a to b that does not go via at least one edge in the same logical. If such a
        path does exist, then matching is not trivial and False is returned.

        Cached to avoid recomputation.
        """
        if not isinstance(self.decoding_graph, NXDecodingGraph):
            return False

        for logical in self.logicals:
            simple_form_logical = {tuple(sorted((a, b))) for a, b in logical}
            no_logical_edges_view = nx.subgraph_view(
                self.decoding_graph.graph,
                filter_edge=lambda a, b, _logical=simple_form_logical:
                tuple(sorted((a, b))) not in _logical)
            if any(nx.has_path(no_logical_edges_view, a, b) for a, b in logical):
                return False

        return True

    def decode_to_matching(self, syndrome: OrderedSyndrome
                           ) -> Matching:
        if self.trivial_match_enabled:
            # Get clustering then arbitrarily pair
            clustering = self.decode_to_clustering(syndrome)
            matching: Matching = []
            for cluster in clustering:
                evens = islice(cluster, 0, len(cluster), 2)
                odds = islice(cluster, 1, len(cluster), 2)
                matching.extend(zip(evens, odds))
            return matching
        raise NotImplementedError()
