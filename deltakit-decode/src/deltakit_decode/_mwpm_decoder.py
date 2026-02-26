# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property

import networkx as nx
import numpy as np
import pymatching
import stim
from deltakit_circuit import Circuit
from deltakit_core.decoding_graphs import (
    DecodingHyperEdge,
    OrderedDecodingEdges,
    OrderedSyndrome,
)

from deltakit_decode._abstract_matching_decoders import GraphDecoder
from deltakit_decode.utils._graph_circuit_helpers import parse_stim_circuit


class PyMatchingDecoder(GraphDecoder):
    """PyMatching decoder for minimum weight perfect matching (MWPM),
    configured to use our decoding graph representation.

    Parameters
    ----------
    decoding_graph : NXDecodingGraph
        Decoding graph to define the matching problem.
    logicals : Tuple[OrderedDecodingEdges, ...]
        Tuple of reference logicals to keep track of, as a collection of
        decoding edges.

    Notes
    -----
        MWPM relies on contiguous indices for nodes.
    """

    name = "PyMatching2"

    def _make_matcher(
        self, fault_edges: Iterable[Iterable[DecodingHyperEdge]]
    ) -> pymatching.Matching:
        local_nx: nx.Graph = self.decoding_graph.graph.copy()
        for fault_id, edges in enumerate(fault_edges):
            for edge in edges:
                if "fault_ids" in local_nx.edges[edge]:
                    local_nx.edges[edge]["fault_ids"].add(fault_id)
                else:
                    local_nx.edges[edge]["fault_ids"] = {fault_id}
        for boundary in self.decoding_graph.boundaries:
            local_nx.nodes[boundary]["is_boundary"] = True
        return pymatching.Matching.from_networkx(
            local_nx, min_num_fault_ids=len(self.logicals)
        )

    def __str__(self) -> str:
        return "MWPM"

    @cached_property
    def _logical_flip_matcher(self) -> pymatching.Matching:
        return self._make_matcher(self.logicals)

    @cached_property
    def _full_matcher(self) -> pymatching.Matching:
        return self._make_matcher([[edge] for edge in self.decoding_graph.edges])

    def decode_to_logical_flip(self, syndrome: OrderedSyndrome) -> tuple[bool, ...]:
        py_matching_syndrome = syndrome.as_bitstring(max(self.decoding_graph.nodes) + 1)
        corrections = self._logical_flip_matcher.decode(py_matching_syndrome)
        return tuple(bool(corr) for corr in corrections)

    def decode_to_full_correction(
            self, syndrome: OrderedSyndrome) -> OrderedDecodingEdges:

        py_matching_syndrome = syndrome.as_bitstring(
            max(self.decoding_graph.nodes) + 1)
        corrections = self._full_matcher.decode(py_matching_syndrome)

        return OrderedDecodingEdges(
            [self.decoding_graph.edges[i] for i in np.nonzero(corrections)[0]]
        )

    def decode_batch_to_full_correction(
        self, syndrome_batch: np.ndarray
    ) -> np.ndarray:
        return self._full_matcher.decode_batch(syndrome_batch)

    def decode_batch_to_logical_flip(
        self, syndrome_batch: np.ndarray
    ) -> np.ndarray:
        return self._logical_flip_matcher.decode_batch(syndrome_batch)

    @classmethod
    def construct_decoder_and_stim_circuit(
        cls, circuit: Circuit
    ) -> tuple[PyMatchingDecoder, stim.Circuit]:
        """Helper factory to create a MWPM decoder and the Stim circuit used
        during its construction.

        Parameters
        ----------
        circuit : Circuit
            Circuit to use to construct the decoder.

        Returns
        -------
        Tuple[PyMatchingDecoder, stim.Circuit]
        """
        stim_circuit = circuit.as_stim_circuit()
        graph, logicals, stim_circuit = parse_stim_circuit(stim_circuit)
        return cls(graph, logicals), stim_circuit

    def __getstate__(self):
        inner_state = self.__dict__.copy()
        if "_logical_flip_matcher" in inner_state:
            del inner_state["_logical_flip_matcher"]
        if "_full_matcher" in inner_state:
            del inner_state["_full_matcher"]
        return inner_state
