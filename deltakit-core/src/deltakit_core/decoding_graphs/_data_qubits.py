# (c) Copyright Riverlane 2020-2025.
"""Datastructures for data qubits."""

from __future__ import annotations

import math
from collections import Counter, UserDict
from collections.abc import (
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from collections.abc import (
    Set as AbstractSet,
)
from functools import cached_property
from itertools import chain
from typing import Any, Generic, TypeVar

from typing_extensions import overload

from deltakit_core.decoding_graphs._syndromes import (
    Bit,
    DetectorRecord,
    OrderedSyndrome,
)


class EdgeRecord(UserDict[str, Any]):
    """Dictionary for recording information about an edge.
    String attributes for arbitrary values. Error probability
    given as special data that is always defined.

    Parameters
    ----------
    p_err : float, optional
        Probability of the error mechanism, by default 0.0.
    """

    def __init__(self, p_err: float = 0.0, **kwargs: Any):
        super().__init__(p_err=p_err, **kwargs)
        self.data["weight"] = self.weight

    @property
    def weight(self) -> float:
        """Log-likelihood weight of the error mechanism."""
        if self.p_err == 0:
            return math.inf
        if self.p_err == 1:
            return -math.inf
        if not 0 < self.p_err < 1:
            msg = f"Edge weight undefined for error probability {self.p_err}"
            raise ValueError(msg)
        return math.log((1 - self.p_err) / self.p_err)

    @property
    def p_err(self) -> float:
        """Probability of the error mechanism occurring."""
        return self.data["p_err"]

    @p_err.setter
    def p_err(self, value: float):
        self.data["p_err"] = value
        self.data["weight"] = self.weight

    @classmethod
    def from_dict(cls, property_dict: dict[str, Any]) -> EdgeRecord:
        """Create a EdgeRecord from a given property dict of optional values.

        Parameters
        ----------
        property_dict : Dict[str, Any]
            Any optional properties such as p_err.

        Returns
        -------
        EdgeRecord
        """
        record = EdgeRecord(p_err=property_dict.get("p_err", 0.0))
        record.update(property_dict)
        return record

    @classmethod
    def from_loglikelihood(cls, weight: float) -> EdgeRecord:
        """Create a EdgeRecord from a loglikelihood weight rather than a p_err.

        Parameters
        ----------
        weight: float
            Edge weight as a loglikelihood, defined as w=ln((1-p)/p)
            where w is the weight, p is error probability and
            ln is the natural log.

        Returns
        -------
        EdgeRecord
        """
        return EdgeRecord(p_err=1 / (1 + math.exp(weight)))


class DecodingHyperEdge(Collection[int]):
    """Representation of an immutable decoding edge on a hyper-graph.
    This is a set of syndrome bits.

    Parameters
    ----------
    vertices : Iterable[int]
        Syndrome bits connected by this edge.
    """

    def __init__(self, vertices: Iterable[int]):
        self._vertices = frozenset(vertices)
        self._hash = hash(self.vertices)

    @property
    def vertices(self) -> frozenset[int]:
        """Vertices in this edge."""
        return self._vertices

    def to_decoding_edge(self, boundary: int | None = None) -> DecodingEdge:
        """Cast this edge into a decoding edge if possible."""
        degree_target = 2
        if (degree := len(self.vertices)) == degree_target:
            return DecodingEdge(*self.vertices)
        if degree == 1:
            if boundary is None:
                msg = f"Boundary vertex is required for edge: {self} of degree one."
                raise ValueError(msg)
            return DecodingEdge(next(iter(self.vertices)), boundary)
        msg = f"Cannot cast edge: {self} of degree {degree} to decoding edge."
        raise ValueError(msg)

    def __repr__(self) -> str:
        return str(tuple(self.vertices))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DecodingHyperEdge)
            and self._hash == other._hash
            and self.vertices == other.vertices
        )

    def __iter__(self) -> Iterator[int]:
        return self.vertices.__iter__()

    def __len__(self) -> int:
        return len(self.vertices)

    def __contains__(self, x: object) -> bool:
        return x in self.vertices


class DecodingEdge(DecodingHyperEdge):
    """Representation of an immutable decoding edge on a decoding graph, limited to
    standard graphs where edges connect two distinct detectors.

    Parameters
    ----------
    first_detector : int
        First detector this edge is incident to.
    second_detector : int
        Second detector this edge is incident to.
    """

    def __init__(self, first_detector: int, second_detector: int):
        if first_detector == second_detector:
            msg = (
                f"Invalid DecodingEdge between detectors {first_detector} "
                f"and {second_detector}."
            )
            raise ValueError(msg)
        super().__init__((first_detector, second_detector))

    @property
    def first(self) -> int:
        """Return the first vertex in this edge, order is arbitrary."""
        return next(iter(self._vertices))

    @property
    def second(self) -> int:
        """Return the second vertex in this edge, order is arbitrary."""
        return tuple(self._vertices)[1]

    def is_timelike(self, detector_records: Mapping[int, DetectorRecord]) -> bool:
        """Return True if this decoding edge is between the same data qubit at different
        time steps.
        """
        return (
            detector_records[self.first].time != detector_records[self.second].time
        ) and (
            detector_records[self.first].spatial_coord
            == detector_records[self.second].spatial_coord
        )

    def is_spacelike(self, detector_records: Mapping[int, DetectorRecord]) -> bool:
        """Return True if this decoding edge is between the same data qubit at different
        time steps.
        """
        return (
            detector_records[self.first].time == detector_records[self.second].time
        ) and (
            detector_records[self.first].spatial_coord
            != detector_records[self.second].spatial_coord
        )

    def is_hooklike(self, detector_records: Mapping[int, DetectorRecord]) -> bool:
        """Return True if this decoding edge is between the same data qubit at different
        time steps.
        """
        return (
            detector_records[self.first].time != detector_records[self.second].time
        ) and (
            detector_records[self.first].spatial_coord
            != detector_records[self.second].spatial_coord
        )


EdgeT = TypeVar("EdgeT", bound=DecodingHyperEdge)


class OrderedDecodingEdges(Sequence[EdgeT], AbstractSet[EdgeT], Generic[EdgeT]):
    """Immutable ordered mod 2 set of decoding edges.

    All decoding edges are thought to be on the same decoding graph.
    I.e. if the z and x decoding graphs are completely separable, then the z and x
    errors will be represented in two different `OrderedDecodingEdges` objects.

    Parameters
    ----------
    decoding_edges : Optional[Iterable[EdgeT]], optional
        Iterable of edges to store, in the given order, by default None. Duplicate
        items are removed if they are present an even number of times. Duplicate items
        are reduced to a single occurrence if they are present an odd number of times.
    mod_2_filter : bool, optional
        False disables the modulo 2 filter, in which case the input
        edges are treated as an ordered set, by default True.
    """

    def __init__(
        self,
        decoding_edges: Iterable[EdgeT] | None = None,
        mod_2_filter: bool = True,
    ):
        _decoding_edges = [] if decoding_edges is None else decoding_edges
        if mod_2_filter:
            self._decoding_edges = self._mod_2_filter(_decoding_edges)
        else:
            self._decoding_edges = dict.fromkeys(_decoding_edges)

    @staticmethod
    def _mod_2_filter(decoding_edges: Iterable[EdgeT]) -> dict[EdgeT, None]:
        edge_counts = Counter(decoding_edges)
        return dict.fromkeys(
            [edge for edge, count in edge_counts.items() if count % 2 == 1]
        )

    def append(
        self, other: OrderedDecodingEdges[EdgeT], mod_2_filter: bool = False
    ) -> None:
        """Appeds edges from another set of OrderedDecodingEdges."""
        # pylint: disable=protected-access
        if mod_2_filter:
            self._decoding_edges = self._mod_2_filter(chain(self, other))
        else:
            self._decoding_edges.update(other._decoding_edges)

    @cached_property
    def _as_tuple(self) -> tuple[EdgeT, ...]:
        """Defined to create immutable object to hash, and to make `__getitem__` a O(1)
        method. Exists as a property to avoid duplication of data in core member
        attributes, and to have this be created only when needed.
        """
        return tuple(self._decoding_edges)

    def __add__(
        self, other: OrderedDecodingEdges[EdgeT]
    ) -> OrderedDecodingEdges[EdgeT]:
        result = OrderedDecodingEdges(self._decoding_edges.keys(), mod_2_filter=False)
        result.append(other)
        return result

    def __len__(self) -> int:
        return len(self._decoding_edges)

    @overload
    def __getitem__(self, index: int) -> EdgeT: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[EdgeT, ...]: ...

    def __getitem__(self, index: int | slice) -> EdgeT | tuple[EdgeT, ...]:
        return self._as_tuple[index]

    def __contains__(self, x: object) -> bool:
        return self._decoding_edges.__contains__(x)

    def __iter__(self) -> Iterator[EdgeT]:
        return self._decoding_edges.__iter__()

    def __repr__(self) -> str:
        return f"[{', '.join(map(str, self))}]"

    def __hash__(self) -> int:
        return hash(self._as_tuple)

    def __eq__(self, other: object) -> bool:
        # Perhaps allow other sequences to be comparable also?
        return (
            isinstance(other, OrderedDecodingEdges)
            and self._as_tuple == other._as_tuple
        )

    def as_bitstring(self, edges: Sequence[EdgeT]) -> list[Bit]:
        """Convert given edges to a bitstring representation of `len(edges)` bits
        from condition of each edge inside  _decoding_edges.

        Parameters
        ----------
        edges : Sequence[EdgeT]
            Edges to be converted to bitstring .

        Returns
        -------
        List[Bit]
            Bitstring of whether each edge is in `_decoding_edges`.
        """
        return [1 if edge in self else 0 for edge in edges]

    @classmethod
    def from_syndrome_indices(
        cls, indices: Iterable[tuple[int, int]]
    ) -> OrderedDecodingEdges[DecodingEdge]:
        """Given a list of pairs of syndrome indices, construct the corresponding
        decoding edges and return in an `OrderedDecodingEdges` collection.

        Parameters
        ----------
        indices : Iterable[Tuple[int, int]]
            Indices into the syndrome sequence.

        Returns
        -------
        OrderedDecodingEdges
        """
        return OrderedDecodingEdges(DecodingEdge(u, v) for u, v in indices)


def errors_to_syndrome(
    decoding_edges: Iterable[EdgeT], boundaries: AbstractSet[int] = frozenset()
) -> OrderedSyndrome:
    """Take decoding edges and convert to an ordered syndrome that would be observed
    if all decoding edges were to be active.

    Parameters
    ----------
    decoding_edges : Iterable[EdgeT]
        Decoding edges to convert.
    boundaries : AbstractSet[int], optional
        Set of detectors that are virtual boundary detectors and should hence not show
        in a syndrome. By default None.

    Returns
    -------
    OrderedSyndrome
        Resultant ordered syndrome.
    """
    return OrderedSyndrome(
        symptom
        for symptom in chain.from_iterable(decoding_edges)
        if symptom not in boundaries
    )
