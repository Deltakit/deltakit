# (c) Copyright Riverlane 2020-2025.
"""Module for parsing detector error models."""

from __future__ import annotations

import collections.abc
import warnings
from collections import Counter
from itertools import chain, zip_longest
from typing import (
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    cast,
)

import stim

from deltakit_core.decoding_graphs._data_qubits import (
    DecodingEdge,
    DecodingHyperEdge,
    EdgeRecord,
)
from deltakit_core.decoding_graphs._decoding_graph import (
    _QECNX,
    DecodingHyperGraph,
    NXDecodingGraph,
)
from deltakit_core.decoding_graphs._syndromes import DetectorRecord


class CoordinateOffset(tuple):  # noqa: PLW1641
    """Class to track the coordinate offset in a Detector Error Model."""

    def __new__(cls, offset: Iterable[float | int] = ()):
        return super().__new__(cls, offset)

    def __add__(self, other: object) -> CoordinateOffset:
        if isinstance(other, collections.abc.Iterable):
            return CoordinateOffset(
                self_offset + other_offset
                for self_offset, other_offset in zip_longest(self, other, fillvalue=0)
            )
        return NotImplemented

    def __radd__(self, other: object) -> CoordinateOffset:
        return self.__add__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CoordinateOffset):
            return super().__eq__(other)
        return NotImplemented


class ErrorHandler(Protocol):
    """General callable protocol used in the DEM parser to handler stim's
    error instructions."""

    def __call__(self, error: stim.DemInstruction, detector_offset: int) -> None: ...


class DetectorHandler(Protocol):
    """General callable protocol used in the DEM parser to handle stim's
    detector instructions."""

    def __call__(
        self,
        detector: stim.DemInstruction,
        detector_offset: int,
        coordinate_offset: CoordinateOffset,
    ) -> None: ...


class ObservableHandler(Protocol):
    """General callable protocol used in the DEM parser to handler stim's
    logical_observable instructions."""

    def __call__(self, logical_observable: stim.DemInstruction) -> None: ...


class DetectorRecorder(DetectorHandler):
    """A type of detector handler which populates a dictionary of
    DetectorRecords."""

    def __init__(self) -> None:
        self._detector_records: Dict[int, DetectorRecord] = {}

    @property
    def detector_records(self) -> Dict[int, DetectorRecord]:
        """Get the detector records for this class."""
        return self._detector_records

    def __call__(
        self,
        detector: stim.DemInstruction,
        detector_offset: int,
        coordinate_offset: CoordinateOffset,
    ):
        detector_index = (
            detector_offset + cast(List[stim.DemTarget], detector.targets_copy())[0].val
        )
        detector_coordinate = coordinate_offset + detector.args_copy()
        self._detector_records[detector_index] = DetectorRecord.from_sequence(
            detector_coordinate
        )


def observable_warning(logical_observable: stim.DemInstruction) -> None:
    """Type of logical observable handler which raises a warning when this
    instruction is encountered."""
    logicals = " ".join(
        f"L{target.val}"
        for target in cast(List[stim.DemTarget], logical_observable.targets_copy())
    )
    warnings.warn(
        f"Isolated logical observables {logicals} declared in DEM file.",
        UserWarning,
        stacklevel=2,
    )


def collect_edges(
    error: stim.DemInstruction, detector_offset: int
) -> Iterator[Tuple[Set[int], float, List[int]]]:
    """Iterate through the DEM targets in the error and yield all the
    detectors which make up the edge, the probability of an error on this edge
    and the logical observables which would be affected by an error on the
    edge.

    Edges are yielded when the target is a target separator.

    Parameters
    ----------
    error : stim.DemInstruction
        A DemInstruction with type as "error"
    detector_offset : int
        The offset to add to all detectors in the edge

    Yields
    ------
    Iterator[Tuple[Set[int], float, int | None]]
        Set[int]:
            These are the detectors which are in the edge
        float:
            This is the probability of an error on this edge
        List[int]:
            List of which logicals are affected by errors on this edge. If no
            logical is affected this will be an empty list.
    """
    p_err = error.args_copy()[0]
    vertices_in_edge: Set[int] = set()
    logicals_affected: List[int] = []
    for target in chain(
        cast(List[stim.DemTarget], error.targets_copy()), (stim.target_separator(),)
    ):
        if target.is_separator():
            yield vertices_in_edge, p_err, logicals_affected
            vertices_in_edge = set()
            logicals_affected = []
        elif target.is_relative_detector_id():
            vertices_in_edge.add(detector_offset + target.val)
        elif target.is_logical_observable_id():
            logicals_affected.append(target.val)


class LogicalsInEdges(ErrorHandler):
    """A type of edge handler which collects the errors into a set of edges
    and edges which affect the logicals."""

    def __init__(self, num_logicals: int):
        self._edges: Set[DecodingHyperEdge] = set()
        self._edge_records: Dict[DecodingHyperEdge, EdgeRecord] = {}
        self._logicals: List[Set[DecodingHyperEdge]] = [
            set() for _ in range(num_logicals)
        ]

    @property
    def edges(self) -> Set[DecodingHyperEdge]:
        """Gets all edges from a detector error model."""
        return self._edges

    @property
    def logicals(self) -> List[Set[DecodingHyperEdge]]:
        """Get the edges which affect each logical of a detector error
        model."""
        return self._logicals

    def __call__(self, error: stim.DemInstruction, detector_offset: int):
        for vertices, p_err, logicals in collect_edges(error, detector_offset):
            edge = DecodingHyperEdge(vertices)
            self._edge_records[edge] = EdgeRecord(p_err=p_err)
            self._edges.add(edge)
            for logical in logicals:
                self._logicals[logical].add(edge)


EH = TypeVar("EH", bound=ErrorHandler)
DH = TypeVar("DH", bound=DetectorHandler)


class DemParser(Generic[EH, DH]):
    """Class which handles parsing a detector error model. This class tracks
    the detector and coordinate offsets as member data while parsing and passes
    this information into the relevant handlers.

    The error handler is a function which takes a "error" type DEM instruction
    and the current detector offset and tracks the state of the errors.

    The detector handler is a function which takes a "detector" type DEM
    instruction, the current detector offset and the coordinate offset and
    tracks the detectors.
    """

    def __init__(
        self,
        error_handler: EH,
        detector_handler: DH,
        observable_handler: ObservableHandler = observable_warning,
    ):
        self._error_handler = error_handler
        self._detector_handler = detector_handler
        self._observable_handler = observable_handler
        self._detector_offset = 0
        self._coordinate_offset = CoordinateOffset()

    @property
    def error_handler(self) -> EH:
        """Get the error handler for this DEM parser."""
        return self._error_handler

    @property
    def detector_handler(self) -> DH:
        """Get the detector handler for this DEM parser."""
        return self._detector_handler

    @property
    def observable_handler(self) -> ObservableHandler:
        """Get the logical observable handler for this DEM parser."""
        return self._observable_handler

    def parse(self, detector_error_model: stim.DetectorErrorModel):
        """Parse a detector error model using the given error and detector
        handlers. The class keeps track of the detector and coordinate offsets.

        Parameters
        ----------
        detector_error_model : stim.DetectorErrorModel
            A stim detector error model.
        """
        for instruction in detector_error_model:
            if isinstance(instruction, stim.DemRepeatBlock):
                for _ in range(instruction.repeat_count):
                    self.parse(instruction.body_copy())
            elif isinstance(instruction, stim.DemInstruction):
                if instruction.type == "error":
                    self._error_handler(instruction, self._detector_offset)
                elif instruction.type == "shift_detectors":
                    self._detector_offset += cast(int, instruction.targets_copy()[0])
                    self._coordinate_offset += instruction.args_copy()
                elif instruction.type == "detector":
                    self._detector_handler(
                        instruction, self._detector_offset, self._coordinate_offset
                    )
                elif instruction.type == "logical_observable":
                    self._observable_handler(instruction)


def dem_to_hypergraph_and_logicals(
    dem: stim.DetectorErrorModel,
) -> Tuple[DecodingHyperGraph, List[Set[DecodingHyperEdge]]]:
    """Convert a Stim detector error model into a DecodingHyperGraph and a
    list of edges which affect the logical observable at the index in the list.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Stim detector error model to convert.

    Returns
    -------
    Tuple[DecodingHyperGraph, List[Set[DecodingHyperEdge]]]
        DecodingHyperGraph created from detector error model and a list of
        edges which affect the logical.
    """
    edge_records: Dict[DecodingHyperEdge, EdgeRecord] = {}
    logicals_affected: List[Set[DecodingHyperEdge]] = [
        set() for _ in range(dem.num_observables)
    ]

    def error_handler_no_multiedges(error: stim.DemInstruction, detector_offset: int):
        """Type of error handler for constructing decoding hypergraph and
        logicals hyperedges which are not multiedges from error instructions."""
        for vertices, p_err, logicals in collect_edges(error, detector_offset):
            hyperedge = DecodingHyperEdge(vertices)
            old_hyperedge_record = edge_records.get(hyperedge)
            if old_hyperedge_record is not None:
                p_err = p_err * (  # noqa: PLW2901
                    1 - old_hyperedge_record.p_err
                ) + old_hyperedge_record.p_err * (1 - p_err)
                old_hyperedge_record.p_err = p_err
            edge_records[hyperedge] = EdgeRecord(p_err=p_err)
            for logical_index in logicals:
                logicals_affected[logical_index].add(hyperedge)

    parser = DemParser(
        error_handler_no_multiedges, DetectorRecorder(), observable_warning
    )
    parser.parse(dem)

    return (
        DecodingHyperGraph(
            list(edge_records.items()),
            parser.detector_handler.detector_records,
        ),
        logicals_affected,
    )


def dem_to_decoding_graph_and_logicals(
    dem: stim.DetectorErrorModel,
) -> Tuple[NXDecodingGraph, List[Set[DecodingEdge]]]:
    """Convert a detector error model into a NXDecodingGraph object and a list
    of edges which effect particular logicals.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The stim Detector Error Model to convert into a graph. The edges
        should already be decomposed into edges of max degree two.

    Returns
    -------
    Tuple[NXDecodingGraph, List[Set[DecodingEdge]]]
        A decoding graph and a list of edges which effect each logical.

    Raises
    ------
    ValueError
        If while parsing the DEM it encounters an edge of degree greater than
        two.
    """
    boundary = dem.num_detectors
    graph = _QECNX()
    graph.add_node(boundary, **DetectorRecord((-1, -1), 0))
    logicals_affected: List[Set[DecodingEdge]] = [
        set() for _ in range(dem.num_observables)
    ]

    def error_handler(error: stim.DemInstruction, detector_offset: int):
        """Type of error handler for constructing decoding graph and
        logical edges from error instructions."""
        degree_target = 2
        for vertices, p_err, logicals in collect_edges(error, detector_offset):
            if (degree := len(vertices)) == 0:
                warnings.warn(
                    "Degree 0 edge has been skipped over in graph creation.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if degree > degree_target:
                raise ValueError(
                    f"Edge of degree {degree} cannot be converted to decoding edge."
                )

            u, v = (next(iter(vertices)), boundary) if degree == 1 else tuple(vertices)
            old_edge_data = (
                edge_data
                if (edge_data := graph.get_edge_data(u, v)) is not None
                else None
            )
            if old_edge_data is not None:
                old_p_err = old_edge_data["p_err"]
                p_err = p_err * (1 - old_p_err) + old_p_err * (1 - p_err)  # noqa: PLW2901

            edge = DecodingEdge(u, v)
            edge_record = EdgeRecord(p_err=p_err)
            graph.add_edge(u, v, **edge_record)
            for logical_index in logicals:
                logicals_affected[logical_index].add(edge)

    def detector_handler(
        detector: stim.DemInstruction,
        detector_offset: int,
        coordinate_offset: CoordinateOffset,
    ):
        """Type of detector handler for adding coordinate annotated detectors
        to the decoding graph."""
        detector_index = (
            detector_offset + cast(List[stim.DemTarget], detector.targets_copy())[0].val
        )
        detector_coordinate = coordinate_offset + detector.args_copy()
        graph.add_node(
            detector_index, **DetectorRecord.from_sequence(detector_coordinate)
        )

    parser = DemParser(error_handler, detector_handler, observable_warning)
    parser.parse(dem)

    return NXDecodingGraph(graph, [boundary]), logicals_affected


class DetectorCounter(ErrorHandler):
    """Error handler that counts the number of detectors that each error flips."""

    def __init__(self) -> None:
        self.counter: Counter[int] = Counter()

    def __call__(self, error: stim.DemInstruction, detector_offset: int) -> None:
        self.counter.update(
            len(detectors) for detectors, _, _ in collect_edges(error, detector_offset)
        )

    def max_num_detectors(self) -> int:
        """
        Returns
        -------
        int
            Maximum number of detectors associated with a single error.
        """
        return max(self.counter.keys(), default=0)
