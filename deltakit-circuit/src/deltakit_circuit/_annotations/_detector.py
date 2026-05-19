# (c) Copyright Riverlane 2020-2025.
"""Module which provides detectors and measurement records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import chain

import deltakit_stim

from deltakit_circuit._deltakit_stim_version_compatibility import (
    is_deltakit_stim_tag_feature_available,
)
from deltakit_circuit._qubit_identifiers import Coordinate, MeasurementRecord


class Detector:
    """Annotates that a set of measurements can be used to detect errors,
    because the set's parity should be deterministic.

    Parameters
    ----------
    measurements : MeasurementRecord | Iterable[MeasurementRecord]
        The measurements that this is the detectors of.
    coordinate: Iterable[float] | None
        An optional coordinate to associate with this detector.
    tag: str | None
        An optional instruction tag.
    """

    deltakit_stim_string = "DETECTOR"

    def __init__(
        self,
        measurements: MeasurementRecord | Iterable[MeasurementRecord],
        coordinate: Iterable[float] | None = None,
        *,
        tag: str | None = None,
    ):
        self._measurements = (
            frozenset((measurements,))
            if isinstance(measurements, MeasurementRecord)
            else frozenset(measurements)
        )
        self._coordinate = Coordinate(*coordinate) if coordinate is not None else None
        self._tag = tag

    @property
    def tag(self) -> str | None:
        return self._tag

    @property
    def coordinate(self) -> Coordinate | None:
        """Get the coordinate which specifies this detector."""
        return self._coordinate

    @property
    def measurements(self) -> frozenset[MeasurementRecord]:
        return self._measurements

    def transform_coordinates(
        self, coordinate_mapping: Mapping[Coordinate, Coordinate]
    ):
        """
        Transform this detectors coordinates according to the coordinate
        mapping. No transformation is performed if coordinate is not in the
        mapping.

        Parameters
        ----------
        coordinate_mapping : Mapping[Coordinate, Coordinate]
            A mapping of qubit types to other qubit types
        """
        # Functionally passing None to the get method is fine but mypy doesn't
        # like argument to get being Optional[Coordinate].
        if (current_coordinate := self._coordinate) is not None:
            self._coordinate = coordinate_mapping.get(
                current_coordinate, current_coordinate
            )

    def permute_deltakit_stim_circuit(
        self, deltakit_stim_circuit: deltakit_stim.Circuit, _qubit_mapping=None
    ):
        """Updates deltakit_stim_circuit with the deltakit_stim circuit.

        The deltakit_stim circuit specifies the single detector.

        Parameters
        ----------
        deltakit_stim_circuit : deltakit_stim.Circuit
            The deltakit_stim circuit to be updated with the deltakit_stim
            representation of this detector.

        _qubit_mapping : None, optional
            Unused argument to make interface to this method equal to the
            same methods in layer classes.
        """
        deltakit_stim_targets = chain.from_iterable(
            record.deltakit_stim_targets() for record in self.measurements
        )
        deltakit_stim_arguments = self.coordinate if self.coordinate is not None else ()
        kwargs = (
            {"tag": self.tag}
            if self.tag is not None and is_deltakit_stim_tag_feature_available()
            else {}
        )
        deltakit_stim_circuit.append(
            self.deltakit_stim_string,
            deltakit_stim_targets,
            deltakit_stim_arguments,
            **kwargs,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Detector)
            and self.measurements == other.measurements
            and self.coordinate == other.coordinate
        )

    def __hash__(self) -> int:
        return hash((self._measurements, self._coordinate))

    def __repr__(self) -> str:
        tag_repr = f"[{self._tag}]" if self._tag is not None else ""
        return (
            f"Detector{tag_repr}({list(self.measurements)}, "
            f"coordinate={self.coordinate})"
        )
