# (c) Copyright Riverlane 2020-2025.
"""Module which defines a way to shift coordinates."""

from __future__ import annotations

from collections.abc import Iterable

import deltakit_stim

from deltakit_circuit._deltakit_stim_version_compatibility import (
    is_deltakit_stim_tag_feature_available,
)
from deltakit_circuit._qubit_identifiers import Coordinate


class ShiftCoordinates:
    """Annotates a shift in the coordinates within a deltakit_stim circuit. This
    modifies coordinates associated to detectors and the user is required to
    put each shift coordinate in manually.

    Parameters
    ----------
    coordinate_shift: Iterable[int | float]
        The coordinate shift to impose. Note that this is the delta and not
        the absolute coordinate.
    """

    def __init__(
        self,
        coordinate_shift: Iterable[int | float],
        *,
        tag: str | None = None,
    ):
        self._coordinate_shift = Coordinate(*coordinate_shift)
        self._tag = tag

    @property
    def tag(self) -> str | None:
        return self._tag

    def permute_deltakit_stim_circuit(
        self, deltakit_stim_circuit: deltakit_stim.Circuit, _qubit_mapping=None
    ):
        """Updates deltakit_stim_circuit with the single deltakit_stim circuit.

        The deltakit_stim circuit contains the single coordinate shift.

        Parameters
        ----------
        deltakit_stim_circuit : deltakit_stim.Circuit
            The deltakit_stim circuit to be updated with the deltakit_stim
            representation of this single coordinate shift

        _qubit_mapping : None
            Unused argument to keep interface with other layer classes clean.
        """
        kwargs = (
            {"tag": self.tag}
            if self.tag is not None and is_deltakit_stim_tag_feature_available()
            else {}
        )
        deltakit_stim_circuit.append(
            "SHIFT_COORDS", [], self._coordinate_shift, **kwargs
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ShiftCoordinates)
            and self._coordinate_shift == other._coordinate_shift
        )

    def __hash__(self) -> int:
        return hash(self._coordinate_shift)

    def __repr__(self) -> str:
        tag_repr = f"[{self._tag}]" if self._tag is not None else ""
        return f"ShiftCoordinates{tag_repr}({self._coordinate_shift})"
