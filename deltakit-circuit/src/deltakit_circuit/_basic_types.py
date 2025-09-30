# (c) Copyright Riverlane 2020-2025.
"""
This module defines important basic classes used in quantum error correction code
modelling.
"""

from __future__ import annotations

from functools import cached_property
from typing import NamedTuple, Union

from deltakit_circuit._qubit_identifiers import Coordinate
from deltakit_circuit.gates import CX, CY, CZ, PauliBasis

PauliBases = tuple[PauliBasis, ...]

# pylint: disable=invalid-name
CP = Union[CX, CY, CZ]


class Coord2DDelta(NamedTuple):
    """
    Class to accommodate a vector that points from one point in the x-y plane to
    another.

    Parameters
    ----------
    x : float
        The component of the vector in the x direction.
    y : float
        The component of the vector in the y direction.
    """

    x: float
    y: float

    def __add__(self, other) -> Coord2DDelta:
        """
        Add the Coord2DDelta object and another Coord2DDelta object.
        """
        if isinstance(other, Coord2DDelta):
            return Coord2DDelta(self.x + other.x, self.y + other.y)
        return NotImplemented

    def dot(self, other: object) -> float:
        """
        Multiply the Coord2DDelta object with another object. This is only defined
        if the other object is also a Coord2DDelta object, in which case the inner
        product is calculated.
        """
        if isinstance(other, Coord2DDelta):
            return self.x * other.x + self.y * other.y
        raise TypeError(f"Dot product is not implemented for {type(other)}")


class Coord2D(Coordinate):
    """Class to accommodate a coordinate in the x-y plane."""

    @cached_property
    def x(self) -> float:
        """The x-coordinate."""
        return self[0]

    @cached_property
    def y(self) -> float:
        """The y-coordinate."""
        return self[1]

    def __add__(self, other) -> Coord2D:
        """
        Add the Coord2D object and another object. This is only defined if the other
        object is another Coord2D object or a Coord2DDelta object.
        """
        if isinstance(other, (Coord2DDelta, Coord2D)):
            return Coord2D(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other: Coord2D) -> Coord2DDelta:
        """
        Subtract another object from the Coord2D object. This is only defined if the
        other object is a Coord2D object, in which case a Coord2DDelta object is
        returned, representing a vector point from one coordinate to the other.
        """
        if isinstance(other, Coord2D):
            return Coord2DDelta(self.x - other.x, self.y - other.y)
        return NotImplemented

    def to_torus_coord(
        self,
        x_mod: int,
        y_mod: int,
    ) -> Coord2D:
        """
        From planar coordinates, get the torus version by taking moduli
        with respect to the x and y directional lengths (x_mod and y_mod).
        """
        return Coord2D(self.x % x_mod, self.y % y_mod)

    def __repr__(self) -> str:
        return f"Coord2D({self.x}, {self.y})"
