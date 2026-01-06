# (c) Copyright Riverlane 2020-2025.
import pytest

from deltakit_circuit._basic_types import Coord2D, Coord2DDelta


class TestCoord2DDelta:
    def test_raises_error_if_multiplied_with_qcoord(self):
        with pytest.raises(TypeError):
            Coord2DDelta(1, 1).dot(Coord2D(1, 1))


class TestCoord2D:
    def test_raises_error_if_tuple_added(self):
        with pytest.raises(TypeError):
            Coord2D(1, 1) + (1, 1)  # noqa: RUF005

    def test_raises_error_if_qcoorddelta_subtracted(self):
        with pytest.raises(TypeError):
            Coord2D(1, 1) - Coord2DDelta(1, 1)
