# (c) Copyright Riverlane 2020-2025.
import pytest
import stim
from deltakit_circuit import ShiftCoordinates


def test_shift_coordinates_stim_circuit_gives_expected_circuit(empty_circuit):
    ShiftCoordinates((1, 2, 3, 4)).permute_stim_circuit(empty_circuit)
    assert empty_circuit == stim.Circuit("SHIFT_COORDS(1, 2, 3, 4)")


def test_shift_coordinate_instances_are_equal_if_their_shifts_are_equal():
    assert ShiftCoordinates((1, 2, 3, 4)) == ShiftCoordinates((1, 2, 3, 4))


def test_shift_coordinate_instances_are_not_equal_if_their_shifts_are_different():
    assert ShiftCoordinates((1, 2, 3, 4)) != ShiftCoordinates((4, 3, 2, 1))


def test_repr_of_shift_coordinates_matches_expected_representation():
    assert repr(ShiftCoordinates((0, 1, 2))) == "ShiftCoordinates(Coordinate(0, 1, 2))"


@pytest.mark.parametrize("tag", [None, "", "sjkdhf", "λ", "leaky<0>"])
def test_tag_on_instruction(tag: str | None) -> None:
    instr = ShiftCoordinates((0, 1, 2), tag=tag)
    assert instr.tag == tag
