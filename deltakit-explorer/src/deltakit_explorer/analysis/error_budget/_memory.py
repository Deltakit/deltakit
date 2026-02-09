from collections.abc import Mapping
from typing import Protocol

from deltakit_circuit import Circuit
from deltakit_circuit.gates import PauliBasis

from deltakit_explorer.codes import RotatedPlanarCode, css_code_memory_circuit


class MemoryGenerator(Protocol):
    def __call__(self, distance: int, num_rounds: int) -> Circuit: ...


class PreComputedMemoryGenerator(MemoryGenerator):
    def __init__(self, circuits: Mapping[int, Mapping[int, Circuit]]) -> None:
        """A memory generator that used pre-computed circuits.

        Args:
            circuits: a mapping from distance values to another mapping that maps num_rounds values
                to actual quantum circuits. Will be used as ``circuits[distance][num_rounds]``.
        """
        super().__init__()
        self._circuits = circuits

    def __call__(self, distance: int, num_rounds: int) -> Circuit:
        if distance not in self._circuits:
            msg = f"No circuit provided for {distance=}."
            raise RuntimeError(msg)
        circuits_of_distance = self._circuits[distance]
        if num_rounds not in circuits_of_distance:
            msg = f"No circuit provided for {num_rounds=}."
            raise RuntimeError(msg)
        return circuits_of_distance[num_rounds]


def get_rotated_surface_code_memory_circuit(distance: int, num_rounds: int) -> Circuit:
    """Returns a rotated surface code Z memory experiment."""
    return css_code_memory_circuit(
        RotatedPlanarCode(distance, distance), num_rounds, PauliBasis.Z
    )
