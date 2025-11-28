from typing import Protocol

from deltakit_circuit import Circuit
from deltakit_circuit.gates import PauliBasis

from deltakit_explorer.codes import RotatedPlanarCode, css_code_memory_circuit


class MemoryGenerator(Protocol):
    def __call__(self, distance: int, num_rounds: int) -> Circuit: ...


def get_rotated_surface_code_memory_circuit(distance: int, num_rounds: int) -> Circuit:
    """Returns a rotated surface code Z memory experiment."""
    return css_code_memory_circuit(
        RotatedPlanarCode(distance, distance), num_rounds, PauliBasis.Z
    )
