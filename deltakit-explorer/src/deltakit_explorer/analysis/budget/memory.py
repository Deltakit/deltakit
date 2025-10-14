from typing import Protocol

from deltakit_circuit._circuit import Circuit
from deltakit_circuit.gates._abstract_gates import PauliBasis
from deltakit_explorer.codes._css._css_code_experiment_circuit import css_code_memory_circuit
from deltakit_explorer.codes._planar_code._rotated_planar_code import RotatedPlanarCode

class MemoryGenerator(Protocol):
    def __call__(self, distance: int, num_rounds: int) -> Circuit:
        ...


def get_rotated_surface_code_memory_circuit(distance: int, num_rounds: int) -> Circuit:
    return css_code_memory_circuit(
        RotatedPlanarCode(distance, distance), num_rounds, PauliBasis.Z
    )
