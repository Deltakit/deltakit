# (c) Copyright Riverlane 2020-2025.
"""
This module defines commonly-used maps between objects.
"""

from deltakit_circuit import PauliX, PauliY, PauliZ
from deltakit_circuit._basic_types import CP
from deltakit_circuit._qubit_identifiers import PauliGate
from deltakit_circuit.gates import CX, CY, CZ, PauliBasis

BASIS_TO_PAULI = {PauliBasis.X: PauliX, PauliBasis.Y: PauliY, PauliBasis.Z: PauliZ}

PAULI_TO_CP = {
    PauliX: CX,
    PauliY: CY,
    PauliZ: CZ,
}

GATE_TO_PAULI: dict[CP, PauliGate] = {
    CX: PauliX,
    CZ: PauliZ,
}
