# (c) Copyright Riverlane 2020-2025.
"""``deltakit.circuit.gates`` contains class-based representations of Stim gates."""

from deltakit_circuit.gates._abstract_gates import (
    Gate,
    OneQubitCliffordGate,
    OneQubitGate,
    OneQubitMeasurementGate,
    OneQubitResetGate,
    PauliBasis,
    SymmetricTwoQubitGate,
    TwoOperandGate,
)
from deltakit_circuit.gates._measurement_gates import (
    HERALD_LEAKAGE_EVENT,
    MEASUREMENT_GATES,
    MPP,
    MRX,
    MRY,
    MRZ,
    MX,
    MY,
    MZ,
    ONE_QUBIT_MEASUREMENT_GATES,
    _MeasurementGate,
)
from deltakit_circuit.gates._one_qubit_gates import (
    C_XYZ,
    C_ZYX,
    H_XY,
    H_YZ,
    ONE_QUBIT_GATES,
    S_DAG,
    SQRT_X,
    SQRT_X_DAG,
    SQRT_Y,
    SQRT_Y_DAG,
    H,
    I,
    S,
    X,
    Y,
    Z,
    _OneQubitCliffordGate,
)
from deltakit_circuit.gates._reset_gates import RESET_GATES, RX, RY, RZ, _ResetGate
from deltakit_circuit.gates._two_qubit_gates import (
    CX,
    CXSWAP,
    CY,
    CZ,
    CZSWAP,
    ISWAP,
    ISWAP_DAG,
    SQRT_XX,
    SQRT_XX_DAG,
    SQRT_YY,
    SQRT_YY_DAG,
    SQRT_ZZ,
    SQRT_ZZ_DAG,
    SWAP,
    TWO_QUBIT_GATES,
    XCX,
    XCY,
    XCZ,
    YCX,
    YCY,
    YCZ,
    _TwoQubitGate,
)

_Gate = _OneQubitCliffordGate | _TwoQubitGate | _ResetGate | _MeasurementGate

ONE_QUBIT_GATE_MAPPING: dict[str, type[_OneQubitCliffordGate]] = {
    **{gate.stim_string: gate for gate in ONE_QUBIT_GATES},
    "H_XZ": H,
    "SQRT_Z": S,
    "SQRT_Z_DAG": S_DAG,
}

TWO_QUBIT_GATE_MAPPING: dict[str, type[_TwoQubitGate]] = {
    **{gate.stim_string: gate for gate in TWO_QUBIT_GATES},
    "ZCX": CX,
    "CNOT": CX,
    "ZCY": CY,
    "ZCZ": CZ,
}

ONE_QUBIT_MEASUREMENT_GATE_MAPPING = {
    **{gate.stim_string: gate for gate in MEASUREMENT_GATES - {MPP}},
    "M": MZ,
    "MR": MRZ,
}

MEASUREMENT_GATE_MAPPING: dict[str, type[_MeasurementGate]] = {
    **ONE_QUBIT_MEASUREMENT_GATE_MAPPING,
    MPP.stim_string: MPP,
}

RESET_GATE_MAPPING: dict[str, type[_ResetGate]] = {
    **{gate.stim_string: gate for gate in RESET_GATES},
    "R": RZ,
}

GATE_MAPPING: dict[str, type[_Gate]] = {
    **ONE_QUBIT_GATE_MAPPING,
    **TWO_QUBIT_GATE_MAPPING,
    **MEASUREMENT_GATE_MAPPING,
    **RESET_GATE_MAPPING,
}

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
