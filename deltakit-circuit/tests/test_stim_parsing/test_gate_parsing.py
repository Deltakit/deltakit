# (c) Copyright Riverlane 2020-2025.
import deltakit_circuit as sp
import pytest
import stim
from deltakit_circuit._parse_stim import (
    _classify_pauli_target,
    parse_stim_gate_instruction,
)


def test_probability_is_added_on_measurement_gates():
    probability = 0.001
    layer = parse_stim_gate_instruction(
        sp.gates.MZ, [stim.GateTarget(0)], [probability], "", qubit_mapping={}
    )
    assert list(layer.gates)[0].probability == probability


@pytest.mark.parametrize("non_pauli_target", [stim.GateTarget(0)])
def test_error_is_raised_when_classifying_non_pauli_target(non_pauli_target):
    with pytest.raises(ValueError, match=r"Target:.*is not a Pauli gate target"):
        _classify_pauli_target(non_pauli_target, qubit_mapping={})


def test_error_is_raised_when_noise_class_is_passed_to_gate_parser():
    with pytest.raises(
        ValueError,
        match=r"Given gate class:.*is not a "
        "valid deltakit_circuit gate.",
    ):
        parse_stim_gate_instruction(
            sp.noise_channels.PauliXError, [0, 1], [], "", qubit_mapping={}
        )


@pytest.mark.parametrize(
    "stim_circuit, expected_gates",
    [
        (stim.Circuit("X 0 1 2 3 4"), [sp.gates.X(sp.Qubit(i)) for i in range(5)]),
        (
            stim.Circuit("QUBIT_COORDS(0, 0) 0 \nQUBIT_COORDS(0, 1) 1 \nX 0 1"),
            [sp.gates.X(sp.Coordinate(*coord)) for coord in [(0, 0), (0, 1)]],
        ),
        (stim.Circuit("Y 0"), sp.gates.Y(sp.Qubit(0))),
        (stim.Circuit("CNOT 0 1"), sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
        (
            stim.Circuit("QUBIT_COORDS(0, 0) 0 \nQUBIT_COORDS(0, 1) 1 \nCNOT 0 1"),
            [sp.gates.CX(sp.Coordinate(0, 0), sp.Coordinate(0, 1))],
        ),
        (
            stim.Circuit("CNOT 0 1 2 3"),
            [
                sp.gates.CX(sp.Qubit(0), sp.Qubit(1)),
                sp.gates.CX(sp.Qubit(2), sp.Qubit(3)),
            ],
        ),
        (stim.Circuit("RX 0"), sp.gates.RX(sp.Qubit(0))),
        (stim.Circuit("RY 2 3 5"), [sp.gates.RY(sp.Qubit(i)) for i in (2, 3, 5)]),
        (stim.Circuit("MX 0 1 2 3"), [sp.gates.MX(sp.Qubit(i)) for i in range(4)]),
        (
            stim.Circuit("QUBIT_COORDS(0, 0) 0 \nQUBIT_COORDS(0, 1) 1 \nMX 0 1"),
            [sp.gates.MX(sp.Coordinate(*coord)) for coord in [(0, 0), (0, 1)]],
        ),
        (
            stim.Circuit("MRZ(0.1) 2 5"),
            [sp.gates.MRZ(sp.Qubit(i), 0.1) for i in (2, 5)],
        ),
        (
            stim.Circuit("MRY !4 5"),
            [~sp.gates.MRY(sp.Qubit(4)), sp.gates.MRY(sp.Qubit(5))],
        ),
        (
            stim.Circuit("MZ(0.4) 2 !0"),
            [sp.gates.MZ(sp.Qubit(2), 0.4), ~sp.gates.MZ(sp.Qubit(0), 0.4)],
        ),
        (stim.Circuit("MPP Z0"), sp.gates.MPP(sp.PauliZ(sp.Qubit(0)))),
        (stim.Circuit("MPP !Z5"), sp.gates.MPP(~sp.InvertiblePauliZ(sp.Qubit(5)))),
        (stim.Circuit("MPP(0.2) X0"), sp.gates.MPP(sp.PauliX(sp.Qubit(0)), 0.2)),
        (
            stim.Circuit("QUBIT_COORDS(0, 0) 0 \nMPP(0.2) X0"),
            sp.gates.MPP(sp.PauliX(sp.Coordinate(0, 0)), 0.2),
        ),
        (
            stim.Circuit("MPP(0.1) !Y5"),
            sp.gates.MPP(~sp.InvertiblePauliY(sp.Qubit(5)), 0.1),
        ),
        (
            stim.Circuit("MPP(0.2) X0 Z1"),
            [
                sp.gates.MPP(sp.PauliX(sp.Qubit(0)), 0.2),
                sp.gates.MPP(sp.PauliZ(sp.Qubit(1)), 0.2),
            ],
        ),
        (
            stim.Circuit("MPP X1*Y2"),
            sp.gates.MPP(
                sp.MeasurementPauliProduct(
                    [sp.PauliX(sp.Qubit(1)), sp.PauliY(sp.Qubit(2))]
                )
            ),
        ),
        (
            stim.Circuit("MPP(0.01) Z1*X3"),
            sp.gates.MPP(
                sp.MeasurementPauliProduct(
                    [sp.PauliZ(sp.Qubit(1)), sp.PauliX(sp.Qubit(3))]
                ),
                0.01,
            ),
        ),
        (
            stim.Circuit("MPP !X4*Y2"),
            sp.gates.MPP(
                sp.MeasurementPauliProduct(
                    [~sp.InvertiblePauliX(sp.Qubit(4)), sp.PauliY(sp.Qubit(2))]
                )
            ),
        ),
        (
            stim.Circuit("MPP(0.02) Z5*!Y1"),
            sp.gates.MPP(
                sp.MeasurementPauliProduct(
                    [sp.PauliZ(sp.Qubit(5)), ~sp.InvertiblePauliY(sp.Qubit(1))]
                ),
                0.02,
            ),
        ),
        (
            stim.Circuit("MPP X0 X1*X2"),
            [
                sp.gates.MPP(sp.PauliX(sp.Qubit(0))),
                sp.gates.MPP(
                    sp.MeasurementPauliProduct(sp.PauliX(sp.Qubit(i)) for i in (1, 2))
                ),
            ],
        ),
        (
            stim.Circuit("MPP Z0*Z1 Z2"),
            [
                sp.gates.MPP(
                    sp.MeasurementPauliProduct(sp.PauliZ(sp.Qubit(i)) for i in (0, 1))
                ),
                sp.gates.MPP(sp.PauliZ(sp.Qubit(2))),
            ],
        ),
        (
            stim.Circuit("MPP X0 Z1*Z2 Y3"),
            [
                sp.gates.MPP(sp.PauliX(sp.Qubit(0))),
                sp.gates.MPP(
                    sp.MeasurementPauliProduct(sp.PauliZ(sp.Qubit(i)) for i in (1, 2))
                ),
                sp.gates.MPP(sp.PauliY(sp.Qubit(3))),
            ],
        ),
        (
            stim.Circuit("MPP X0*X1*X2"),
            sp.gates.MPP(
                sp.MeasurementPauliProduct(
                    sp.InvertiblePauliX(sp.Qubit(i)) for i in (0, 1, 2)
                )
            ),
        ),
        (
            stim.Circuit("CXSWAP 0 1 2 3"),
            [
                sp.gates.CXSWAP(sp.Qubit(0), sp.Qubit(1)),
                sp.gates.CXSWAP(sp.Qubit(2), sp.Qubit(3)),
            ],
        ),
        (
            stim.Circuit("CZSWAP 0 1 2 3"),
            [
                sp.gates.CZSWAP(sp.Qubit(0), sp.Qubit(1)),
                sp.gates.CZSWAP(sp.Qubit(2), sp.Qubit(3)),
            ],
        ),
    ],
)
def test_parsing_stim_circuit_with_single_gate_layer_returns_the_correct_deltakit_circuit_circuit(
    stim_circuit, expected_gates
):
    expected_circuit = sp.Circuit(sp.GateLayer(expected_gates))
    assert sp.Circuit.from_stim_circuit(stim_circuit) == expected_circuit


@pytest.mark.parametrize(
    "stim_circuit, expected_deltakit_circuit_circuit",
    [
        (
            stim.Circuit("CX rec[-1] 0"),
            sp.Circuit(sp.GateLayer(sp.gates.CX(sp.MeasurementRecord(-1), 0))),
        ),
        (
            stim.Circuit("CY rec[-1] 0"),
            sp.Circuit(sp.GateLayer(sp.gates.CY(sp.MeasurementRecord(-1), 0))),
        ),
        (
            stim.Circuit("CZ rec[-1] 0"),
            sp.Circuit(sp.GateLayer(sp.gates.CZ(sp.MeasurementRecord(-1), 0))),
        ),
        (
            stim.Circuit("XCZ 0 rec[-1]"),
            sp.Circuit(sp.GateLayer(sp.gates.XCZ(0, sp.MeasurementRecord(-1)))),
        ),
        (
            stim.Circuit("YCZ 0 rec[-1]"),
            sp.Circuit(sp.GateLayer(sp.gates.YCZ(0, sp.MeasurementRecord(-1)))),
        ),
    ],
)
def test_parsing_controlled_gates_with_measurement_records_gives_expected_deltakit_circuit_circuit(
    stim_circuit: stim.Circuit, expected_deltakit_circuit_circuit: sp.Circuit
):
    assert (
        sp.Circuit.from_stim_circuit(stim_circuit) == expected_deltakit_circuit_circuit
    )


@pytest.mark.parametrize(
    "stim_circuit, expected_deltakit_circuit_circuit",
    [
        (
            stim.Circuit("CX sweep[1] 0"),
            sp.Circuit(sp.GateLayer(sp.gates.CX(sp.SweepBit(1), 0))),
        ),
        (
            stim.Circuit("CY sweep[1] 0"),
            sp.Circuit(sp.GateLayer(sp.gates.CY(sp.SweepBit(1), 0))),
        ),
        (
            stim.Circuit("CZ sweep[1] 0"),
            sp.Circuit(sp.GateLayer(sp.gates.CZ(sp.SweepBit(1), 0))),
        ),
        (
            stim.Circuit("XCZ 0 sweep[1]"),
            sp.Circuit(sp.GateLayer(sp.gates.XCZ(0, sp.SweepBit(1)))),
        ),
        (
            stim.Circuit("YCZ 0 sweep[1]"),
            sp.Circuit(sp.GateLayer(sp.gates.YCZ(0, sp.SweepBit(1)))),
        ),
    ],
)
def test_parsing_controlled_gates_with_sweep_bits_gives_expected_deltakit_circuit_circuit(
    stim_circuit: stim.Circuit, expected_deltakit_circuit_circuit: sp.Circuit
):
    assert (
        sp.Circuit.from_stim_circuit(stim_circuit) == expected_deltakit_circuit_circuit
    )


@pytest.mark.parametrize(
    "stim_circuit, expected_circuit",
    [
        (
            stim.Circuit("X 1\nY 0"),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.X(sp.Qubit(1))),
                    sp.GateLayer(sp.gates.Y(sp.Qubit(0))),
                ]
            ),
        ),
        (
            stim.Circuit("H 0\nCZ 0 1"),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.H(sp.Qubit(0))),
                    sp.GateLayer(sp.gates.CZ(sp.Qubit(0), sp.Qubit(1))),
                ]
            ),
        ),
        (
            stim.Circuit("RZ 0 1 2 3 4\nM 0 1 2 3 4"),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.RZ(sp.Qubit(i)) for i in range(5)),
                    sp.GateLayer(sp.gates.MZ(sp.Qubit(i)) for i in range(5)),
                ]
            ),
        ),
        (
            stim.Circuit("S 0 1\nMPP X0*X1"),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.S(sp.Qubit(i)) for i in (0, 1)),
                    sp.GateLayer(
                        sp.gates.MPP(
                            sp.MeasurementPauliProduct(
                                [sp.PauliX(sp.Qubit(0)), sp.PauliX(sp.Qubit(1))]
                            )
                        )
                    ),
                ]
            ),
        ),
    ],
)
def test_parsing_multiple_gate_stim_circuit_returns_correct_deltakit_circuit_circuit(
    stim_circuit, expected_circuit
):
    assert sp.Circuit.from_stim_circuit(stim_circuit) == expected_circuit


@pytest.mark.parametrize(
    "stim_circuit, expected_circuit",
    [
        (
            stim.Circuit("X 0 1 2 1 2"),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.X(qubit) for qubit in (0, 1, 2)),
                    sp.GateLayer(sp.gates.X(qubit) for qubit in (1, 2)),
                ]
            ),
        ),
        (
            stim.Circuit("RZ 0 0 1 2 3"),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.RZ(qubit) for qubit in (0, 1, 2, 3)),
                    sp.GateLayer(sp.gates.RZ(0)),
                ]
            ),
        ),
    ],
)
def test_parsing_single_gate_with_duplicate_qubits_returns_correct_circuit(
    stim_circuit, expected_circuit
):
    assert sp.Circuit.from_stim_circuit(stim_circuit) == expected_circuit
