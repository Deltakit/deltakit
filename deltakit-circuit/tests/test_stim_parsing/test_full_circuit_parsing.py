# (c) Copyright Riverlane 2020-2025.
import pytest
import stim

import deltakit_circuit as sp
from deltakit_circuit._parse_stim import (
    InstructionNotImplemented,
    parse_circuit_instruction,
)


@pytest.mark.parametrize(
    "unhandled_instruction",
    [
        stim.CircuitInstruction("TICK", []),
        stim.CircuitInstruction("QUBIT_COORDS", [0], (0, 1)),
    ],
)
def test_error_is_raised_when_parsing_unhandled_circuit_instruction(
    unhandled_instruction,
):
    with pytest.raises(InstructionNotImplemented):
        parse_circuit_instruction(unhandled_instruction, {})


# @pytest.mark.parametrize("gate_class",
#                          sp.gates.TWO_QUBIT_GATES | sp.gates.MEASUREMENT_GATES
#                          )
# def test_error_is_raised_if_qubit_is_acted_on_more_than_once_in_a_stim_circuit(
#         gate_class):
#     """Currently two qubit and measurement gates do not support grouping which
#     means deltakit_circuit cannot handle these gates acting on the same qubit multiple
#     times in a single stim operation. It is possible to implement grouping for
#     two qubit gates but just hasn't been implemented so for now raise an error.
#     """
#     stim_circuit = (stim.Circuit("MPP X0 Z0")
#                     if issubclass(gate_class, sp.gates.MPP) else
#                     stim.Circuit(f"{gate_class.stim_string} 0 1 0 2"))
#     with pytest.raises(DuplicateQubitError):
#         sp.Circuit.from_stim_circuit(stim_circuit)


@pytest.mark.parametrize("gate_class", sp.gates.ONE_QUBIT_GATES | sp.gates.RESET_GATES)
def test_multiple_gate_layers_are_created_for_gates_that_support_qubit_grouping(
    gate_class,
):
    stim_circuit = stim.Circuit(f"{gate_class.stim_string} 0 1 0 2")
    assert len(sp.Circuit.from_stim_circuit(stim_circuit).layers) == 2


@pytest.mark.parametrize(
    ("stim_circuit", "expected_circuit"),
    [
        # (
        #     stim.Circuit("X 0 1 2\nDEPOLARIZE1(0.02) 0 2\nLEAKAGE(0.001) 0"),
        #     sp.Circuit([
        #         sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in range(3)),
        #         sp.NoiseLayer(sp.noise_channels.Depolarise1(
        #             sp.Qubit(i), 0.02) for i in (0, 2)),
        #         sp.NoiseLayer(sp.noise_channels.Leakage(sp.Qubit(0), 0.001))
        #     ])
        # ),
        (
            stim.Circuit("S 0\nX_ERROR(0.02) 0\nCZ 0 1\nDEPOLARIZE2(0.3) 0 1"),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.S(sp.Qubit(0))),
                    sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.02)),
                    sp.GateLayer(sp.gates.CZ(sp.Qubit(0), sp.Qubit(1))),
                    sp.NoiseLayer(
                        sp.noise_channels.Depolarise2(sp.Qubit(0), sp.Qubit(1), 0.3)
                    ),
                ]
            ),
        )
    ],
)
def test_parsing_noisy_circuit_returns_correct_deltakit_circuit_circuit(
    stim_circuit, expected_circuit
):
    assert sp.Circuit.from_stim_circuit(stim_circuit) == expected_circuit


def test_moving_from_stim_to_deltakit_circuit_to_stim_preserves_coordinates_at_start_of_stim_file():
    stim_coordinates = "QUBIT_COORDS(1, 0) 0\nQUBIT_COORDS(0, 0) 1\n"
    instructions = stim_coordinates + "X 1 \nCX 1 0\nM 0"
    deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(stim.Circuit(instructions))
    assert str(deltakit_circuit_circuit.as_stim_circuit()).startswith(stim_coordinates)


def test_moving_from_stim_to_deltakit_circuit_to_stim_preserves_stim_ids():
    stim_coordinates = "QUBIT_COORDS(1, 0) 0\nQUBIT_COORDS(0, 0) 1\n"
    instructions = stim_coordinates + "X 1 \nCX 1 0\nM 0"
    deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(stim.Circuit(instructions))
    qubits = list(deltakit_circuit_circuit.qubits)
    assert qubits[0].stim_identifier == 0 or qubits[1].stim_identifier == 0
    assert qubits[0].stim_identifier == 1 or qubits[1].stim_identifier == 1
    assert len(deltakit_circuit_circuit.qubits) == 2


@pytest.mark.parametrize(
    "stim_generated_circuit",
    [
        stim.Circuit.generated("repetition_code:memory", rounds=100, distance=9),
        stim.Circuit.generated(
            "surface_code:unrotated_memory_x", rounds=100, distance=3
        ),
        stim.Circuit.generated(
            "surface_code:unrotated_memory_z", rounds=100, distance=3
        ),
        stim.Circuit.generated("surface_code:rotated_memory_x", rounds=100, distance=3),
        stim.Circuit.generated("surface_code:rotated_memory_z", rounds=100, distance=3),
    ],
)
def test_parsing_stim_generated_circuits_does_not_raise_an_error(
    stim_generated_circuit,
):
    assert sp.Circuit.from_stim_circuit(stim_generated_circuit)
