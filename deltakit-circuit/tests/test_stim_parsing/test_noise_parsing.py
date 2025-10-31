# (c) Copyright Riverlane 2020-2025.
import deltakit_circuit as sp
import pytest
import stim
from deltakit_circuit._parse_stim import parse_stim_noise_instruction


def test_probability_is_added_on_noise_channels():
    probability = 0.01
    layer = parse_stim_noise_instruction(
        sp.noise_channels.PauliXError,
        [stim.GateTarget(0)],
        [probability],
        instruction_tag="",
        qubit_mapping={},
    )
    assert layer.noise_channels[0].probability == probability


def test_error_is_raised_when_gate_class_is_passed_to_the_noise_parser():
    with pytest.raises(
        ValueError,
        match=r"Given noise class:.*is not a "
        "valid deltakit_circuit noise channel.",
    ):
        parse_stim_noise_instruction(
            sp.gates.X, [0, 2, 3], [0.01], instruction_tag="", qubit_mapping={}
        )


@pytest.mark.parametrize(
    "stim_circuit, expected_noise_channel",
    [
        (
            stim.Circuit("X_ERROR(0.2) 0"),
            sp.noise_channels.PauliXError(sp.Qubit(0), 0.2),
        ),
        (
            stim.Circuit("QUBIT_COORDS(0, 0) 0 \nX_ERROR(0.2) 0"),
            sp.noise_channels.PauliXError(sp.Coordinate(0, 0), 0.2),
        ),
        (
            stim.Circuit("Y_ERROR(0.4) 2"),
            sp.noise_channels.PauliYError(sp.Qubit(2), 0.4),
        ),
        (
            stim.Circuit("Z_ERROR(0.3) 1"),
            sp.noise_channels.PauliZError(sp.Qubit(1), 0.3),
        ),
        (
            stim.Circuit("QUBIT_COORDS(0, 0) 0 \nDEPOLARIZE1(0.001) 0"),
            sp.noise_channels.Depolarise1(sp.Coordinate(0, 0), 0.001),
        ),
        (
            stim.Circuit("DEPOLARIZE2(0.002) 0 4"),
            sp.noise_channels.Depolarise2(sp.Qubit(0), sp.Qubit(4), 0.002),
        ),
        (
            stim.Circuit(
                "QUBIT_COORDS(0, 0) 0 \n QUBIT_COORDS(2, 3) 4 \nDEPOLARIZE2(0.002) 0 4"
            ),
            sp.noise_channels.Depolarise2(
                sp.Coordinate(0, 0), sp.Coordinate(2, 3), 0.002
            ),
        ),
        (
            stim.Circuit("PAULI_CHANNEL_1(0.01, 0.02, 0.03) 3"),
            sp.noise_channels.PauliChannel1(sp.Qubit(3), 0.01, 0.02, 0.03),
        ),
        (
            stim.Circuit("QUBIT_COORDS(0, 0) 3 \nPAULI_CHANNEL_1(0.01, 0.02, 0.03) 3"),
            sp.noise_channels.PauliChannel1(sp.Coordinate(0, 0), 0.01, 0.02, 0.03),
        ),
        (
            stim.Circuit(
                "PAULI_CHANNEL_2(0.001, 0.002, 0.003, 0.004, 0.005, "
                "0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, "
                "0.014, 0.015) 2 1"
            ),
            sp.noise_channels.PauliChannel2(
                sp.Qubit(2),
                sp.Qubit(1),
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.011,
                0.012,
                0.013,
                0.014,
                0.015,
            ),
        ),
        (
            stim.Circuit(
                "QUBIT_COORDS(0, 0) 2 \n QUBIT_COORDS(4, 5) 1 \n"
                "PAULI_CHANNEL_2(0.001, 0.002, 0.003, 0.004, 0.005, "
                "0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, "
                "0.014, 0.015) 2 1"
            ),
            sp.noise_channels.PauliChannel2(
                sp.Coordinate(0, 0),
                sp.Coordinate(4, 5),
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.011,
                0.012,
                0.013,
                0.014,
                0.015,
            ),
        ),
        (
            stim.Circuit("CORRELATED_ERROR(0.002) X0 Y3"),
            sp.noise_channels.CorrelatedError(
                sp.PauliProduct((sp.PauliX(sp.Qubit(0)), sp.PauliY(sp.Qubit(3)))), 0.002
            ),
        ),
        (
            stim.Circuit(
                "QUBIT_COORDS(3, 2) 0 \nQUBIT_COORDS(2, 2) 3 \nCORRELATED_ERROR(0.002) X0 Y3"
            ),
            sp.noise_channels.CorrelatedError(
                sp.PauliProduct(
                    (sp.PauliX(sp.Coordinate(3, 2)), sp.PauliY(sp.Coordinate(2, 2)))
                ),
                0.002,
            ),
        ),
        (
            stim.Circuit("ELSE_CORRELATED_ERROR(0.001) Z2 X3"),
            sp.noise_channels.ElseCorrelatedError(
                sp.PauliProduct((sp.PauliZ(sp.Qubit(2)), sp.PauliX(sp.Qubit(3)))), 0.001
            ),
        ),
        # (stim.Circuit("LEAKAGE(0.002) 0"),
        #  sp.noise_channels.Leakage(sp.Qubit(0), 0.002)),
        # (stim.Circuit("RELAX(0.001) 0"),
        #  sp.noise_channels.Relax(sp.Qubit(0), 0.001)),
    ],
)
def test_parsing_single_noise_channel_stim_circuit_returns_correct_deltakit_circuit_circuit(
    stim_circuit, expected_noise_channel
):
    expected_circuit = sp.Circuit(sp.NoiseLayer(expected_noise_channel))
    assert sp.Circuit.from_stim_circuit(stim_circuit) == expected_circuit


@pytest.mark.parametrize(
    "stim_circuit, expected_circuit",
    [
        (
            stim.Circuit("X_ERROR(0.001) 0 1 2\nZ_ERROR(0.02) 0 1 2"),
            sp.Circuit(
                [
                    sp.NoiseLayer(
                        sp.noise_channels.PauliXError(sp.Qubit(i), 0.001)
                        for i in range(3)
                    ),
                    sp.NoiseLayer(
                        sp.noise_channels.PauliZError(sp.Qubit(i), 0.02)
                        for i in range(3)
                    ),
                ]
            ),
        ),
        # (
        #     stim.Circuit("LEAKAGE(0.001) 0 1 2\nRELAX(0.02) 0 1 2"),
        #     sp.Circuit([
        #         sp.NoiseLayer(sp.noise_channels.Leakage(
        #             sp.Qubit(i), 0.001) for i in range(3)),
        #         sp.NoiseLayer(sp.noise_channels.Relax(
        #             sp.Qubit(i), 0.02) for i in range(3))
        #     ])
        # ),
        (
            stim.Circuit("DEPOLARIZE1(0.002) 0 4 3\nCORRELATED_ERROR(0.01) X3 Y5"),
            sp.Circuit(
                [
                    sp.NoiseLayer(
                        sp.noise_channels.Depolarise1(sp.Qubit(i), 0.002)
                        for i in (0, 4, 3)
                    ),
                    sp.NoiseLayer(
                        sp.noise_channels.CorrelatedError(
                            sp.PauliProduct(
                                [sp.PauliX(sp.Qubit(3)), sp.PauliY(sp.Qubit(5))]
                            ),
                            0.01,
                        )
                    ),
                ]
            ),
        ),
    ],
)
def test_parsing_multiple_noise_channel_stim_circuit_returns_correct_deltakit_circuit_circuit(
    stim_circuit, expected_circuit
):
    assert sp.Circuit.from_stim_circuit(stim_circuit) == expected_circuit
