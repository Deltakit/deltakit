# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import Circuit, GateLayer, NoiseContext, Qubit, SweepBit
from deltakit_circuit.gates import CX, MRX, MRY, MRZ, RX, RY, RZ, H
from deltakit_circuit.noise_channels import (
    Depolarise1,
    Depolarise2,
    PauliXError,
    PauliZError,
)

from deltakit_explorer.qpu._noise import SI1000Noise


class TestSI1000Noise:
    @pytest.fixture(scope="class")
    def si1000_noise_model(self):
        return SI1000Noise(p=0.1)

    @pytest.fixture(scope="class")
    def si1000_reset_noise_model(self):
        return SI1000Noise(p=0.1)

    @pytest.mark.parametrize("prob", [0.1])
    def test_correct_probability_is_defined(self, prob):
        assert SI1000Noise(p=prob).p == prob

    @pytest.mark.parametrize(
        ("input_layer", "expected_output_layers"),
        [
            (
                GateLayer([H(1), CX(0, 2)]),
                [
                    Depolarise1(1, 0.01),
                    Depolarise2(0, (2), 0.1),
                ],
            ),
            (
                GateLayer([CX(5, 6), CX(3, 4), H(9)]),
                [
                    Depolarise1(9, 0.01),
                    Depolarise2(5, (6), 0.1),
                    Depolarise2(3, (4), 0.1),
                ],
            ),
            (
                GateLayer(
                    [
                        CX(control=SweepBit(1), target=Qubit(1)),
                        CX(control=SweepBit(2), target=Qubit(2)),
                        CX(control=SweepBit(3), target=Qubit(3)),
                        CX(control=SweepBit(4), target=Qubit(4)),
                    ]
                ),
                [
                    Depolarise1(1, 0.01),
                    Depolarise1(2, 0.01),
                    Depolarise1(3, 0.01),
                    Depolarise1(4, 0.01),
                ],
            ),
        ],
    )
    def test_default_gate_noise_channels(
        self, si1000_noise_model, input_layer, expected_output_layers
    ):
        output_layers = []
        for gate_noise_channel in si1000_noise_model.gate_noise:
            output_layers.extend(
                gate_noise_channel(NoiseContext(Circuit(input_layer), input_layer))
            )

        assert output_layers == expected_output_layers

    @pytest.mark.parametrize("depolarize_1", [Depolarise1(1, 0.01)])
    def test_default_idle_noise(self, si1000_noise_model, depolarize_1):
        assert si1000_noise_model.idle_noise(1) == depolarize_1

    @pytest.mark.parametrize(
        ("gate", "x_after"),
        [
            (RX, PauliZError(1, 0.2)),
            (RY, PauliXError(1, 0.2)),
            (RZ, PauliXError(1, 0.2)),
            (MRX, PauliZError(1, 0.2)),
            (MRY, PauliXError(1, 0.2)),
            (MRZ, PauliXError(1, 0.2)),
        ],
    )
    def test_default_reset_noise(self, si1000_noise_model, gate, x_after):
        layer = GateLayer(gate(1))
        circuit = Circuit(layer)
        noise_channels = [
            nc
            for noise_channel in si1000_noise_model.reset_noise
            for nc in noise_channel(NoiseContext(circuit, layer))
        ]
        assert noise_channels == [x_after]

    @pytest.mark.parametrize(("si1000_noise", "str_val"), [
        (SI1000Noise(p=0.123), "SI1000_noise_1e-01"),
        (SI1000Noise(p=0.123, pL=0.234), "SI1000_noise_1e-01_2e-01")
    ])
    def test_si1000_str(self, si1000_noise, str_val):
        assert str(si1000_noise) == str_val

    def test_position_args_raise_error(self):
        with pytest.raises(TypeError, match="positional argument"):
            SI1000Noise(0.01)
