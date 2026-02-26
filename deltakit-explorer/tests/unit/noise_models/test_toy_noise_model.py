# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import Circuit, GateLayer, NoiseContext, Qubit
from deltakit_circuit.gates import CX, MZ, RZ, H
from deltakit_circuit.noise_channels import Depolarise1, Depolarise2

from deltakit_explorer.qpu._noise import ToyNoise


class TestToyNoise:
    @pytest.mark.parametrize("p_measurement_flip", [1e-2])
    def test_default_measurement_flip(self, p_measurement_flip):
        assert ToyNoise(p=1e-2).p_measurement_flip == p_measurement_flip

    @pytest.fixture(scope="class")
    def toy_noise_model(self):
        return ToyNoise(p=0.1)

    @pytest.mark.parametrize("prob", [0.1])
    def test_correct_probability_is_defined(self, prob):
        assert ToyNoise(p=prob).p == prob

    @pytest.mark.parametrize("p_measurement_flip", [1e-4])
    def test_default_measurement_flip_is_overwritten(self, p_measurement_flip):
        assert (
            ToyNoise(p=1e-2, p_measurement_flip=p_measurement_flip).p_measurement_flip
            == p_measurement_flip
        )

    @pytest.mark.parametrize(
        ("input_layer", "expected_output_layers"),
        [
            (
                GateLayer([H(Qubit(1)), CX(0, 2)]),
                [
                    Depolarise1(qubit=Qubit(1), probability=0.01),
                    Depolarise2(Qubit(0), Qubit(2), probability=0.1),
                ],
            ),
            (
                GateLayer([CX(5, 6), CX(3, 4), H(9)]),
                [
                    Depolarise1(qubit=Qubit(9), probability=0.01),
                    Depolarise2(Qubit(5), Qubit(6), probability=0.1),
                    Depolarise2(Qubit(3), Qubit(4), probability=0.1),
                ],
            ),
        ],
    )
    def test_default_gate_noise_channels(
        self, toy_noise_model, input_layer, expected_output_layers
    ):
        output_layers = []
        for gate_noise_channel in toy_noise_model.gate_noise:
            output_layers.extend(
                gate_noise_channel(NoiseContext(Circuit(input_layer), input_layer))
            )

        assert output_layers == expected_output_layers

    @pytest.mark.parametrize("depolarize_1", [Depolarise1(Qubit(1), probability=0.01)])
    def test_default_idle_noise(self, toy_noise_model, depolarize_1):
        assert toy_noise_model.idle_noise(Qubit(1)) == depolarize_1

    @pytest.mark.parametrize("depolarize_1", [Depolarise1(Qubit(1), probability=0.01)])
    def test_default_reset_noise(self, toy_noise_model, depolarize_1):
        layer = GateLayer(RZ(Qubit(1)))
        circuit = Circuit(layer)
        assert toy_noise_model.reset_noise[0](NoiseContext(circuit, layer)) == [
            depolarize_1
        ]

    @pytest.mark.parametrize("depolarize_1", [Depolarise1(Qubit(1), probability=0.01)])
    def test_default_measurement_noise(self, toy_noise_model, depolarize_1):
        layer = GateLayer(MZ(Qubit(1)))
        circuit = Circuit(layer)
        assert toy_noise_model.measurement_noise_after[0](
            NoiseContext(circuit, layer)
        ) == [depolarize_1]

    @pytest.mark.parametrize(("toy_noise", "str_val"), [
        (ToyNoise(p=0.123), "toy_noise_1e-01"),
        (ToyNoise(p=0.123, p_measurement_flip=0.234), "toy_noise_1e-01_2e-01")
    ])
    def test_toy_noise_str(self, toy_noise, str_val):
        assert str(toy_noise) == str_val

    def test_position_args_raise_error(self):
        with pytest.raises(TypeError, match="positional argument"):
            ToyNoise(0.01)
