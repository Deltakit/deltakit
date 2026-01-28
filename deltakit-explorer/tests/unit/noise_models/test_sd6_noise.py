# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import Circuit, GateLayer, NoiseContext, NoiseLayer, Qubit
from deltakit_circuit.gates import CX, MRX, MRY, MRZ, MZ, RZ, H, X, Y
from deltakit_circuit.noise_channels import (
    Depolarise1,
    Depolarise2,
    PauliXError,
    PauliZError,
)

from deltakit_explorer.qpu import QPU
from deltakit_explorer.qpu._native_gate_set import NativeGateSet
from deltakit_explorer.qpu._noise import SD6Noise


class TestSD6NoiseModel:
    @pytest.fixture(scope="class")
    def noise_model(self):
        return SD6Noise(p=0.1)

    @pytest.mark.parametrize("prob", [0.1])
    def test_correct_probability_is_defined(self, prob):
        assert SD6Noise(p=prob).p == prob

    @pytest.mark.parametrize(
        ("gate_layer", "expected_gate_layer"),
        [
            (GateLayer(MZ(0)), GateLayer(MZ(0, 0.1))),
            (GateLayer(MRX(0)), GateLayer(MRX(0, 0.1))),
            (GateLayer(MRY(0)), GateLayer(MRY(0, 0.1))),
            (GateLayer(MRZ(0)), GateLayer(MRZ(0, 0.1))),
        ],
    )
    def test_measurement_flip_noise_construction(
        self, noise_model, gate_layer, expected_gate_layer
    ):
        gate_layer.replace_gates(noise_model.measurement_flip)
        assert gate_layer == expected_gate_layer

    @pytest.mark.parametrize(
        ("input_layer", "expected_output_layers"),
        [
            (
                GateLayer([H(Qubit(1)), CX(0, 2)]),
                [
                    Depolarise1(qubit=Qubit(1), probability=0.1),
                    Depolarise2(Qubit(0), Qubit(2), probability=0.1),
                ],
            ),
            (
                GateLayer([CX(5, 6), CX(3, 4), H(9)]),
                [
                    Depolarise1(qubit=Qubit(9), probability=0.1),
                    Depolarise2(Qubit(5), Qubit(6), probability=0.1),
                    Depolarise2(Qubit(3), Qubit(4), probability=0.1),
                ],
            ),
        ],
    )
    def test_default_gate_noise_channels(
        self, noise_model, input_layer, expected_output_layers
    ):
        output_layers = []
        for gate_noise_channel in noise_model.gate_noise:
            output_layers.extend(
                gate_noise_channel(NoiseContext(Circuit(input_layer), input_layer))
            )

        assert output_layers == expected_output_layers

    @pytest.mark.parametrize(
        "expected_noise_channel", [Depolarise1(Qubit(1), probability=0.1)]
    )
    def test_default_idle_noise(self, noise_model, expected_noise_channel):
        assert noise_model.idle_noise(Qubit(1)) == expected_noise_channel

    @pytest.mark.parametrize(
        ("gate_layer", "noise_after"),
        [
            (GateLayer(RZ(Qubit(1))), PauliXError(1, 0.1)),
            (GateLayer(MRX(Qubit(1))), PauliZError(1, 0.1)),
            (GateLayer(MRY(Qubit(1))), PauliXError(1, 0.1)),
            (GateLayer(MRZ(Qubit(1))), PauliXError(1, 0.1)),
        ],
    )
    def test_default_reset_noise(self, noise_model, gate_layer, noise_after):
        circuit = Circuit(gate_layer)
        noise_channels = [
            nc
            for noise_channel in noise_model.reset_noise
            for nc in noise_channel(NoiseContext(circuit, gate_layer))
        ]
        assert noise_channels == [noise_after]

    def test_fixed_circuit_reset(self, noise_model):
        circuit = Circuit(
            [
                GateLayer([RZ(qubit) for qubit in [0, 1, 2, 3, 4, 5]]),
                GateLayer([MRZ(qubit) for qubit in [0, 1, 2, 3, 4, 5]]),
                GateLayer([H(1), CX(3, 4)]),
                GateLayer([CX(0, 1), CX(4, 5), X(2)]),
                GateLayer([X(2), MZ(3)]),
                GateLayer([CX(1, 2), Y(3), RZ(0)]),
                GateLayer([CX(0, 1), CX(4, 5)]),
                GateLayer([MZ(qubit) for qubit in [0, 1, 2, 3, 4, 5]]),
            ]
        )
        expected_noisy_circuit = Circuit(
            [
                GateLayer([RZ(qubit) for qubit in [0, 1, 2, 3, 4, 5]]),
                NoiseLayer([PauliXError(qubit, 0.1) for qubit in [0, 1, 2, 3, 4, 5]]),
                GateLayer([MRZ(qubit, 0.1) for qubit in [0, 1, 2, 3, 4, 5]]),
                NoiseLayer([PauliXError(qubit, 0.1) for qubit in [0, 1, 2, 3, 4, 5]]),
                GateLayer([H(1), CX(3, 4)]),
                NoiseLayer([Depolarise1(1, 0.1), Depolarise2(3, 4, 0.1)]),
                NoiseLayer(
                    [Depolarise1(0, 0.1), Depolarise1(2, 0.1), Depolarise1(5, 0.1)]
                ),
                GateLayer([CX(0, 1), CX(4, 5), X(2)]),
                NoiseLayer(
                    [
                        Depolarise1(2, 0.1),
                        Depolarise2(0, 1, 0.1),
                        Depolarise2(4, 5, 0.1),
                    ]
                ),
                NoiseLayer(Depolarise1(3, 0.1)),
                GateLayer([X(2), MZ(3, 0.1)]),
                NoiseLayer(
                    [
                        Depolarise1(2, 0.1),
                    ]
                ),
                NoiseLayer(
                    [
                        Depolarise1(0, 0.1),
                        Depolarise1(1, 0.1),
                        Depolarise1(4, 0.1),
                        Depolarise1(5, 0.1),
                    ]
                ),
                GateLayer([CX(1, 2), Y(3), RZ(0)]),
                NoiseLayer(
                    [Depolarise1(3, 0.1), Depolarise2(1, 2, 0.1), PauliXError(0, 0.1)]
                ),
                NoiseLayer([Depolarise1(4, 0.1), Depolarise1(5, 0.1)]),
                GateLayer([CX(0, 1), CX(4, 5)]),
                NoiseLayer([Depolarise2(0, 1, 0.1), Depolarise2(4, 5, 0.1)]),
                NoiseLayer([Depolarise1(2, 0.1), Depolarise1(3, 0.1)]),
                GateLayer([MZ(qubit, 0.1) for qubit in [0, 1, 2, 3, 4, 5]]),
            ]
        )
        native_gates = NativeGateSet(
            one_qubit_gates={H, X, Y},
            two_qubit_gates={CX},
            reset_gates={RZ},
            measurement_gates={MZ, MRZ},
        )
        qpu = QPU(
            qubits=circuit.qubits,
            noise_model=noise_model,
            native_gates_and_times=native_gates,
        )
        noisy_circuit = qpu.add_noise_to_circuit(circuit)
        assert expected_noisy_circuit == noisy_circuit

    def test_sd6_noise_str(self):
        noise_model = SD6Noise(p=0.123)
        assert str(noise_model) == 'sd6_noise_1e-01'

    def test_position_args_raise_error(self):
        with pytest.raises(TypeError, match="positional argument"):
            SD6Noise(0.01)

