# (c) Copyright Riverlane 2020-2025.
from itertools import chain

import pytest
from deltakit_circuit import GateLayer, PauliX, Qubit
from deltakit_circuit._basic_types import Coord2D
from deltakit_circuit.gates import MPP, MRX, MRY, MRZ, MX, MY, MZ, H, I
from deltakit_circuit.noise_channels._depolarising_noise import Depolarise1

from deltakit_explorer.qpu._noise import PhenomenologicalNoise
from deltakit_explorer.qpu._noise._phenomenological_noise import (
    ToyPhenomenologicalNoise,
)

qubits = [Qubit(0), Qubit(17), Qubit(Coord2D(1, 1))]


class MockNoiseContext:
    gate_layer = GateLayer([H(0), I(2)])

    def gate_layer_qubits(self, gate_t):
        return chain.from_iterable(
            gate.qubits for gate in self.gate_layer.gates if isinstance(gate, gate_t)
        )


class TestPhenomenologicalNoise:
    def test_noise_profiles_are_initialised_to_empty_lists(self):
        noise_model = PhenomenologicalNoise()
        assert noise_model.idle_noise is None
        assert noise_model.reset_noise == []
        assert noise_model.measurement_noise_after == []

    @pytest.mark.parametrize(
        ("noise_model", "probability"),
        [
            (
                PhenomenologicalNoise(
                    phenomenological_noise=lambda qubit: Depolarise1(qubit, 0.0),
                ),
                0.0,
            ),
            (
                PhenomenologicalNoise(
                    phenomenological_noise=lambda qubit: Depolarise1(qubit, 0.001),
                ),
                0.001,
            ),
            (
                PhenomenologicalNoise(
                    phenomenological_noise=lambda qubit: Depolarise1(qubit, 0.02),
                ),
                0.02,
            ),
        ],
    )
    @pytest.mark.parametrize("qubit", qubits)
    def test_correct_phenomenological_noise(self, noise_model, probability, qubit):
        assert noise_model.phenomenological_noise(qubit) == Depolarise1(
            qubit, probability
        )

    @pytest.mark.parametrize(
        ("phenomenological_noise", "probability"),
        [
            (lambda qubit: Depolarise1(qubit, 0.0), 0.0),
            (lambda qubit: Depolarise1(qubit, 0.001), 0.001),
            (lambda qubit: Depolarise1(qubit, 0.02), 0.02),
        ],
    )
    def test_correct_gate_noise_can_be_generated(
        self, phenomenological_noise, probability
    ):
        noise_model = PhenomenologicalNoise(
            phenomenological_noise=phenomenological_noise
        )
        expected_noise_channel = [Depolarise1(Qubit(2), probability)]
        assert len(noise_model.gate_noise) == 1
        assert noise_model.gate_noise[0](MockNoiseContext()) == expected_noise_channel

    def test_position_args_raise_error(self):
        with pytest.raises(TypeError, match="positional argument"):
            PhenomenologicalNoise(0.01)

class TestToyPhenomenologicalNoise:
    def test_noise_profiles_are_initialised_to_empty_lists(self):
        noise_model = ToyPhenomenologicalNoise()
        assert noise_model.idle_noise is None
        assert noise_model.reset_noise == []
        assert noise_model.measurement_noise_after == []

    @pytest.mark.parametrize(
        ("noise_model", "probability"),
        [
            (ToyPhenomenologicalNoise(), 0.0),
            (ToyPhenomenologicalNoise(p=0.001), 0.001),
            (ToyPhenomenologicalNoise(p=0.02, p_measurement_flip=0.005), 0.02),
        ],
    )
    @pytest.mark.parametrize("qubit", qubits)
    def test_correct_phenomenological_noise(self, noise_model, probability, qubit):
        assert noise_model.phenomenological_noise(qubit) == Depolarise1(
            qubit, probability
        )

    @pytest.mark.parametrize(
        "gate",
        [
            MX(0),
            MY(0),
            MZ(0),
            MRX(0),
            MRY(0),
            MRZ(0),
            MPP([PauliX(0)]),
        ],
    )
    def test_correct_measurement_flip_noise(self, gate):
        noise_model = ToyPhenomenologicalNoise(p=0.1)
        gate_t = type(gate)
        assert noise_model.measurement_flip[gate_t](gate).probability == 0.1

    def test_position_args_raise_error(self):
        with pytest.raises(TypeError, match="positional argument"):
            ToyPhenomenologicalNoise(0.01)
