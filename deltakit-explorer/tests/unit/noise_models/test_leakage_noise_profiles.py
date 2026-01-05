# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import Circuit, GateLayer, NoiseContext
from deltakit_circuit.gates import (
    CX,
    MX,
    MY,
    MZ,
    ONE_QUBIT_GATES,
    RX,
    RY,
    RZ,
    TWO_QUBIT_GATES,
    H,
    X,
)
from deltakit_circuit.noise_channels import Depolarise1, Leakage, Relax

from deltakit_explorer.qpu._noise._leakage_noise_profiles import (
    idle_qubit_relaxation_noise_profile,
    one_qubit_clifford_gate_relaxation_noise_profile,
    qubit_reset_leakage_noise_profile,
    resonator_idle_qubit_relaxation_noise_profile,
    two_qubit_gate_leakage_noise_profile,
    two_qubit_gate_relaxation_noise_profile,
)

ONE_QUBIT_GATES = list(ONE_QUBIT_GATES)
ONE_QUBIT_GATES = sorted(ONE_QUBIT_GATES, key=lambda x: x.__name__)
one_qubit_gate_params = [pytest.param(gate) for gate in ONE_QUBIT_GATES]


TWO_QUBIT_GATES = list(TWO_QUBIT_GATES)
TWO_QUBIT_GATES = sorted(TWO_QUBIT_GATES, key=lambda x: x.__name__)
two_qubit_gate_params = [pytest.param(gate) for gate in TWO_QUBIT_GATES]


class TestRelaxationNoiseProfiles:
    relaxation_prob = 0.5

    @pytest.fixture(scope="class")
    def five_qubit_circuit(self):
        return Circuit(GateLayer(RZ(qubit) for qubit in range(5)))

    @pytest.mark.parametrize(
        ("input_layer", "expected_noise_channels"),
        [
            (GateLayer(X(1)), [Relax(1, relaxation_prob)]),
            (
                GateLayer([H(1), CX(0, 2)]),
                [
                    Relax(1, relaxation_prob),
                ],
            ),
            (
                GateLayer([X(2), CX(3, 4), H(9)]),
                [
                    Relax(2, relaxation_prob),
                    Relax(9, relaxation_prob),
                ],
            ),
            (
                GateLayer([CX(5, 6), CX(3, 4), H(9)]),
                [
                    Relax(9, relaxation_prob),
                ],
            ),
        ],
    )
    def test_one_qubit_clifford_gate_relaxation_noise_profile(
        self, input_layer, expected_noise_channels
    ):
        context = NoiseContext(Circuit(input_layer), input_layer)
        noise_channels = one_qubit_clifford_gate_relaxation_noise_profile(
            self.relaxation_prob
        )(context)
        assert noise_channels == expected_noise_channels

    @pytest.mark.parametrize("gate_cls", one_qubit_gate_params)
    def test_one_qubit_clifford_gate_relaxation_noise_profile_exhaustive_gateset(
        self, gate_cls
    ):
        input_layer = GateLayer(gate_cls(0))
        context = NoiseContext(Circuit(input_layer), input_layer)
        noise_channels = one_qubit_clifford_gate_relaxation_noise_profile(
            self.relaxation_prob
        )(context)
        assert noise_channels == [Relax(0, self.relaxation_prob)]

    @pytest.mark.parametrize(
        ("input_layer", "expected_noise_channels"),
        [
            (
                GateLayer(X(4)),
                [
                    Relax(0, relaxation_prob),
                    Relax(1, relaxation_prob),
                    Relax(2, relaxation_prob),
                    Relax(3, relaxation_prob),
                ],
            ),
            (
                GateLayer([H(1), CX(0, 2)]),
                [
                    Relax(3, relaxation_prob),
                    Relax(4, relaxation_prob),
                ],
            ),
            (
                GateLayer([X(2), CX(3, 4), H(1)]),
                [
                    Relax(0, relaxation_prob),
                ],
            ),
            (
                GateLayer([CX(1, 0), CX(3, 4)]),
                [
                    Relax(2, relaxation_prob),
                ],
            ),
        ],
    )
    def test_idle_qubit_relaxation_noise_profile(
        self, five_qubit_circuit, input_layer, expected_noise_channels
    ):
        five_qubit_circuit.append_layers(input_layer)
        context = NoiseContext(five_qubit_circuit, input_layer)
        noise_channels = idle_qubit_relaxation_noise_profile(self.relaxation_prob)(
            context
        )
        assert noise_channels == expected_noise_channels

    @pytest.mark.parametrize(
        ("input_layer", "expected_noise_channels"),
        [
            (
                GateLayer(MZ(4)),
                [
                    Depolarise1(0, 0.1),
                    Depolarise1(1, 0.1),
                    Depolarise1(2, 0.1),
                    Depolarise1(3, 0.1),
                    Relax(0, 0.5),
                    Relax(1, 0.5),
                    Relax(2, 0.5),
                    Relax(3, 0.5),
                ],
            ),
            (
                GateLayer([MX(3), MY(4)]),
                [
                    Depolarise1(0, 0.1),
                    Depolarise1(1, 0.1),
                    Depolarise1(2, 0.1),
                    Relax(0, 0.5),
                    Relax(1, 0.5),
                    Relax(2, 0.5),
                ],
            ),
            (
                GateLayer([RZ(0), MY(4)]),
                [
                    Depolarise1(1, 0.1),
                    Depolarise1(2, 0.1),
                    Depolarise1(3, 0.1),
                    Relax(1, 0.5),
                    Relax(2, 0.5),
                    Relax(3, 0.5),
                ],
            ),
            (
                GateLayer([RZ(0), RX(1), RZ(2), RY(3), RX(4)]),
                [],
            ),
        ],
    )
    def test_resonator_idle_qubit_relaxation_noise_profile(
        self, five_qubit_circuit, input_layer, expected_noise_channels
    ):
        p_err = 0.1
        five_qubit_circuit.append_layers(input_layer)
        context = NoiseContext(five_qubit_circuit, input_layer)
        noise_channels = resonator_idle_qubit_relaxation_noise_profile(
            p_err, self.relaxation_prob
        )(context)
        assert noise_channels == expected_noise_channels

    @pytest.mark.parametrize(
        ("input_layer", "expected_noise_channels"),
        [
            (GateLayer(X(1)), []),
            (
                GateLayer([H(1), CX(0, 2)]),
                [Relax(0, relaxation_prob), Relax(2, relaxation_prob)],
            ),
            (
                GateLayer([X(2), CX(3, 4), H(9)]),
                [
                    Relax(3, relaxation_prob),
                    Relax(4, relaxation_prob),
                ],
            ),
            (
                GateLayer([CX(1, 0), CX(3, 4)]),
                [
                    Relax(1, relaxation_prob),
                    Relax(0, relaxation_prob),
                    Relax(3, relaxation_prob),
                    Relax(4, relaxation_prob),
                ],
            ),
        ],
    )
    def test_two_qubit_gate_relaxation_noise_profile(
        self, input_layer, expected_noise_channels
    ):
        context = NoiseContext(Circuit(input_layer), input_layer)
        noise_channels = two_qubit_gate_relaxation_noise_profile(self.relaxation_prob)(
            context
        )
        assert noise_channels == expected_noise_channels

    @pytest.mark.parametrize("gate_cls", two_qubit_gate_params)
    def test_two_qubit_gate_relaxation_noise_profile_exhaustive_gateset(self, gate_cls):
        input_layer = GateLayer(gate_cls(0, 1))
        context = NoiseContext(Circuit(input_layer), input_layer)
        noise_channels = two_qubit_gate_relaxation_noise_profile(self.relaxation_prob)(
            context
        )
        assert noise_channels == [
            Relax(0, self.relaxation_prob),
            Relax(1, self.relaxation_prob),
        ]


class TestLeakageNoiseProfiles:
    leakage_prob = 0.5

    @pytest.mark.parametrize(
        ("input_layer", "expected_noise_channels"),
        [
            (GateLayer(X(1)), []),
            (
                GateLayer([RZ(1), CX(0, 2)]),
                [Leakage(1, leakage_prob)],
            ),
            (
                GateLayer([RZ(1), CX(3, 4), RY(2)]),
                [
                    Leakage(1, leakage_prob),
                    Leakage(2, leakage_prob),
                ],
            ),
            (
                GateLayer([CX(1, 0), RZ(2), RX(3), RY(4)]),
                [
                    Leakage(2, leakage_prob),
                    Leakage(3, leakage_prob),
                    Leakage(4, leakage_prob),
                ],
            ),
        ],
    )
    def test_qubit_reset_leakage_noise_profile(
        self, input_layer, expected_noise_channels
    ):
        context = NoiseContext(Circuit(input_layer), input_layer)
        noise_channels = qubit_reset_leakage_noise_profile(self.leakage_prob)(context)
        assert noise_channels == expected_noise_channels

    @pytest.mark.parametrize(
        ("input_layer", "expected_noise_channels"),
        [
            (GateLayer(X(1)), []),
            (
                GateLayer([H(1), CX(0, 2)]),
                [Leakage(0, leakage_prob), Leakage(2, leakage_prob)],
            ),
            (
                GateLayer([X(2), CX(3, 4), H(9)]),
                [
                    Leakage(3, leakage_prob),
                    Leakage(4, leakage_prob),
                ],
            ),
            (
                GateLayer([CX(1, 0), CX(3, 4)]),
                [
                    Leakage(1, leakage_prob),
                    Leakage(0, leakage_prob),
                    Leakage(3, leakage_prob),
                    Leakage(4, leakage_prob),
                ],
            ),
        ],
    )
    def test_two_qubit_gate_leakage_noise_profile(
        self, input_layer, expected_noise_channels
    ):
        context = NoiseContext(Circuit(input_layer), input_layer)
        noise_channels = two_qubit_gate_leakage_noise_profile(self.leakage_prob)(
            context
        )
        assert noise_channels == expected_noise_channels

    @pytest.mark.parametrize("gate_cls", two_qubit_gate_params)
    def test_two_qubit_gate_leakage_noise_profile_exhaustive_gateset(self, gate_cls):
        input_layer = GateLayer(gate_cls(0, 1))
        context = NoiseContext(Circuit(input_layer), input_layer)
        noise_channels = two_qubit_gate_leakage_noise_profile(self.leakage_prob)(
            context
        )
        assert noise_channels == [
            Leakage(0, self.leakage_prob),
            Leakage(1, self.leakage_prob),
        ]
