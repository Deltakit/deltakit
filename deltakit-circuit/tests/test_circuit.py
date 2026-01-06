# (c) Copyright Riverlane 2020-2025.
import contextlib
from itertools import combinations

import pytest
import stim

import deltakit_circuit as sp


@pytest.fixture
def empty_circuit() -> sp.Circuit:
    return sp.Circuit()


@pytest.fixture
def noiseless_circuit(empty_circuit: sp.Circuit) -> sp.Circuit:
    empty_circuit.append_layers(
        [
            sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in range(10)),
            sp.GateLayer(
                sp.gates.CX(sp.Qubit(i), sp.Qubit(i + 1)) for i in range(0, 10, 2)
            ),
            sp.GateLayer(sp.gates.MZ(sp.Qubit(i)) for i in range(10)),
        ]
    )
    return empty_circuit


@pytest.fixture
def noisy_circuit(empty_circuit: sp.Circuit) -> sp.Circuit:
    empty_circuit.append_layers(
        [
            sp.GateLayer(sp.gates.X(sp.Qubit(0))),
            sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
            sp.GateLayer(sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
            sp.NoiseLayer(
                sp.noise_channels.Depolarise2(sp.Qubit(0), sp.Qubit(1), 0.02)
            ),
            sp.GateLayer(sp.gates.MX(sp.Qubit(0), 0.01)),
        ]
    )
    return empty_circuit


@pytest.fixture
def nested_circuit_with_noise() -> sp.Circuit:
    return sp.Circuit(
        sp.Circuit(
            [
                sp.NoiseLayer(
                    sp.noise_channels.PauliXError(sp.Qubit(0), probability=0.01)
                ),
                sp.GateLayer(sp.gates.MX(sp.Qubit(0), 0.01)),
            ],
            2,
        )
    )


@pytest.fixture
def noisy_measurement_circuit(empty_circuit: sp.Circuit) -> sp.Circuit:
    empty_circuit.append_layers(
        [
            sp.GateLayer(sp.gates.X(sp.Qubit(0))),
            sp.GateLayer(sp.gates.MX(sp.Qubit(0), 0.01)),
            sp.GateLayer(sp.gates.MPP(sp.PauliX(sp.Qubit(0)), 0.01)),
        ]
    )
    return empty_circuit


@pytest.mark.parametrize(
    "layers",
    [
        (
            sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in (0, 1)),
            sp.GateLayer(sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
        ),
        (
            sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
            sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(1), 0.02)),
        ),
        (
            sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in range(1)),
            sp.NoiseLayer(sp.noise_channels.Depolarise2(sp.Qubit(0), sp.Qubit(1), 0.2)),
            sp.GateLayer(sp.gates.MZ(sp.Qubit(i)) for i in range(1)),
        ),
    ],
)
def test_layers_are_added_to_circuit_if_passed_on_construction(layers):
    circuit = sp.Circuit(layers)
    assert all(layer in circuit.layers for layer in layers)


def test_constructing_circuits_from_objects_that_are_not_layers_raises_error():
    with pytest.raises(ValueError, match=r"Layer type is not one of .*"):
        sp.Circuit([sp.gates.X(sp.Qubit(0)), sp.gates.CX(sp.Qubit(0), sp.Qubit(1))])


def test_appending_objects_to_circuit_that_are_not_layers_raises_error():
    with pytest.raises(ValueError, match=r"Layer type is not one of .*"):
        sp.Circuit().append_layers(sp.gates.X(sp.Qubit(0)))


@pytest.mark.parametrize(
    ("circuit", "expected_repr"),
    [
        (
            sp.Circuit(sp.GateLayer(sp.gates.X(i) for i in range(2))),
            "Circuit([\n"
            "    GateLayer([\n"
            "        X(Qubit(0))\n"
            "        X(Qubit(1))\n"
            "    ])\n"
            "], iterations=1)",
        ),
        (
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.CX(0, 1)),
                    sp.NoiseLayer(sp.noise_channels.PauliXError(0, 0.002)),
                    sp.GateLayer([sp.gates.MZ(0), sp.gates.MZ(1)]),
                    sp.Detector([sp.MeasurementRecord(-1), sp.MeasurementRecord(-2)]),
                ]
            ),
            "Circuit([\n"
            "    GateLayer([\n"
            "        CX(control=Qubit(0), target=Qubit(1))\n"
            "    ])\n"
            "    NoiseLayer([\n"
            "        X_ERROR(Qubit(0), probability=0.002)\n"
            "    ])\n"
            "    GateLayer([\n"
            "        MZ(Qubit(0), probability=0.0)\n"
            "        MZ(Qubit(1), probability=0.0)\n"
            "    ])\n"
            "    Detector([MeasurementRecord(-1), MeasurementRecord(-2)], coordinate=None)\n"
            "], iterations=1)",
        ),
    ],
)
def test_repr_of_circuit_matches_expected_representation(circuit, expected_repr):
    assert repr(circuit) == expected_repr


def test_appending_single_gate_layer_adds_it_to_the_circuit(empty_circuit: sp.Circuit):
    gate_layer = sp.GateLayer(sp.gates.X(sp.Qubit(0)))
    empty_circuit.append_layers(gate_layer)
    assert gate_layer in empty_circuit.layers


def test_appending_multiple_gate_layers_adds_them_all_to_the_circuit(
    empty_circuit: sp.Circuit,
):
    gate_layers = [
        sp.GateLayer(sp.gates.X(sp.Qubit(0))),
        sp.GateLayer(sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
        sp.GateLayer([sp.gates.MZ(sp.Qubit(i)) for i in (0, 1)]),
    ]
    empty_circuit.append_layers(gate_layers)
    assert all(layer in empty_circuit.layers for layer in gate_layers)


def test_circuit_can_return_nested_gate_layers():
    deltakit_circuit_circuit = sp.Circuit(
        [sp.GateLayer(sp.gates.X(0)), sp.Circuit(sp.GateLayer(sp.gates.H(1)))]
    )
    assert deltakit_circuit_circuit.gate_layers() == [
        sp.GateLayer(sp.gates.X(0)),
        sp.GateLayer(sp.gates.H(1)),
    ]


def test_noisy_circuit_is_noisy(noisy_circuit: sp.Circuit):
    assert noisy_circuit.is_noisy


def test_circuit_with_measurement_noise_is_noisy(noisy_measurement_circuit: sp.Circuit):
    assert noisy_measurement_circuit.is_noisy


def test_circuit_with_noiseless_nested_circuit_is_not_noisy(empty_circuit: sp.Circuit):
    empty_circuit.append_layers(
        sp.Circuit(
            sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in range(10)), iterations=2
        )
    )
    assert not empty_circuit.is_noisy


@pytest.mark.parametrize(
    "nested_circuit",
    [
        sp.Circuit(
            sp.NoiseLayer(
                sp.noise_channels.Depolarise2(sp.Qubit(0), sp.Qubit(1), 0.01)
            ),
            iterations=2,
        ),
        sp.Circuit(sp.GateLayer(sp.gates.MX(0, 0.001)), iterations=2),
    ],
)
def test_noisy_nested_circuit_is_noisy(empty_circuit: sp.Circuit, nested_circuit):
    empty_circuit.append_layers(nested_circuit)
    assert empty_circuit.is_noisy


def test_nested_circuit_with_measurement_noise_is_noisy(empty_circuit: sp.Circuit):
    empty_circuit.append_layers(
        sp.Circuit(sp.GateLayer(sp.gates.MX(sp.Qubit(0), 0.01)), iterations=2)
    )
    assert empty_circuit.is_noisy


def test_noiseless_circuit_is_not_noisy(noiseless_circuit: sp.Circuit):
    assert not noiseless_circuit.is_noisy


def test_appending_single_noise_layer_adds_it_to_the_circuit(empty_circuit: sp.Circuit):
    noise_layer = sp.NoiseLayer(sp.noise_channels.PauliZError(sp.Qubit(0), 0.01))
    empty_circuit.append_layers(noise_layer)
    assert noise_layer in empty_circuit.layers


def test_appending_multiple_noise_layers_adds_them_all_to_the_circuit(
    empty_circuit: sp.Circuit,
):
    noise_layers = [
        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
        sp.NoiseLayer(sp.noise_channels.PauliZError(sp.Qubit(0), 0.02)),
    ]
    empty_circuit.append_layers(noise_layers)
    assert all(layer in empty_circuit.layers for layer in noise_layers)


def test_removing_noise_from_a_circuit_with_noise_layers_removes_all_noise_layers(
    noisy_circuit: sp.Circuit,
):
    noisy_circuit.remove_noise()
    assert not any(isinstance(layer, sp.NoiseLayer) for layer in noisy_circuit.layers)


def test_circuit_can_return_nested_noise_layers(nested_circuit_with_noise: sp.Circuit):
    assert nested_circuit_with_noise.noise_layers() == [
        sp.NoiseLayer([sp.noise_channels.PauliXError(0, 0.01)])
    ]


def test_remove_noise_handles_nested_circuits(nested_circuit_with_noise: sp.Circuit):
    nested_circuit_with_noise.remove_noise()
    assert not nested_circuit_with_noise.is_noisy


def test_remove_noise_does_not_remove_noise_from_nested_circuits_if_recursive_flag_set_to_false(
    nested_circuit_with_noise: sp.Circuit,
):
    nested_circuit_with_noise.remove_noise(recursive=False)
    assert all(
        layer.is_noisy
        for layer in nested_circuit_with_noise.layers
        if isinstance(layer, sp.Circuit)
    )


def test_removing_noise_from_a_circuit_with_measurement_noise_removes_measurement_noise(
    noisy_measurement_circuit: sp.Circuit,
):
    noisy_measurement_circuit.remove_noise()
    assert all(
        gate.probability == 0 for gate in noisy_measurement_circuit.measurement_gates
    )


class TestCircuitApproxEquals:
    def test_two_circuits_with_identical_layers_are_approx_equal(self):
        circuit1 = sp.Circuit(sp.GateLayer(sp.gates.X(sp.Qubit(i))) for i in range(10))
        circuit2 = sp.Circuit(sp.GateLayer(sp.gates.X(sp.Qubit(i))) for i in range(10))
        assert circuit1.approx_equals(circuit2)

    @pytest.mark.parametrize(
        ("layers1", "layers2"),
        [
            (
                sp.GateLayer(sp.gates.H(sp.Qubit(0))),
                sp.GateLayer(sp.gates.H(sp.Qubit(1))),
            ),
            (
                sp.GateLayer(sp.gates.S(sp.Qubit(0))),
                sp.GateLayer(sp.gates.C_ZYX(sp.Qubit(1))),
            ),
            (
                sp.GateLayer(sp.gates.X(sp.Qubit(0))),
                sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
            ),
            (
                sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
                sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.02)),
            ),
            (
                sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
                sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(1), 0.01)),
            ),
            (
                sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
                sp.NoiseLayer(
                    sp.noise_channels.Depolarise2(sp.Qubit(0), sp.Qubit(1), 0.02)
                ),
            ),
            (
                sp.GateLayer(sp.gates.X(i) for i in range(10)),
                [sp.GateLayer(sp.gates.X(i) for i in range(10)), sp.NoiseLayer()],
            ),
        ],
    )
    def test_two_circuits_with_different_layers_are_not_approx_equal(
        self, layers1, layers2
    ):
        assert not sp.Circuit(layers1).approx_equals(sp.Circuit(layers2))

    def test_two_circuits_with_different_number_of_iterations_are_not_approx_equal(
        self,
    ):
        layers = [
            sp.GateLayer(sp.gates.X(sp.Qubit(0))),
            sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
        ]
        assert not sp.Circuit(layers, 2).approx_equals(sp.Circuit(layers, 3))

    @pytest.mark.parametrize(
        ("layer1", "layer2"),
        combinations(
            [
                sp.Detector(sp.MeasurementRecord(-1)),
                sp.Detector((sp.MeasurementRecord(-1), sp.MeasurementRecord(-2))),
                sp.Observable(0, sp.MeasurementRecord(-1)),
                sp.Observable(1, (sp.MeasurementRecord(-1), sp.MeasurementRecord(-2))),
                sp.ShiftCoordinates((0, 0, 1)),
            ],
            2,
        ),
    )
    def test_circuits_with_deterministic_layers_in_wrong_order_are_not_approx_equal(
        self, layer1, layer2
    ):
        assert not sp.Circuit(layer1).approx_equals(sp.Circuit(layer2))

    @pytest.mark.parametrize(
        "other_layer",
        [
            sp.NoiseLayer(sp.noise_channels.PauliXError(0, 0.001)),
            sp.Detector([sp.MeasurementRecord(-1)]),
            sp.Observable(0, [sp.MeasurementRecord(-1)]),
            sp.ShiftCoordinates((0, 0, 1)),
            sp.GateLayer(sp.gates.H(0)),
        ],
    )
    def test_circuit_and_non_circuit_are_not_approx_equal(self, other_layer):
        circuit = sp.Circuit(sp.GateLayer(sp.gates.H(0)))
        assert not circuit.approx_equals(other_layer)

    def test_approx_equal_circuits_are_approx_equal_default_tol(self):
        circuit1 = sp.Circuit(
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.Z(1), sp.gates.MZ(2, 0.01)]),
                sp.Detector([sp.MeasurementRecord(-1)]),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(0, 0.02),
                        sp.noise_channels.CorrelatedError(sp.PauliX(1), 0.01),
                        sp.noise_channels.PauliZError(2, 0.02),
                        sp.noise_channels.ElseCorrelatedError(sp.PauliZ(2), 0.03),
                        sp.noise_channels.Depolarise2(3, 4, 0.02),
                    ]
                ),
                sp.Observable(0, [sp.MeasurementRecord(-1)]),
                sp.ShiftCoordinates((0, 0, 1)),
            ]
        )
        circuit2 = sp.Circuit(
            [
                sp.GateLayer(
                    [sp.gates.X(0), sp.gates.Z(1), sp.gates.MZ(2, 0.01000000001)]
                ),
                sp.Detector([sp.MeasurementRecord(-1)]),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(0, 0.02000000001),
                        sp.noise_channels.CorrelatedError(sp.PauliX(1), 0.01000000001),
                        sp.noise_channels.PauliZError(2, 0.02000000001),
                        sp.noise_channels.ElseCorrelatedError(
                            sp.PauliZ(2), 0.03000000001
                        ),
                        sp.noise_channels.Depolarise2(3, 4, 0.02000000001),
                    ]
                ),
                sp.Observable(0, [sp.MeasurementRecord(-1)]),
                sp.ShiftCoordinates((0, 0, 1)),
            ]
        )
        assert circuit1.approx_equals(circuit2)

    @pytest.mark.parametrize(("rel_tol", "abs_tol"), [(1e-7, 0.0), (1e-9, 1e-9)])
    def test_approx_equal_circuits_are_approx_equal_other_tol(self, rel_tol, abs_tol):
        circuit1 = sp.Circuit(
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.Z(1), sp.gates.MZ(2, 0.01)]),
                sp.Detector([sp.MeasurementRecord(-1)]),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(0, 0.02),
                        sp.noise_channels.CorrelatedError(sp.PauliX(1), 0.01),
                        sp.noise_channels.PauliZError(2, 0.002),
                        sp.noise_channels.ElseCorrelatedError(sp.PauliZ(2), 0.03),
                        sp.noise_channels.Depolarise2(3, 4, 0.02),
                    ]
                ),
                sp.Observable(0, [sp.MeasurementRecord(-1)]),
                sp.ShiftCoordinates((0, 0, 1)),
            ]
        )
        circuit2 = sp.Circuit(
            [
                sp.GateLayer(
                    [sp.gates.X(0), sp.gates.Z(1), sp.gates.MZ(2, 0.010000001)]
                ),
                sp.Detector([sp.MeasurementRecord(-1)]),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(0, 0.020000001),
                        sp.noise_channels.CorrelatedError(sp.PauliX(1), 0.010000001),
                        sp.noise_channels.PauliZError(2, 0.0020000001),
                        sp.noise_channels.ElseCorrelatedError(
                            sp.PauliZ(2), 0.030000001
                        ),
                        sp.noise_channels.Depolarise2(3, 4, 0.020000001),
                    ]
                ),
                sp.Observable(0, [sp.MeasurementRecord(-1)]),
                sp.ShiftCoordinates((0, 0, 1)),
            ]
        )
        assert circuit1.approx_equals(circuit2, rel_tol=rel_tol, abs_tol=abs_tol)

    @pytest.mark.parametrize(("rel_tol", "abs_tol"), [(1e-7, 0.0), (1e-9, 1e-9)])
    def test_not_approx_equal_circuits_are_not_approx_equal_other_tol(
        self, rel_tol, abs_tol
    ):
        circuit1 = sp.Circuit(
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.Z(1), sp.gates.MZ(2, 0.01)]),
                sp.Detector([sp.MeasurementRecord(-1)]),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(0, 0.02),
                        sp.noise_channels.CorrelatedError(sp.PauliX(1), 0.01),
                        sp.noise_channels.PauliZError(2, 0.002),
                        sp.noise_channels.ElseCorrelatedError(sp.PauliZ(2), 0.03),
                        sp.noise_channels.Depolarise2(3, 4, 0.02),
                    ]
                ),
                sp.Observable(0, [sp.MeasurementRecord(-1)]),
                sp.ShiftCoordinates((0, 0, 1)),
            ]
        )
        circuit2 = sp.Circuit(
            [
                sp.GateLayer(
                    [sp.gates.X(0), sp.gates.Z(1), sp.gates.MZ(2, 0.01000001)]
                ),
                sp.Detector([sp.MeasurementRecord(-1)]),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(0, 0.020000001),
                        sp.noise_channels.CorrelatedError(sp.PauliX(1), 0.010000001),
                        sp.noise_channels.PauliZError(2, 0.0020000001),
                        sp.noise_channels.ElseCorrelatedError(
                            sp.PauliZ(2), 0.030000001
                        ),
                        sp.noise_channels.Depolarise2(3, 4, 0.020000001),
                    ]
                ),
                sp.Observable(0, [sp.MeasurementRecord(-1)]),
                sp.ShiftCoordinates((0, 0, 1)),
            ]
        )
        assert not circuit1.approx_equals(circuit2, rel_tol=rel_tol, abs_tol=abs_tol)


def test_two_circuits_with_identical_layers_are_equal():
    circuit1 = sp.Circuit(sp.GateLayer(sp.gates.X(sp.Qubit(i))) for i in range(10))
    circuit2 = sp.Circuit(sp.GateLayer(sp.gates.X(sp.Qubit(i))) for i in range(10))
    assert circuit1 == circuit2


@pytest.mark.parametrize(
    ("layers1", "layers2"),
    [
        (sp.GateLayer(sp.gates.H(sp.Qubit(0))), sp.GateLayer(sp.gates.H(sp.Qubit(1)))),
        (
            sp.GateLayer(sp.gates.S(sp.Qubit(0))),
            sp.GateLayer(sp.gates.C_ZYX(sp.Qubit(1))),
        ),
        (
            sp.GateLayer(sp.gates.X(sp.Qubit(0))),
            sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
        ),
        (
            sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
            sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.02)),
        ),
        (
            sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
            sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(1), 0.01)),
        ),
        (
            sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
            sp.NoiseLayer(
                sp.noise_channels.Depolarise2(sp.Qubit(0), sp.Qubit(1), 0.02)
            ),
        ),
    ],
)
def test_two_circuits_with_different_layers_are_not_equal(layers1, layers2):
    assert sp.Circuit(layers1) != sp.Circuit(layers2)


def test_two_circuits_with_different_number_of_iterations_are_not_equal():
    layers = [
        sp.GateLayer(sp.gates.X(sp.Qubit(0))),
        sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01)),
    ]
    assert sp.Circuit(layers, 2) != sp.Circuit(layers, 3)


def test_qubits_property_of_circuit_returns_all_qubits_in_each_layer():
    circuit = sp.Circuit(
        [
            sp.GateLayer([sp.gates.X(sp.Qubit(0)), sp.gates.Z(sp.Qubit(1))]),
            sp.NoiseLayer(
                [
                    sp.noise_channels.PauliXError(sp.Qubit(0), 0.02),
                    sp.noise_channels.Depolarise2(sp.Qubit(4), sp.Qubit(2), 0.01),
                    sp.noise_channels.CorrelatedError(sp.PauliX(sp.Qubit(1)), 0.05),
                ]
            ),
            sp.GateLayer([sp.gates.CX(sp.Qubit(2), sp.Qubit(3))]),
        ]
    )
    assert circuit.qubits == frozenset(
        (sp.Qubit(0), sp.Qubit(1), sp.Qubit(2), sp.Qubit(3), sp.Qubit(4))
    )


def test_circuit_can_return_the_measurement_gates_that_it_contains_and_preserve_their_ordering(
    empty_circuit: sp.Circuit,
):
    empty_circuit.append_layers(
        [
            sp.GateLayer(sp.gates.X(sp.Qubit(0))),
            sp.GateLayer(sp.gates.MRZ(sp.Qubit(0))),
            sp.GateLayer(sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
            sp.GateLayer([sp.gates.MX(sp.Qubit(1)), sp.gates.MX(sp.Qubit(0))]),
        ]
    )
    assert empty_circuit.measurement_gates == (
        sp.gates.MRZ(sp.Qubit(0)),
        sp.gates.MX(sp.Qubit(1)),
        sp.gates.MX(sp.Qubit(0)),
    )


def test_circuit_can_return_measurement_gates_from_within_nested_circuits(
    empty_circuit: sp.Circuit,
):
    empty_circuit.append_layers(
        [
            sp.GateLayer([sp.gates.MX(sp.Qubit(1)), sp.gates.MX(sp.Qubit(0))]),
            sp.Circuit(sp.Circuit(sp.GateLayer(sp.gates.MX(sp.Qubit(4), 0.01)), 2)),
        ]
    )
    assert empty_circuit.measurement_gates == (
        sp.gates.MX(sp.Qubit(1)),
        sp.gates.MX(sp.Qubit(0)),
        sp.gates.MX(sp.Qubit(4), 0.01),
    )


def test_appending_circuit_with_one_iteration_flattens_the_circuit(
    empty_circuit: sp.Circuit,
):
    nested_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in range(2)),
            sp.NoiseLayer(
                sp.noise_channels.Depolarise1(sp.Qubit(i), 0.001) for i in range(2)
            ),
        ]
    )
    empty_circuit.append_layers(nested_circuit)
    assert empty_circuit.layers == nested_circuit.layers


def test_appending_circuit_with_more_than_one_iteration_adds_the_circuit(
    empty_circuit: sp.Circuit,
):
    nested_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in range(2)),
            sp.NoiseLayer(
                sp.noise_channels.Depolarise1(sp.Qubit(i), 0.001) for i in range(2)
            ),
        ],
        iterations=20,
    )
    empty_circuit.append_layers(nested_circuit)
    assert empty_circuit.layers[0] == nested_circuit


def test_hash(empty_circuit: sp.Circuit):
    with pytest.raises(NotImplementedError):
        hash(empty_circuit)


@pytest.mark.parametrize(
    "body",
    [
        sp.GateLayer([sp.gates.X(0, tag="tag")]),
        sp.NoiseLayer([sp.noise_channels.Depolarise1(0, 0.1, tag="tag")]),
        sp.Detector(sp.MeasurementRecord(-1), tag="tag"),
        sp.Observable(0, sp.MeasurementRecord(-1), tag="tag"),
        sp.ShiftCoordinates((1, 1, 1), tag="tag"),
    ],
)
def test_circuit_with_tag_exports_does_not_raise(body) -> None:
    circuit_with_tags: sp.Circuit[int] = sp.Circuit()
    circuit_with_tags.append_layers([body])
    circuit_with_tags.as_stim_circuit()


class TestApplyingGateNoise:
    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_applying_empty_gate_noise_profile_doesnt_add_noise_layer(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        empty_circuit.append_layers(sp.GateLayer(sp.gates.X(0)))
        empty_circuit.apply_gate_noise({}, adjacency)
        assert not empty_circuit.is_noisy

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_applying_empty_gate_noise_profile_doesnt_remove_existing_noise(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        noise_layer = sp.NoiseLayer(sp.noise_channels.PauliXError(0, 0.001))
        empty_circuit.append_layers(noise_layer)
        empty_circuit.apply_gate_noise({}, adjacency)
        assert empty_circuit.noise_layers() == [noise_layer]

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_applying_gate_noise_doesnt_add_noise_when_gate_not_in_circuit(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        empty_circuit.append_layers(sp.GateLayer(sp.gates.X(0)))
        empty_circuit.apply_gate_noise(
            lambda noise_context: [
                sp.noise_channels.PauliZError(qubit, 0.01)
                for qubit in noise_context.gate_layer_qubits(sp.gates.Z)
            ],
            adjacency,
        )
        assert not empty_circuit.is_noisy

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_noise_profiles_can_return_single_noise_channels_that_are_not_iterable(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        empty_circuit.append_layers(sp.GateLayer(sp.gates.X(0)))
        empty_circuit.apply_gate_noise(
            lambda _: sp.noise_channels.PauliZError(0, 0.01), adjacency
        )
        assert empty_circuit.is_noisy

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_applying_gate_noise_doesnt_remove_gate_layers_when_gate_not_in_circuit(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        gate_layer = sp.GateLayer(sp.gates.X(0))
        empty_circuit.append_layers(gate_layer)
        empty_circuit.apply_gate_noise(
            lambda noise_context: [
                sp.noise_channels.PauliZError(qubit, 0.01)
                for qubit in noise_context.gate_layer_qubits(sp.gates.Z)
            ],
            adjacency,
        )
        assert empty_circuit.gate_layers() == [gate_layer]
        assert not empty_circuit.is_noisy

    @pytest.mark.parametrize(
        ("adjacency", "expected_noise_index"),
        [(sp.Circuit.LayerAdjacency.BEFORE, 0), (sp.Circuit.LayerAdjacency.AFTER, -1)],
    )
    def test_applying_gate_noise_adds_a_noise_layer_to_the_circuit(
        self, empty_circuit: sp.Circuit, adjacency, expected_noise_index
    ):
        empty_circuit.append_layers(sp.GateLayer(sp.gates.X(sp.Qubit(0))))
        empty_circuit.apply_gate_noise(
            lambda noise_context: [
                sp.noise_channels.PauliXError(qubit, 0.1)
                for qubit in noise_context.gate_layer_qubits(sp.gates.X)
            ],
            adjacency,
        )
        assert isinstance(empty_circuit.layers[expected_noise_index], sp.NoiseLayer)

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_noise_layer_compresses_consecutive_noise_channels_if_they_occur_across_different_qubits(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        empty_circuit.append_layers(
            sp.GateLayer([sp.gates.X(sp.Qubit(0)), sp.gates.Z(sp.Qubit(1))])
        )
        empty_circuit.apply_gate_noise(
            lambda noise_context: [
                sp.noise_channels.PauliXError(qubit, 0.01)
                for qubit in noise_context.gate_layer_qubits(sp.gates.X)
            ],
            lambda noise_context: [
                sp.noise_channels.PauliXError(qubit, 0.01)
                for qubit in noise_context.gate_layer_qubits(sp.gates.Z)
            ],
            adjacency,
        )
        assert len(empty_circuit.noise_layers()) == 1

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_consecutive_noise_layers_are_not_compressed_if_they_operate_on_the_same_qubit(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        empty_circuit.append_layers(
            [
                sp.GateLayer(sp.gates.X(sp.Qubit(0))),
                sp.GateLayer(sp.gates.Z(sp.Qubit(0))),
            ]
        )
        empty_circuit.apply_gate_noise(
            [
                lambda noise_context: [
                    sp.noise_channels.PauliXError(qubit, 0.01)
                    for qubit in noise_context.gate_layer_qubits(sp.gates.X)
                ],
                lambda noise_context: [
                    sp.noise_channels.PauliZError(qubit, 0.01)
                    for qubit in noise_context.gate_layer_qubits(sp.gates.Z)
                ],
            ],
            adjacency,
        )
        assert len(empty_circuit.noise_layers()) == 2

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_applying_gate_noise_recursively_applies_noise_to_nested_circuits(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        def noise_generator(noise_context):
            return [
                sp.noise_channels.PauliXError(qubit, 0.001)
                for qubit in noise_context.gate_layer_qubits(sp.gates.X)
            ]

        empty_circuit.append_layers(
            sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2)
        )
        empty_circuit.apply_gate_noise(noise_generator, adjacency, recursive=True)
        assert empty_circuit.layers[0].noise_layers()[0] == sp.NoiseLayer(
            sp.noise_channels.PauliXError(sp.Qubit(0), 0.001)
        )

    @pytest.mark.parametrize("adjacency", sp.Circuit.LayerAdjacency)
    def test_applying_gate_noise_non_recursively_doesnt_add_noise_to_nested_circuits(
        self, empty_circuit: sp.Circuit, adjacency
    ):
        empty_circuit.append_layers(
            sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2)
        )

        empty_circuit.apply_gate_noise(
            lambda noise_context: [
                sp.noise_channels.PauliXError(qubit, 0.01)
                for qubit in noise_context.gate_layer_qubits(sp.gates.X)
            ],
            adjacency,
            recursive=False,
        )
        assert not empty_circuit.layers[0].is_noisy

    @pytest.mark.parametrize(
        ("adjacency", "expected_circuit"),
        [
            (
                sp.Circuit.LayerAdjacency.AFTER,
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.X(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(0, 0.01)),
                        sp.GateLayer(sp.gates.Z(1)),
                        sp.NoiseLayer(sp.noise_channels.Depolarise1(1, 0.001)),
                    ]
                ),
            ),
            (
                sp.Circuit.LayerAdjacency.BEFORE,
                sp.Circuit(
                    [
                        sp.NoiseLayer(sp.noise_channels.PauliXError(0, 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.NoiseLayer(sp.noise_channels.Depolarise1(1, 0.001)),
                        sp.GateLayer(sp.gates.Z(1)),
                    ]
                ),
            ),
        ],
    )
    def test_applying_gate_noise_to_circuit_matches_expected_circuit(
        self, empty_circuit: sp.Circuit, adjacency, expected_circuit
    ):
        empty_circuit.append_layers(expected_circuit.gate_layers())
        empty_circuit.apply_gate_noise(
            [
                lambda noise_context: [
                    sp.noise_channels.PauliXError(qubit, 0.01)
                    for qubit in noise_context.gate_layer_qubits(sp.gates.X)
                ],
                lambda noise_context: [
                    sp.noise_channels.Depolarise1(qubit, 0.001)
                    for qubit in noise_context.gate_layer_qubits(sp.gates.Z)
                ],
            ],
            adjacency,
        )
        assert empty_circuit == expected_circuit


class TestReplaceGates:
    def test_replacing_gates_with_empty_replacement_policy_does_not_change_gates(
        self, empty_circuit: sp.Circuit
    ):
        gate_layers = [sp.GateLayer(sp.gates.MX(0))]
        empty_circuit.append_layers(gate_layers)
        empty_circuit.replace_gates({})
        assert empty_circuit.gate_layers() == gate_layers

    def test_replacing_gates_which_are_not_in_the_circuit_does_not_change_gates(
        self, empty_circuit: sp.Circuit
    ):
        gate_layers = [sp.GateLayer(sp.gates.X(0))]
        empty_circuit.append_layers(gate_layers)
        empty_circuit.replace_gates(
            {sp.gates.MZ: lambda gate: sp.gates.MZ(gate.qubit, 0.01)}
        )
        assert empty_circuit.gate_layers() == gate_layers

    def test_replacing_single_instance_of_gate_changes_that_gate_single_gate_layer(
        self, empty_circuit: sp.Circuit
    ):
        gate_to_replace = sp.gates.X(0)

        def gate_generator(gate):
            return sp.gates.Z(gate.qubit)

        empty_circuit.append_layers(sp.GateLayer(gate_to_replace))
        empty_circuit.replace_gates({gate_to_replace: gate_generator})
        assert empty_circuit.gate_layers() == [
            sp.GateLayer(gate_generator(gate_to_replace))
        ]

    def test_replacing_all_types_of_gate_changes_all_gates(
        self, empty_circuit: sp.Circuit
    ):
        gate_to_replace = sp.gates.X(0)

        def gate_generator(gate):
            return sp.gates.Z(gate.qubit)

        empty_circuit.append_layers(
            [sp.GateLayer(gate_to_replace), sp.GateLayer(gate_to_replace)]
        )
        empty_circuit.replace_gates({type(gate_to_replace): gate_generator})
        assert empty_circuit.gate_layers() == [
            sp.GateLayer(gate_generator(gate_to_replace)),
            sp.GateLayer(gate_generator(gate_to_replace)),
        ]

    def test_replace_gates_can_be_used_to_change_measurement_gate_probability(
        self, empty_circuit: sp.Circuit
    ):
        def noise_generator(gate):
            return sp.gates.MZ(gate.qubit, 0.001)

        gate = sp.gates.MZ(0)

        empty_circuit.append_layers(sp.GateLayer(gate))
        empty_circuit.replace_gates({type(gate): noise_generator})
        assert empty_circuit.measurement_gates[0] == noise_generator(gate)

    @pytest.mark.parametrize(
        ("layers", "replacement_policy", "expected_layers"),
        [
            (
                sp.Circuit(sp.GateLayer(sp.gates.MX(0)), iterations=2),
                {sp.gates.MX: lambda gate: sp.gates.MZ(gate.qubit)},
                [sp.Circuit(sp.GateLayer(sp.gates.MZ(0)), iterations=2)],
            ),
            (
                sp.Circuit(sp.GateLayer(sp.gates.MX(0)), iterations=2),
                {sp.gates.MX(0): lambda gate: sp.gates.MZ(gate.qubit)},
                [sp.Circuit(sp.GateLayer(sp.gates.MZ(0)), iterations=2)],
            ),
        ],
    )
    def test_replacing_gates_recursively_changes_gates_in_nested_circuits(
        self, empty_circuit: sp.Circuit, layers, replacement_policy, expected_layers
    ):
        empty_circuit.append_layers(layers)
        empty_circuit.replace_gates(replacement_policy, recursive=True)
        assert empty_circuit.layers == expected_layers

    @pytest.mark.parametrize(
        ("nested_circuit", "replacement_policy"),
        [
            (
                sp.Circuit(sp.GateLayer(sp.gates.MX(0)), iterations=2),
                {sp.gates.MX: lambda gate: sp.gates.MZ(gate.qubit)},
            ),
            (
                sp.Circuit(sp.GateLayer(sp.gates.MX(0)), iterations=2),
                {sp.gates.MX(0): lambda gate: sp.gates.MZ(gate.qubit)},
            ),
        ],
    )
    def test_replacing_gates_non_recursively_doesnt_change_nested_circuits(
        self, empty_circuit: sp.Circuit, nested_circuit, replacement_policy
    ):
        empty_circuit.append_layers(nested_circuit)
        empty_circuit.replace_gates(replacement_policy, recursive=False)
        assert empty_circuit.layers == [nested_circuit]

    @pytest.mark.parametrize(
        ("deltakit_circuit_circuit", "noise_to_apply", "expected_gate_layers"),
        [
            (
                sp.Circuit(sp.GateLayer(sp.gates.MZ(i) for i in range(4))),
                {sp.gates.MZ: lambda gate: sp.gates.MZ(gate.qubit, 0.001)},
                [sp.GateLayer(sp.gates.MZ(sp.Qubit(i), 0.001) for i in range(4))],
            ),
            (
                sp.Circuit(sp.GateLayer(sp.gates.MX(0))),
                {sp.gates.MX: lambda gate: sp.gates.MX(gate.qubit, 0.073)},
                [sp.GateLayer(sp.gates.MX(sp.Qubit(0), 0.073))],
            ),
            (
                sp.Circuit(sp.GateLayer(sp.gates.MRZ(44))),
                {sp.gates.MRZ: lambda gate: sp.gates.MRZ(gate.qubit, 0.1)},
                [sp.GateLayer(sp.gates.MRZ(sp.Qubit(44), 0.1))],
            ),
            (
                sp.Circuit(
                    sp.GateLayer([sp.gates.X(0), sp.gates.MRZ(44), sp.gates.Z(5)])
                ),
                {sp.gates.MRZ: lambda gate: sp.gates.MRZ(gate.qubit, 0.1)},
                [
                    sp.GateLayer(
                        [sp.gates.X(0), sp.gates.MRZ(sp.Qubit(44), 0.1), sp.gates.Z(5)]
                    )
                ],
            ),
            (
                sp.Circuit(
                    sp.GateLayer(
                        [
                            sp.gates.MPP(
                                sp.MeasurementPauliProduct(
                                    [sp.PauliZ(sp.Qubit(1)), sp.PauliX(sp.Qubit(3))]
                                )
                            )
                        ]
                    )
                ),
                {sp.gates.MPP: lambda gate: sp.gates.MPP(gate.pauli_product, 0.01)},
                [
                    sp.GateLayer(
                        [
                            sp.gates.MPP(
                                sp.MeasurementPauliProduct(
                                    [sp.PauliZ(sp.Qubit(1)), sp.PauliX(sp.Qubit(3))]
                                ),
                                0.01,
                            )
                        ]
                    )
                ],
            ),
            (
                sp.Circuit(
                    [sp.GateLayer(sp.gates.MX(0)), sp.GateLayer(sp.gates.MRZ(4))]
                ),
                {
                    sp.gates.MRZ: lambda gate: sp.gates.MRZ(gate.qubit, 0.43),
                    sp.gates.MX: lambda gate: sp.gates.MX(gate.qubit, 0.01),
                },
                [
                    sp.GateLayer(sp.gates.MX(0, 0.01)),
                    sp.GateLayer(sp.gates.MRZ(4, 0.43)),
                ],
            ),
        ],
    )
    def test_replacing_one_qubit_measurement_gates_introduces_noisy_measurements(
        self, deltakit_circuit_circuit, noise_to_apply, expected_gate_layers
    ):
        deltakit_circuit_circuit.replace_gates(noise_to_apply)
        assert deltakit_circuit_circuit.gate_layers() == expected_gate_layers


class TestReorderingDetectors:
    def test_reordering_detectors_in_correct_order_does_not_modify_layer_order(self):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 0)),
            sp.Detector(sp.MeasurementRecord(-2), sp.Coordinate(0, 1)),
            sp.Detector(sp.MeasurementRecord(-3), sp.Coordinate(1, 1)),
        ]
        layers = [sp.GateLayer(sp.gates.MZ(i) for i in range(3)), *detectors]
        deltakit_circuit_circuit = sp.Circuit(layers)
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.layers == layers

    def test_other_layers_between_blocks_are_not_modified_when_reordering_detectors(
        self,
    ):
        layers = [
            sp.GateLayer(sp.gates.MZ(i) for i in range(2)),
            sp.GateLayer(sp.gates.I(i) for i in range(2)),
        ]
        deltakit_circuit_circuit = sp.Circuit(layers)
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.layers == layers

    def test_reordering_detectors_without_coordinates_using_default_ordering_does_nothing(
        self,
    ):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1)),
            sp.Detector(sp.MeasurementRecord(-2)),
        ]
        deltakit_circuit_circuit = sp.Circuit(
            [sp.GateLayer(sp.gates.MZ(i) for i in range(2)), *detectors]
        )
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.detectors() == detectors

    def test_reordering_detectors_not_between_measurements_puts_them_in_correct_order(
        self,
    ):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 1, 1)),
            sp.Detector(sp.MeasurementRecord(-2), sp.Coordinate(0, 1, 0)),
            sp.Detector(sp.MeasurementRecord(-3), sp.Coordinate(0, 0, 0)),
        ]
        deltakit_circuit_circuit = sp.Circuit(
            [sp.GateLayer(sp.gates.MZ(i) for i in range(3)), *detectors]
        )
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.detectors() == [
            detectors[2],
            detectors[1],
            detectors[0],
        ]

    def test_reordering_detectors_between_measurements_only_reorders_within_the_block(
        self,
    ):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1), coordinate=sp.Coordinate(2, 1)),
            sp.Detector(sp.MeasurementRecord(-2), coordinate=sp.Coordinate(2, 0)),
            sp.Detector(sp.MeasurementRecord(-1), coordinate=sp.Coordinate(1, 0)),
            sp.Detector(sp.MeasurementRecord(-2), coordinate=sp.Coordinate(0, 0)),
        ]
        deltakit_circuit_circuit = sp.Circuit(
            [
                sp.GateLayer(sp.gates.MZ(i) for i in range(2)),
                *detectors[0:2],
                sp.GateLayer(sp.gates.MZ(i) for i in range(2)),
                *detectors[2:4],
            ]
        )
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.detectors() == [
            detectors[1],
            detectors[0],
            detectors[3],
            detectors[2],
        ]

    def test_reordering_detectors_given_a_different_key_puts_detectors_in_correct_order(
        self,
    ):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1), coordinate=sp.Coordinate(0, 0)),
            sp.Detector(sp.MeasurementRecord(-2), coordinate=sp.Coordinate(0, 1)),
            sp.Detector(sp.MeasurementRecord(-1), coordinate=sp.Coordinate(1, 0)),
        ]
        deltakit_circuit_circuit = sp.Circuit(
            [sp.GateLayer(sp.gates.MZ(i) for i in range(3)), *detectors]
        )
        deltakit_circuit_circuit.reorder_detectors(reverse=True)
        assert deltakit_circuit_circuit.detectors() == [
            detectors[2],
            detectors[1],
            detectors[0],
        ]

    def test_detectors_within_nested_circuit_are_reordered(self):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 1)),
            sp.Detector(sp.MeasurementRecord(-2), sp.Coordinate(1, 0)),
            sp.Detector(sp.MeasurementRecord(-3), sp.Coordinate(0, 0)),
        ]
        deltakit_circuit_circuit = sp.Circuit(
            sp.Circuit(
                [sp.GateLayer(sp.gates.MZ(i) for i in range(3)), *detectors],
                iterations=5,
            )
        )
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.detectors() == [
            detectors[2],
            detectors[0],
            detectors[1],
        ]

    def test_detectors_outside_and_inside_nested_circuits_are_reordered_separately(
        self,
    ):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-2), sp.Coordinate(0, 3)),
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 1)),
            sp.Detector(sp.MeasurementRecord(-2), sp.Coordinate(0, 4)),
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 2)),
        ]
        deltakit_circuit_circuit = sp.Circuit(
            [
                sp.GateLayer(sp.gates.MZ(i) for i in range(2)),
                *detectors[0:2],
                sp.Circuit(
                    [sp.GateLayer(sp.gates.MX(i) for i in range(2)), *detectors[2:]],
                    iterations=5,
                ),
            ]
        )
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.detectors() == [
            detectors[1],
            detectors[0],
            detectors[3],
            detectors[2],
        ]

    def test_detectors_are_not_reordered_across_shift_coordinate_shift_instructions(
        self,
    ):
        detectors = [
            sp.Detector(sp.MeasurementRecord(-2), sp.Coordinate(0, 4)),
            sp.Detector(sp.MeasurementRecord(-2), sp.Coordinate(0, 3)),
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 2)),
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 1)),
        ]
        deltakit_circuit_circuit = sp.Circuit(
            [*detectors[0:2], sp.ShiftCoordinates((0, 1)), *detectors[2:4]]
        )
        deltakit_circuit_circuit.reorder_detectors()
        assert deltakit_circuit_circuit.detectors() == [
            detectors[1],
            detectors[0],
            detectors[3],
            detectors[2],
        ]


class TestStimCircuit:
    @pytest.mark.parametrize(
        ("deltakit_circuit_circuit", "expected_circuit"),
        [
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.X(sp.Qubit(0))),
                        sp.GateLayer(sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
                        sp.GateLayer(sp.gates.MZ(sp.Qubit(1))),
                    ]
                ),
                stim.Circuit("X 0\nTICK\nCNOT 0 1\nTICK\nMZ 1"),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.MPP(sp.PauliX(0))),
                    ]
                ),
                stim.Circuit("MPP X0"),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(
                            sp.gates.MPP(sp.MeasurementPauliProduct(sp.PauliX(0)))
                        ),
                    ]
                ),
                stim.Circuit("MPP X0"),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(
                            sp.gates.MPP(sp.MeasurementPauliProduct([sp.PauliX(0)]))
                        ),
                    ]
                ),
                stim.Circuit("MPP X0"),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.MPP([sp.PauliX(0), sp.PauliY(1)])),
                    ]
                ),
                stim.Circuit("MPP X0*Y1"),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(
                            sp.gates.MPP(
                                sp.MeasurementPauliProduct([sp.PauliX(0), sp.PauliY(1)])
                            )
                        ),
                    ]
                ),
                stim.Circuit("MPP X0*Y1"),
            ),
        ],
    )
    def test_noiseless_deltakit_circuit_circuit_has_correct_stim_representation(
        self, deltakit_circuit_circuit, expected_circuit
    ):
        assert deltakit_circuit_circuit.as_stim_circuit() == expected_circuit

    # @pytest.mark.parametrize("measurement_gate", [
    #     gate(sp.Qubit(0), 0.0)
    #     for gate in (sp.gates.MEASUREMENT_GATES - {sp.gates.MPP})
    # ])
    # def test_measurement_gate_with_zero_error_probability_does_not_add_error_to_stim(
    #         self, measurement_gate):
    #     deltakit_circuit_circuit = sp.Circuit(sp.GateLayer(measurement_gate))
    #     assert deltakit_circuit_circuit.as_stim_circuit() == stim.Circuit(
    #         f"{measurement_gate.stim_string} 0")

    def test_deltakit_circuit_circuit_can_be_converted_into_a_detector_error_model(
        self,
    ):
        stim_circuit = stim.Circuit("""
            X_ERROR(0.125) 0
            X_ERROR(0.25) 1
            CORRELATED_ERROR(0.375) X0 X1
            M 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]
        """)
        assert (
            stim_circuit.detector_error_model()
            == sp.Circuit.from_stim_circuit(stim_circuit).as_detector_error_model()
        )

    def test_deltakit_circuit_circuit_circuits_wraps_all_the_dem_parameters_exposed_via_Stim(
        self,
    ):
        """This test does not contain an assertion. It simply tests that
        deltakit_circuit.Circuit.as_detector_error_model exposes the same arguments
        as stim.Circuit.detector_error_model"""
        stim_circuit = stim.Circuit("""
            X_ERROR(0.125) 0
            M 0
            DETECTOR rec[-1]
        """)
        sp.Circuit.from_stim_circuit(stim_circuit).as_detector_error_model(
            decompose_errors=True
        )
        sp.Circuit.from_stim_circuit(stim_circuit).as_detector_error_model(
            flatten_loops=True
        )
        sp.Circuit.from_stim_circuit(stim_circuit).as_detector_error_model(
            allow_gauge_detectors=True
        )
        sp.Circuit.from_stim_circuit(stim_circuit).as_detector_error_model(
            approximate_disjoint_errors=0.1
        )
        sp.Circuit.from_stim_circuit(stim_circuit).as_detector_error_model(
            ignore_decomposition_failures=True
        )
        sp.Circuit.from_stim_circuit(stim_circuit).as_detector_error_model(
            block_decomposition_from_introducing_remnant_edges=True
        )

    def test_that_value_error_is_raised_if_approximate_disjoint_error_is_not_specified_as_a_valid_probability(
        self,
    ):
        with pytest.raises(
            ValueError, match="approximate_disjoint_errors is not a valid probability"
        ):
            sp.Circuit(sp.GateLayer(sp.gates.H(0))).as_detector_error_model(
                approximate_disjoint_errors=1.2
            )

    def test_noisy_deltakit_circuit_circuit_has_correct_stim_representation(
        self, noisy_circuit
    ):
        assert noisy_circuit.as_stim_circuit() == stim.Circuit(
            "X 0\nTICK\nX_ERROR(0.01) 0\nCNOT 0 1\nTICK\nDEPOLARIZE2(0.02) 0 1\nMX(0.01) 0"
        )

    @pytest.mark.parametrize(
        ("deltakit_circuit_circuit", "expected_string"),
        [
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(sp.Qubit(0))),
                        sp.GateLayer(sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
                        sp.GateLayer(sp.gates.MZ(sp.Qubit(i)) for i in (0, 1)),
                        sp.Detector(sp.MeasurementRecord(-1)),
                    ]
                ),
                "DETECTOR rec[-1]",
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(sp.Qubit(0))),
                        sp.GateLayer(sp.gates.CX(sp.Qubit(0), sp.Qubit(1))),
                        sp.GateLayer(sp.gates.MZ(sp.Qubit(i)) for i in (0, 1)),
                        sp.Observable(0, sp.MeasurementRecord(-1)),
                    ]
                ),
                "OBSERVABLE_INCLUDE(0) rec[-1]",
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(sp.Qubit(0))),
                        sp.GateLayer(sp.gates.MZ(sp.Qubit(0))),
                        sp.Detector(sp.MeasurementRecord(-1)),
                        sp.ShiftCoordinates((0, 0, 1)),
                    ]
                ),
                "SHIFT_COORDS(0, 0, 1)",
            ),
        ],
    )
    def test_deltakit_circuit_circuit_with_single_annotation_ends_with_expected_annotation_string(
        self, deltakit_circuit_circuit: sp.Circuit, expected_string: str
    ):
        assert str(deltakit_circuit_circuit.as_stim_circuit()).endswith(expected_string)

    @pytest.mark.parametrize(
        ("deltakit_circuit_circuit", "qubit_mapping", "expected_string"),
        [
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(sp.Qubit(sp.Coordinate(0, 0)))),
                        sp.GateLayer(
                            sp.gates.CX(sp.Coordinate(0, 0), sp.Coordinate(0, 1))
                        ),
                        sp.GateLayer(sp.gates.MX(sp.Qubit(sp.Coordinate(0, 1)))),
                    ]
                ),
                {sp.Qubit(sp.Coordinate(0, 0)): 0, sp.Qubit(sp.Coordinate(0, 1)): 1},
                "QUBIT_COORDS(0, 1) 1\nQUBIT_COORDS(0, 0) 0\nH 0\nTICK\nCX 0 1\nTICK\nMX 1",
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(sp.Qubit(1))),
                        sp.GateLayer(sp.gates.CX(sp.Qubit(1), sp.Qubit(0))),
                        sp.GateLayer(sp.gates.MX(sp.Qubit(0))),
                    ]
                ),
                {sp.Qubit(1): 1, sp.Qubit(0): 0},
                "H 1\nTICK\nCX 1 0\nTICK\nMX 0",
            ),
        ],
    )
    def test_deltakit_circuit_circuit_with_mapping_gives_expected_stim_circuit(
        self, deltakit_circuit_circuit: sp.Circuit, qubit_mapping, expected_string
    ):
        assert (
            str(deltakit_circuit_circuit.as_stim_circuit(qubit_mapping))
            == expected_string
        )

    def test_repeated_circuit_can_be_converted_into_correct_stim_circuit(self):
        circuit = sp.Circuit(sp.GateLayer(sp.gates.X(sp.Qubit(0))), iterations=4)
        assert circuit.as_stim_circuit() == stim.Circuit("REPEAT 4 {\nX 0\n}")

    def test_deltakit_circuit_circuit_with_single_repeat_doesnt_create_stim_repeat_block(
        self,
    ):
        circuit = sp.Circuit(sp.GateLayer(sp.gates.X(sp.Qubit(0))), iterations=1)
        assert circuit.as_stim_circuit() == stim.Circuit("X 0")

    def test_circuit_with_multiple_repeat_blocks_can_be_converted_to_stim(self):
        circuit = sp.Circuit(
            [
                sp.GateLayer([sp.gates.X(sp.Qubit(0))]),
                sp.Circuit(sp.GateLayer(sp.gates.CX(0, 1)), 3),
                sp.Circuit(sp.GateLayer(sp.gates.CZ(0, 1)), 5),
            ]
        )
        assert circuit.as_stim_circuit() == stim.Circuit("""
            X 0
            TICK
            REPEAT 3 {
                CX 0 1
            }
            REPEAT 5 {
                CZ 0 1
            }
        """)

    def test_circuit_with_nested_repeats_can_be_converted_to_stim_circuit(self):
        circuit = sp.Circuit(
            [
                sp.GateLayer([sp.gates.X(sp.Qubit(0))]),
                sp.Circuit([sp.GateLayer(sp.gates.CZ(0, 1))], 3),
            ],
            5,
        )
        assert circuit.as_stim_circuit() == stim.Circuit("""
            REPEAT 5 {
                X 0
                TICK
                REPEAT 3 {
                    CZ 0 1
                }
            }
        """)

    def test_exception_raised_if_layers_are_appended_with_non_homogenous_qubit_uid_types(
        self,
    ):
        with pytest.raises(
            TypeError,
            match="All Qubit._unique_identifier fields must be of the same type",
        ):
            sp.Circuit([sp.GateLayer(sp.gates.X((0, 0))), sp.GateLayer(sp.gates.MZ(1))])

    @pytest.mark.parametrize(
        "deltakit_circuit_circuit",
        [
            sp.Circuit([sp.GateLayer(sp.gates.X(0)), sp.GateLayer(sp.gates.X(1))]),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.X(i) for i in range(2)),
                    sp.GateLayer(sp.gates.MZ(i) for i in range(2)),
                    sp.Observable(
                        0, [sp.MeasurementRecord(-1), sp.MeasurementRecord(-2)]
                    ),
                ]
            ),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.CX(0, 1)),
                    sp.GateLayer(sp.gates.MRX(1)),
                    sp.Detector(sp.MeasurementRecord(-1)),
                ]
            ),
        ],
    )
    def test_ticks_are_not_placed_after_final_gate_layer_in_stim(
        self, deltakit_circuit_circuit: sp.Circuit
    ):
        stim_lines = str(deltakit_circuit_circuit.as_stim_circuit()).split("\n")
        stim_identifiers = [line.split()[0] for line in stim_lines][::-1]
        # Count backwards through the stim identifiers and get the first one
        # that is a gate.
        last_gate_line_index = next(
            index
            for index, stim_id in reversed(list(enumerate(stim_identifiers)))
            if stim_id in sp.gates.GATE_MAPPING
        )
        with contextlib.suppress(IndexError):
            assert stim_lines[last_gate_line_index + 1] != "TICK"

    def test_stim_circuit_preserves_ordering_of_measurement_gates(
        self, empty_circuit: sp.Circuit
    ):
        empty_circuit.append_layers(
            sp.GateLayer([sp.gates.MX(0), sp.gates.MZ(1), sp.gates.MX(2)])
        )
        assert empty_circuit.as_stim_circuit() == stim.Circuit("MX 0\nMZ 1\nMX 2")


class TestQubitTransforms:
    def test_qubits_in_gate_layers_are_changed_according_to_mapping(self):
        circuit = sp.Circuit(
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.X(1)]),
                sp.GateLayer(sp.gates.CX(0, 1)),
                sp.GateLayer([sp.gates.MZ(0), sp.gates.MX(1)]),
            ]
        )
        circuit.transform_qubits({0: 2, 1: 3})
        assert circuit == sp.Circuit(
            [
                sp.GateLayer([sp.gates.X(2), sp.gates.X(3)]),
                sp.GateLayer(sp.gates.CX(2, 3)),
                sp.GateLayer([sp.gates.MZ(2), sp.gates.MX(3)]),
            ]
        )
        assert circuit.qubits == {sp.Qubit(2), sp.Qubit(3)}

    def test_qubits_in_noise_layer_are_changed_according_to_mapping(self):
        circuit = sp.Circuit(
            [
                sp.NoiseLayer(
                    [
                        sp.noise_channels.Depolarise1(sp.Qubit(0), 0.01),
                        sp.noise_channels.Depolarise2(sp.Qubit(1), sp.Qubit(2), 0.01),
                    ]
                ),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(sp.Qubit(0), 0.01),
                        sp.noise_channels.PauliYError(sp.Qubit(1), 0.02),
                        sp.noise_channels.PauliZError(sp.Qubit(2), 0.03),
                    ]
                ),
            ]
        )
        circuit.transform_qubits({0: 3, 1: 4, 2: 5})
        assert circuit == sp.Circuit(
            [
                sp.NoiseLayer(
                    [
                        sp.noise_channels.Depolarise1(sp.Qubit(3), 0.01),
                        sp.noise_channels.Depolarise2(sp.Qubit(4), sp.Qubit(5), 0.01),
                    ]
                ),
                sp.NoiseLayer(
                    [
                        sp.noise_channels.PauliXError(sp.Qubit(3), 0.01),
                        sp.noise_channels.PauliYError(sp.Qubit(4), 0.02),
                        sp.noise_channels.PauliZError(sp.Qubit(5), 0.03),
                    ]
                ),
            ]
        )
        assert circuit.qubits == {sp.Qubit(3), sp.Qubit(4), sp.Qubit(5)}

    def test_detector_coordinates_change_according_to_mapping(self):
        circuit = sp.Circuit(
            [
                sp.Detector(
                    [sp.MeasurementRecord(-3), sp.MeasurementRecord(-2)],
                    sp.Coordinate(0, 1, 2),
                ),
                sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 2, 2)),
            ]
        )
        circuit.transform_qubits(
            {
                sp.Coordinate(0, 1, 2): sp.Coordinate(0, 0, 0),
                sp.Coordinate(0, 2, 2): sp.Coordinate(0, 0, 1),
            }
        )
        assert circuit == sp.Circuit(
            [
                sp.Detector(
                    [sp.MeasurementRecord(-3), sp.MeasurementRecord(-2)],
                    sp.Coordinate(0, 0, 0),
                ),
                sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 0, 1)),
            ]
        )

    def test_qubits_in_nested_circuit_change_according_to_mapping(self):
        circuit = sp.Circuit(
            sp.Circuit(
                [
                    sp.GateLayer([sp.gates.X(0), sp.gates.SQRT_Y(1)]),
                    sp.NoiseLayer(sp.noise_channels.PauliYError(0, 0.02)),
                ],
                iterations=5,
            )
        )
        circuit.transform_qubits({0: 2, 1: 3})
        assert circuit == sp.Circuit(
            sp.Circuit(
                [
                    sp.GateLayer([sp.gates.X(2), sp.gates.SQRT_Y(3)]),
                    sp.NoiseLayer(sp.noise_channels.PauliYError(2, 0.02)),
                ],
                iterations=5,
            )
        )

    def test_different_layers_with_reference_to_same_qubit_doesnt_get_transformed_twice(
        self,
    ):
        qubit = sp.Qubit(0)
        circuit = sp.Circuit(
            [
                sp.GateLayer(sp.gates.X(qubit)),
                sp.NoiseLayer(sp.noise_channels.Depolarise1(qubit, 0.01)),
            ]
        )
        circuit.transform_qubits({0: 1, 1: 2})
        assert circuit.noise_layers()[0].qubits == {sp.Qubit(1)}
        assert circuit.gate_layers()[0].qubits == {sp.Qubit(1)}

    def test_transforming_all_coordinates_in_a_circuit_works_as_expected(self):
        circuit = sp.Circuit(
            [
                sp.GateLayer(sp.gates.X(sp.Coordinate(0, 0))),
                sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Coordinate(0, 0), 0.01)),
                sp.GateLayer(sp.gates.MZ(sp.Coordinate(0, 0))),
                sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 0)),
            ]
        )
        circuit.transform_qubits({sp.Coordinate(0, 0): sp.Coordinate(1, 1)})
        assert circuit == sp.Circuit(
            [
                sp.GateLayer(sp.gates.X(sp.Coordinate(1, 1))),
                sp.NoiseLayer(sp.noise_channels.Depolarise1(sp.Coordinate(1, 1), 0.01)),
                sp.GateLayer(sp.gates.MZ(sp.Coordinate(1, 1))),
                sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(1, 1)),
            ]
        )

    def test_error_is_raised_if_mapping_is_not_bijective(self):
        circuit = sp.Circuit()
        with pytest.raises(
            ValueError,
            match="The ID mapping is not bijective, all values must be unique.",
        ):
            circuit.transform_qubits({0: 2, 1: 2})


class TestFlatten:
    def test_flatten_for_trivial_circuit_works_as_expected(self, empty_circuit):
        assert empty_circuit.flatten() == empty_circuit

    def test_flatten_for_empty_circuit_with_repeats(self):
        assert sp.Circuit([], iterations=3).flatten() == sp.Circuit()

    @pytest.mark.parametrize(
        "layers",
        [
            [sp.GateLayer(sp.gates.X(0))],
            [
                sp.GateLayer(sp.gates.X(0)),
                sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
            ],
            [
                sp.GateLayer(sp.gates.X(0)),
                sp.GateLayer(sp.gates.MZ(0)),
            ],
            [
                sp.GateLayer(sp.gates.X(0)),
                sp.GateLayer(sp.gates.MZ(0)),
                sp.Detector([sp.MeasurementRecord(-1)]),
            ],
            [
                sp.GateLayer(sp.gates.X(0)),
                sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
                sp.GateLayer(sp.gates.MZ(0)),
                sp.Detector([sp.MeasurementRecord(-1)]),
            ],
            [sp.GateLayer([sp.gates.X(0), sp.gates.X(1)])],
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.X(1)]),
                sp.GateLayer([sp.gates.H(0), sp.gates.H(1)]),
            ],
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.X(1)]),
                sp.GateLayer([sp.gates.H(0), sp.gates.H(1)]),
                sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(0), 0.01)),
            ],
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.X(1)]),
                sp.GateLayer([sp.gates.H(0), sp.gates.H(1)]),
                sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                sp.GateLayer([sp.gates.MZ(0), sp.gates.MZ(1)]),
            ],
            [
                sp.GateLayer([sp.gates.X(0), sp.gates.X(1)]),
                sp.GateLayer([sp.gates.H(0), sp.gates.H(1)]),
                sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                sp.GateLayer([sp.gates.MZ(0), sp.gates.MZ(1)]),
                sp.Detector([sp.MeasurementRecord(-1)]),
            ],
            [
                sp.GateLayer([sp.gates.X(0)]),
                sp.GateLayer([sp.gates.H(0), sp.gates.H(1)]),
            ],
        ],
    )
    def test_circuit_with_no_nested_circuits_is_the_same_as_itself_when_flattened(
        self, layers
    ):
        assert sp.Circuit(layers).flatten() == sp.Circuit(layers)

    @pytest.mark.parametrize(
        ("layers", "iterations"),
        [
            ([sp.GateLayer(sp.gates.X(0))], 2),
            (
                [
                    sp.GateLayer(sp.gates.X(0)),
                    sp.GateLayer(sp.gates.H(0)),
                ],
                2,
            ),
            (
                [
                    sp.GateLayer(
                        [
                            sp.gates.X(0),
                            sp.gates.H(1),
                        ]
                    )
                ],
                2,
            ),
        ],
    )
    def test_flatten_for_iterating_circuits_works_as_expected(self, layers, iterations):
        assert sp.Circuit(layers, iterations).flatten() == sp.Circuit(
            layers * iterations
        )

    @pytest.mark.parametrize(
        ("circuit", "expected_circuit"),
        [
            (
                sp.Circuit(sp.Circuit(sp.GateLayer(sp.gates.X(0)))),
                sp.Circuit(sp.GateLayer(sp.gates.X(0))),
            ),
            (
                sp.Circuit(sp.Circuit(sp.GateLayer(sp.gates.X(0))), iterations=2),
                sp.Circuit([sp.GateLayer(sp.gates.X(0)), sp.GateLayer(sp.gates.X(0))]),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0))),
                    ]
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0))),
                    ],
                    iterations=2,
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.Circuit(
                            sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2)
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.Circuit(
                            sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                            iterations=2,
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.Circuit(
                            sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                            iterations=2,
                        ),
                    ],
                    iterations=2,
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.NoiseLayer(
                                    sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)
                                ),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.GateLayer(sp.gates.MZ(0)),
                                sp.Detector([sp.MeasurementRecord(-1)]),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.Circuit(sp.GateLayer(sp.gates.MZ(0)), iterations=3),
                                sp.Detector([sp.MeasurementRecord(-1)]),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.Observable(0, sp.MeasurementRecord(-1)),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.Observable(0, sp.MeasurementRecord(-1)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.GateLayer(sp.gates.MZ(0)),
                                sp.Detector([sp.MeasurementRecord(-1)]),
                                sp.Observable(0, sp.MeasurementRecord(-1)),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.Observable(0, sp.MeasurementRecord(-1)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.Observable(0, sp.MeasurementRecord(-1)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                        sp.ShiftCoordinates((0, 0, 1)),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.ShiftCoordinates((0, 0, 1)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.GateLayer(sp.gates.MZ(0)),
                                sp.Detector([sp.MeasurementRecord(-1)]),
                                sp.ShiftCoordinates((0, 0, 1)),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.ShiftCoordinates((0, 0, 1)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.ShiftCoordinates((0, 0, 1)),
                    ]
                ),
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.GateLayer(sp.gates.MZ(0)),
                                sp.Detector([sp.MeasurementRecord(-1)]),
                                sp.Observable(0, sp.MeasurementRecord(-1)),
                                sp.ShiftCoordinates((0, 0, 1)),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.Observable(0, sp.MeasurementRecord(-1)),
                        sp.ShiftCoordinates((0, 0, 1)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.MZ(0)),
                        sp.Detector([sp.MeasurementRecord(-1)]),
                        sp.Observable(0, sp.MeasurementRecord(-1)),
                        sp.ShiftCoordinates((0, 0, 1)),
                    ]
                ),
            ),
        ],
    )
    def test_flatten_for_nested_circuits_works_as_expected(
        self, circuit, expected_circuit
    ):
        assert circuit.flatten() == expected_circuit


class TestDetectorsGates:
    @pytest.mark.parametrize(
        ("circuit", "expected_detectors_gates"),
        [
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                        sp.GateLayer(gate_a := sp.gates.MZ(0)),
                        (det_a := sp.Detector([sp.MeasurementRecord(-1)])),
                    ],
                ),
                [(det_a, [gate_a])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(gate_a := sp.gates.MZ(0)),
                                sp.GateLayer(sp.gates.X(0)),
                            ],
                            iterations=2,
                        ),
                        (det_a := sp.Detector([sp.MeasurementRecord(-1)])),
                    ]
                ),
                [(det_a, [gate_a])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(gate_a := sp.gates.MZ(0)),
                        sp.GateLayer(sp.gates.MX(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(sp.GateLayer(sp.gates.X(0)), iterations=2),
                        (det_a := sp.Detector([sp.MeasurementRecord(-2)])),
                    ],
                ),
                [(det_a, [gate_a])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(gate_a := sp.gates.MZ(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        (det_a := sp.Detector([sp.MeasurementRecord(-1)])),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(gate_b := sp.gates.MZ(0)),
                        (det_b := sp.Detector([sp.MeasurementRecord(-1)])),
                    ]
                ),
                [(det_a, [gate_a]), (det_b, [gate_b])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.GateLayer(gate_a := sp.gates.MZ(0)),
                        sp.GateLayer(sp.gates.MX(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(sp.gates.X(0)),
                        sp.GateLayer(gate_b := sp.gates.MZ(0)),
                        (
                            det_a := sp.Detector(
                                [sp.MeasurementRecord(-1), sp.MeasurementRecord(-3)]
                            )
                        ),
                    ]
                ),
                [(det_a, [gate_b, gate_a])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(gate_a := sp.gates.MX(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.Circuit(
                                    sp.GateLayer(gate_b := sp.gates.MZ(0)),
                                    iterations=1,
                                ),
                                (
                                    det_a := sp.Detector(
                                        [
                                            sp.MeasurementRecord(-2),
                                            sp.MeasurementRecord(-1),
                                        ]
                                    )
                                ),
                            ],
                            iterations=1,
                        ),
                    ],
                ),
                [(det_a, [gate_a, gate_b])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(gate_a := sp.gates.MX(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.Circuit(
                                    sp.GateLayer(gate_b := sp.gates.MZ(0)),
                                    iterations=3,
                                ),
                                (
                                    det_a := sp.Detector(
                                        [
                                            sp.MeasurementRecord(-2),
                                            sp.MeasurementRecord(-1),
                                        ]
                                    )
                                ),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                [(det_a, [gate_b, gate_b])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(gate_a := sp.gates.MX(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.Circuit(
                                    [
                                        sp.GateLayer(gate_b := sp.gates.MZ(0)),
                                        sp.GateLayer(gate_a := sp.gates.MX(0)),
                                    ],
                                    iterations=3,
                                ),
                                (
                                    det_a := sp.Detector(
                                        [
                                            sp.MeasurementRecord(-2),
                                            sp.MeasurementRecord(-1),
                                        ]
                                    )
                                ),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                [(det_a, [gate_b, gate_a])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(gate_a := sp.gates.MX(0)),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(sp.gates.X(0)),
                                sp.Circuit(
                                    [
                                        sp.GateLayer(gate_b := sp.gates.MZ(0)),
                                        sp.GateLayer(gate_a := sp.gates.MX(0)),
                                        (
                                            det_a := sp.Detector(
                                                [
                                                    sp.MeasurementRecord(-2),
                                                    sp.MeasurementRecord(-1),
                                                ]
                                            )
                                        ),
                                    ],
                                    iterations=3,
                                ),
                            ],
                            iterations=2,
                        ),
                    ],
                ),
                [(det_a, [gate_b, gate_a])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(gate_a := sp.gates.MPP(sp.PauliX(sp.Qubit(2)))),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [sp.GateLayer(sp.gates.MX(0)), sp.GateLayer(sp.gates.X(0))],
                            iterations=2,
                        ),
                        (det_a := sp.Detector([sp.MeasurementRecord(-3)])),
                    ],
                ),
                [(det_a, [gate_a])],
            ),
            (
                sp.Circuit(
                    [
                        sp.GateLayer(sp.gates.H(0)),
                        sp.GateLayer(sp.gates.MPP(sp.PauliX(sp.Qubit(2)))),
                        sp.NoiseLayer(sp.noise_channels.PauliXError(sp.Qubit(1), 0.01)),
                        sp.Circuit(
                            [
                                sp.GateLayer(gate_a := sp.gates.MRZ(0)),
                                sp.GateLayer(sp.gates.X(0)),
                            ],
                            iterations=2,
                        ),
                        sp.GateLayer(sp.gates.MX(0)),
                        (det_a := sp.Detector([sp.MeasurementRecord(-2)])),
                    ],
                ),
                [(det_a, [gate_a])],
            ),
        ],
    )
    def test_detectors_gates_returns_correct_values_for_flat_circuit(
        self, circuit, expected_detectors_gates
    ):
        assert circuit.detectors_gates() == expected_detectors_gates

    def test_out_of_index_raised_for_detectors_gates(self):
        circuit = sp.Circuit(
            [
                sp.GateLayer(sp.gates.H(0)),
                sp.GateLayer(sp.gates.MX(0)),
                sp.Detector([sp.MeasurementRecord(-2)]),
            ],
        )
        with pytest.raises(IndexError):
            circuit.detectors_gates()
