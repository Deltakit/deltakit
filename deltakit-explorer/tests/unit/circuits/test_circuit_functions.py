# (c) Copyright Riverlane 2020-2025.

import pytest
from deltakit_circuit import (Circuit, Detector, GateLayer, MeasurementRecord,
                              NoiseLayer, Observable, ShiftCoordinates)
from deltakit_circuit.gates import CX, CZ, MZ, RZ, H, I, PauliBasis, S, X
from deltakit_circuit.noise_channels import Depolarise2
from deltakit_explorer.qpu._circuits import (merge_layers,
                                             parallelise_disjoint_circuits,
                                             parallelise_same_length_circuits,
                                             remove_identities)


def single_stabiliser(circuit_spec: tuple[PauliBasis, str]):
    if circuit_spec[1] == "":
        gate = CX if circuit_spec[0] == PauliBasis.X else CZ

        circuit = Circuit(
            [
                GateLayer(H(0)),
                GateLayer(gate(0, 1)),
                GateLayer(gate(0, 2)),
                GateLayer(gate(0, 3)),
                GateLayer(gate(0, 4)),
                GateLayer(H(0)),
            ]
        )
    elif circuit_spec == (PauliBasis.Z, "swap"):
        circuit = Circuit(
            [
                GateLayer(H(0)),
                GateLayer(CZ(1, 0)),
                GateLayer(CZ(2, 0)),
                GateLayer(CZ(3, 0)),
                GateLayer(CZ(4, 0)),
                GateLayer(H(0)),
            ]
        )
    else:
        msg = f"Unknown circuit spec {circuit_spec} supplied."
        raise ValueError(msg)

    return circuit


class TestParalleliseDisjointCircuits:
    @pytest.mark.parametrize(
        "circuits, expected_parallelised_circuit",
        [
            (
                [
                    Circuit(GateLayer(CX(0, 1))),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(GateLayer([CX(0, 1), CX(2, 3)])),
            ),
            (
                [
                    Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(
                    [
                        GateLayer(H(0)),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                    ]
                ),
            ),
            (
                [
                    Circuit(GateLayer(MZ(0))),
                    Circuit(
                        [
                            GateLayer(H(1)),
                            GateLayer(MZ(1)),
                            GateLayer(CX(2, 1)),
                        ]
                    ),
                ],
                Circuit(
                    [
                        GateLayer(H(1)),
                        GateLayer([MZ(0), MZ(1)]),
                        GateLayer(CX(2, 1)),
                    ]
                ),
            ),
            (
                [
                    Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                    Circuit(GateLayer(CX(2, 3))),
                    Circuit(GateLayer([H(4), H(5)])),
                ],
                Circuit(
                    [
                        GateLayer([H(0), H(4), H(5)]),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                    ]
                ),
            ),
            (
                [
                    Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                ],
                Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
            ),
            ([], Circuit()),
        ],
    )
    def test_parallelise_disjoint_circuits_with_only_gate_layers(
        self, circuits, expected_parallelised_circuit
    ):
        parallelised_circuit = parallelise_disjoint_circuits(circuits)
        assert parallelised_circuit == expected_parallelised_circuit

    @pytest.mark.parametrize(
        "circuits, expected_parallelised_circuit",
        [
            (
                [
                    Circuit([GateLayer(CX(0, 1)), ShiftCoordinates((0, 1))]),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(
                    [
                        GateLayer([CX(0, 1), CX(2, 3)]),
                        ShiftCoordinates((0, 1)),
                    ]
                ),
            ),
            (
                [
                    Circuit(GateLayer(CX(0, 1))),
                    Circuit([GateLayer(CX(2, 3)), ShiftCoordinates((0, 1))]),
                ],
                Circuit(
                    [
                        GateLayer([CX(0, 1), CX(2, 3)]),
                        ShiftCoordinates((0, 1)),
                    ]
                ),
            ),
            (
                [
                    Circuit([ShiftCoordinates((0, 1)), GateLayer(CX(0, 1))]),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(
                    [
                        ShiftCoordinates((0, 1)),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                    ]
                ),
            ),
            (
                [
                    Circuit(GateLayer(CX(0, 1))),
                    Circuit([ShiftCoordinates((0, 1)), GateLayer(CX(2, 3))]),
                ],
                Circuit(
                    [
                        ShiftCoordinates((0, 1)),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                    ]
                ),
            ),
            (
                [
                    Circuit(
                        [
                            GateLayer(MZ(0)),
                            Detector(MeasurementRecord(-1)),
                            GateLayer(CX(0, 1)),
                        ]
                    ),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(
                    [
                        GateLayer(MZ(0)),
                        Detector(MeasurementRecord(-1)),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                    ]
                ),
            ),
            (
                [
                    Circuit(
                        [
                            GateLayer(MZ(0)),
                            GateLayer(CX(0, 1)),
                            Detector(MeasurementRecord(-1)),
                        ]
                    ),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(
                    [
                        GateLayer(MZ(0)),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                        Detector(MeasurementRecord(-1)),
                    ]
                ),
            ),
            (
                [
                    Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                    Circuit([GateLayer(CX(2, 3)), ShiftCoordinates((0, 1))]),
                ],
                Circuit(
                    [
                        GateLayer(H(0)),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                        ShiftCoordinates((0, 1)),
                    ]
                ),
            ),
            (
                [
                    Circuit(
                        [
                            GateLayer(MZ(0)),
                            Detector(MeasurementRecord(-1)),
                            GateLayer(CX(0, 1)),
                            ShiftCoordinates((0, 1)),
                        ]
                    ),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(
                    [
                        GateLayer(MZ(0)),
                        Detector(MeasurementRecord(-1)),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                        ShiftCoordinates((0, 1)),
                    ]
                ),
            ),
        ],
    )
    def test_parallelise_disjoint_circuits_with_non_gate_layers(
        self, circuits, expected_parallelised_circuit
    ):
        parallelised_circuit = parallelise_disjoint_circuits(circuits)
        assert parallelised_circuit == expected_parallelised_circuit

    @pytest.mark.parametrize(
        "circuits, expected_parallelised_circuit",
        [
            (
                [
                    Circuit(
                        [
                            GateLayer(CX(0, 1)),
                            ShiftCoordinates((0, 1)),
                            Circuit(GateLayer(H(0)), iterations=2),
                        ]
                    ),
                    Circuit(GateLayer(CX(2, 3))),
                ],
                Circuit(
                    [
                        GateLayer([CX(0, 1), CX(2, 3)]),
                        ShiftCoordinates((0, 1)),
                        Circuit(GateLayer(H(0)), iterations=2),
                    ]
                ),
            ),
            (
                [
                    Circuit([GateLayer(CX(0, 1)), ShiftCoordinates((0, 1))]),
                    Circuit(
                        [Circuit(GateLayer(H(2)), iterations=2), GateLayer(CX(2, 3))]
                    ),
                ],
                Circuit(
                    [
                        Circuit(GateLayer(H(2)), iterations=2),
                        GateLayer([CX(0, 1), CX(2, 3)]),
                        ShiftCoordinates((0, 1)),
                    ]
                ),
            ),
        ],
    )
    def test_parallelise_disjoint_circuits_with_nested_circuits(
        self, circuits, expected_parallelised_circuit
    ):
        parallelised_circuit = parallelise_disjoint_circuits(circuits)
        assert parallelised_circuit == expected_parallelised_circuit

    def test_parallelise_disjoint_circuits_raises_error_if_overlapping_qubits(self):
        with pytest.raises(
            ValueError,
            match="Circuits to be parallelised do not act on distinct qubits.",
        ):
            parallelise_disjoint_circuits(
                [
                    Circuit(GateLayer(H(0))),
                    Circuit(GateLayer(CX(0, 1))),
                ]
            )

    def test_parallelise_disjoint_circuits_raises_error_with_unequal_iterations(self):
        with pytest.raises(
            ValueError,
            match="Circuits to be parallelised must have the same number of iterations.",
        ):
            parallelise_disjoint_circuits(
                [
                    Circuit(GateLayer(H(0)), iterations=2),
                    Circuit(GateLayer(CX(1, 2))),
                    Circuit(GateLayer(CX(3, 4))),
                ]
            )

    @pytest.mark.parametrize(
        "circuits",
        [
            [
                Circuit([GateLayer(H(0)), GateLayer(H(1))]),
                Circuit(NoiseLayer(Depolarise2(2, 3, 0.01))),
            ],
            [
                Circuit([GateLayer(H(0)), GateLayer(H(1))]),
                Circuit(NoiseLayer(Depolarise2(2, 3, 0.01))),
                Circuit([GateLayer(H(4)), GateLayer(H(5))]),
            ],
            [
                Circuit([GateLayer(H(0)), GateLayer(H(1))]),
                Circuit(Circuit(NoiseLayer(Depolarise2(2, 3, 0.01)), iterations=3)),
            ],
            [
                Circuit([GateLayer(H(0)), GateLayer(H(1))]),
                Circuit(
                    [
                        Circuit(GateLayer(H(0)), iterations=4),
                        Circuit(NoiseLayer(Depolarise2(2, 3, 0.01)), iterations=3),
                    ]
                ),
            ],
        ],
    )
    def test_parallelise_disjoint_circuits_raise_error_with_noise_layer(self, circuits):
        with pytest.raises(
            ValueError,
            match="Circuits to be parallelised may not contain NoiseLayers.",
        ):
            parallelise_disjoint_circuits(circuits)

    def test_parallelise_disjoint_circuits_raises_error_with_multiple_circuit_layers(
        self,
    ):
        circuits = [
            Circuit(
                [
                    GateLayer(H(0)),
                    GateLayer(H(1)),
                    Circuit([GateLayer(H(2))], iterations=2),
                ]
            ),
            Circuit(Circuit([GateLayer(H(2)), GateLayer(H(3))], iterations=2)),
        ]
        with pytest.raises(
            ValueError,
            match="Only one circuit to be parallelised can contain a nested Circuit.",
        ):
            parallelise_disjoint_circuits(circuits)

    @pytest.mark.parametrize(
        "circuits",
        [
            [
                Circuit([GateLayer(MZ(0)), Detector(MeasurementRecord(-1))]),
                Circuit([GateLayer(H(1)), GateLayer(MZ(2))]),
            ],
            [
                Circuit([GateLayer(MZ(0)), Detector(MeasurementRecord(-1))]),
                Circuit(Circuit([GateLayer(H(1)), GateLayer(MZ(2))], iterations=2)),
                Circuit(GateLayer(H(3))),
            ],
            [
                Circuit([GateLayer(MZ(0)), Detector(MeasurementRecord(-1))]),
                Circuit(
                    [
                        Circuit(GateLayer(H(1)), iterations=2),
                        Circuit(GateLayer({H(1), MZ(2)}), iterations=2),
                    ]
                ),
                Circuit(GateLayer(H(3))),
            ],
        ],
    )
    def test_parallelise_disjoint_circuits_raises_error_with_measurement_and_annotation(
        self, circuits
    ):
        with pytest.raises(
            ValueError,
            match="If one circuit to be parallelised contains annotations, "
            + "no other circuit can contain measurements.",
        ):
            parallelise_disjoint_circuits(circuits)

    @pytest.mark.parametrize(
        "circuits",
        [
            [
                Circuit([GateLayer(MZ(0)), Detector(MeasurementRecord(-1))]),
                Circuit([GateLayer(H(1)), ShiftCoordinates((0, 1))]),
            ],
            [
                Circuit(GateLayer(H(0))),
                Circuit(
                    Circuit([GateLayer(H(1)), ShiftCoordinates((0, 1))], iterations=4)
                ),
                Circuit([GateLayer(MZ(2)), Detector(MeasurementRecord(-1))]),
            ],
            [
                Circuit(GateLayer(H(0))),
                Circuit(
                    [
                        Circuit(GateLayer(H(1)), iterations=2),
                        Circuit(
                            [GateLayer(H(1)), ShiftCoordinates((0, 1))], iterations=4
                        ),
                    ]
                ),
                Circuit([GateLayer(MZ(2)), Detector(MeasurementRecord(-1))]),
            ],
        ],
    )
    def test_parallelise_disjoint_circuits_raises_error_with_multiple_annotations(
        self, circuits
    ):
        with pytest.raises(
            ValueError,
            match="Only one circuit to be parallelised can contain annotations",
        ):
            parallelise_disjoint_circuits(circuits)


class TestParalleliseSameLengthCircuits:
    @pytest.mark.parametrize(
        "circuits, expected_parallelised_circuit",
        [
            (
                [
                    Circuit([GateLayer([H(0), S(1)]), GateLayer(CX(2, 3))]),
                    Circuit([GateLayer(X(5)), GateLayer(CX(0, 1))]),
                    Circuit([GateLayer(H(2)), GateLayer(H(5))]),
                ],
                Circuit(
                    [
                        GateLayer([H(0), S(1), X(5), H(2)]),
                        GateLayer([CX(2, 3), CX(0, 1), H(5)]),
                    ]
                ),
            ),
            (
                [Circuit([GateLayer([H(0), S(1)]), GateLayer(CX(2, 3))])],
                Circuit([GateLayer([H(0), S(1)]), GateLayer(CX(2, 3))]),
            ),
            ([], Circuit()),
        ],
    )
    def test_parallelise_same_length_circuits(
        self, circuits, expected_parallelised_circuit
    ):
        actual_circuit = parallelise_same_length_circuits(circuits)
        assert actual_circuit == expected_parallelised_circuit

    def test_parallelise_same_length_circuits_raises_ValueError_if_different_lengths(
        self,
    ):
        circuit1 = Circuit([GateLayer({H(0), S(1)}), GateLayer(CX(2, 3))])
        circuit2 = Circuit([GateLayer(X(5))])
        circuits = [circuit1, circuit2]
        with pytest.raises(ValueError, match="Circuits must all be the same length."):
            parallelise_same_length_circuits(circuits)

    @pytest.mark.parametrize(
        "circuits",
        [
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        NoiseLayer(Depolarise2(0, 1, 0.01)),
                    ]
                ),
                Circuit([GateLayer(CX(2, 3)), GateLayer(CX(0, 1))]),
            ],
            [
                Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                Circuit([GateLayer(MZ(2)), Detector(MeasurementRecord(-1))]),
            ],
            [
                Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                Circuit([GateLayer(MZ(2)), Observable(0, MeasurementRecord(-1))]),
            ],
            [
                Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                Circuit([GateLayer(MZ(2)), ShiftCoordinates((0, 1))]),
            ],
            [
                Circuit([GateLayer(H(0)), GateLayer(CX(0, 1))]),
                Circuit(
                    [
                        GateLayer(MZ(2)),
                        Circuit(
                            [GateLayer(H(0)), GateLayer(CX(0, 1))],
                            iterations=3,
                        ),
                    ]
                ),
            ],
        ],
    )
    def test_parallelise_same_length_circuits_raises_error_with_non_gate_layers(
        self, circuits
    ):
        with pytest.raises(
            ValueError,
            match="Circuits can only be parallelised if they contain only GateLayers.",
        ):
            parallelise_same_length_circuits(circuits)

    def test_parallelise_same_length_circuits_raises_error_with_unequal_iterations(
        self,
    ):
        circuits = [
            Circuit([GateLayer([H(0), S(1)]), GateLayer(CX(2, 3))]),
            Circuit([GateLayer(X(5)), GateLayer(CX(0, 1))], iterations=5),
        ]
        with pytest.raises(
            ValueError,
            match="Circuits to be parallelised must have the same number of iterations.",
        ):
            parallelise_same_length_circuits(circuits)


class TestRemoveIdentities:
    @pytest.mark.parametrize(
        "input_circuit, expected_output_circuit",
        [
            (
                Circuit(
                    [
                        GateLayer([H(0), I(1), S(2)]),
                        GateLayer(CX(0, 1)),
                        GateLayer(I(1)),
                    ]
                ),
                Circuit(
                    [
                        GateLayer([H(0), S(2)]),
                        GateLayer(CX(0, 1)),
                    ]
                ),
            ),
            (
                Circuit(
                    [
                        GateLayer(I(1)),
                        GateLayer([H(0), S(2)]),
                        GateLayer([MZ(0), I(1)]),
                        Detector([MeasurementRecord(-1)]),
                        GateLayer(MZ(2)),
                        Observable(0, MeasurementRecord(-1)),
                        GateLayer([H(0), I(1), S(2)]),
                        ShiftCoordinates((1,)),
                        GateLayer([H(0), I(1), S(2)]),
                        GateLayer([I(0), I(1), I(2)]),
                    ],
                    iterations=7,
                ),
                Circuit(
                    [
                        GateLayer([H(0), S(2)]),
                        GateLayer([MZ(0)]),
                        Detector([MeasurementRecord(-1)]),
                        GateLayer(MZ(2)),
                        Observable(0, MeasurementRecord(-1)),
                        GateLayer([H(0), S(2)]),
                        ShiftCoordinates((1,)),
                        GateLayer([H(0), S(2)]),
                    ],
                    iterations=7,
                ),
            ),
            (
                Circuit(
                    [
                        GateLayer([H(0), S(2)]),
                        GateLayer([I(1), I(2)]),
                        Circuit(
                            [
                                GateLayer([H(0), S(2)]),
                                GateLayer([MZ(0), I(1)]),
                                Detector([MeasurementRecord(-1)]),
                                GateLayer(MZ(2)),
                            ],
                            iterations=3,
                        ),
                        Observable(0, MeasurementRecord(-1)),
                        GateLayer([H(0), I(1), S(2)]),
                        ShiftCoordinates((1,)),
                        Circuit(
                            [
                                Circuit(
                                    [
                                        GateLayer([H(0), S(2)]),
                                        GateLayer([I(1)]),
                                    ],
                                    iterations=4,
                                ),
                                Detector([MeasurementRecord(-1)]),
                                Circuit(GateLayer(I(0)), iterations=5),
                                GateLayer(MZ(2)),
                            ],
                            iterations=3,
                        ),
                        GateLayer([H(0), I(1), S(2)]),
                        GateLayer([I(0), I(1), I(2)]),
                    ],
                    iterations=7,
                ),
                Circuit(
                    [
                        GateLayer([H(0), S(2)]),
                        Circuit(
                            [
                                GateLayer([H(0), S(2)]),
                                GateLayer(MZ(0)),
                                Detector([MeasurementRecord(-1)]),
                                GateLayer(MZ(2)),
                            ],
                            iterations=3,
                        ),
                        Observable(0, MeasurementRecord(-1)),
                        GateLayer([H(0), S(2)]),
                        ShiftCoordinates((1,)),
                        Circuit(
                            [
                                Circuit(GateLayer([H(0), S(2)]), iterations=4),
                                Detector([MeasurementRecord(-1)]),
                                GateLayer(MZ(2)),
                            ],
                            iterations=3,
                        ),
                        GateLayer([H(0), S(2)]),
                    ],
                    iterations=7,
                ),
            ),
        ],
    )
    def test_remove_identities(self, input_circuit, expected_output_circuit):
        assert remove_identities(input_circuit) == expected_output_circuit


class TestMergeLayers:
    @pytest.mark.parametrize(
        "input_circuit, expected_output_circuit",
        [
            (
                Circuit(
                    [
                        GateLayer({H(0), H(1)}),
                        GateLayer(CX(1, 2)),
                        GateLayer(MZ(3)),
                        Detector([MeasurementRecord(-1)]),
                        GateLayer(RZ(4)),
                        GateLayer({H(0), H(1), H(2), H(3)}),
                    ]
                ),
                Circuit(
                    [
                        GateLayer({H(0), H(1)}),
                        GateLayer({CX(1, 2), MZ(3), RZ(4)}),
                        Detector([MeasurementRecord(-1)]),
                        GateLayer({H(0), H(1), H(2), H(3)}),
                    ]
                ),
            ),
            (
                Circuit(
                    [
                        Circuit(
                            [
                                GateLayer({H(0), H(1)}),
                                GateLayer(CX(1, 2)),
                                GateLayer(MZ(3)),
                                Detector([MeasurementRecord(-1)]),
                            ],
                            iterations=3,
                        ),
                        GateLayer(RZ(4)),
                        GateLayer({H(0), H(1), H(2), H(3)}),
                    ]
                ),
                Circuit(
                    [
                        Circuit(
                            [
                                GateLayer({H(0), H(1)}),
                                GateLayer({CX(1, 2), MZ(3)}),
                                Detector([MeasurementRecord(-1)]),
                            ],
                            iterations=3,
                        ),
                        GateLayer({RZ(4), H(0), H(1), H(2), H(3)}),
                    ]
                ),
            ),
        ],
    )
    def test_merge_layers_with_break_repeat_block_false(
        self, input_circuit, expected_output_circuit
    ):
        assert merge_layers(input_circuit) == expected_output_circuit

    @pytest.mark.parametrize(
        "input_circuit, expected_output_circuit",
        [
            (
                Circuit(
                    [
                        GateLayer({H(0), H(1)}),
                        GateLayer(CX(1, 2)),
                        GateLayer(MZ(3)),
                        Detector([MeasurementRecord(-1)]),
                        GateLayer(RZ(4)),
                        GateLayer({H(0), H(1), H(2), H(3)}),
                    ]
                ),
                Circuit(
                    [
                        GateLayer({H(0), H(1)}),
                        GateLayer({CX(1, 2), MZ(3), RZ(4)}),
                        Detector([MeasurementRecord(-1)]),
                        GateLayer({H(0), H(1), H(2), H(3)}),
                    ]
                ),
            ),
            (
                Circuit(
                    [
                        Circuit(
                            [
                                GateLayer({H(0), H(1)}),
                                GateLayer(CX(1, 2)),
                                GateLayer(MZ(3)),
                                Detector([MeasurementRecord(-1)]),
                            ],
                            iterations=3,
                        ),
                        GateLayer(MZ(4)),
                        GateLayer({H(0), H(1), H(2), H(3)}),
                        Circuit(
                            [
                                GateLayer({H(0), H(1)}),
                                GateLayer(CX(1, 2)),
                                GateLayer(MZ(3)),
                                Detector([MeasurementRecord(-1)]),
                            ],
                            iterations=3,
                        ),
                        GateLayer(MZ(4)),
                    ]
                ),
                Circuit(
                    [
                        Circuit(
                            [
                                GateLayer({H(0), H(1)}),
                                GateLayer({CX(1, 2), MZ(3)}),
                                Detector([MeasurementRecord(-1)]),
                            ],
                            iterations=3,
                        ),
                        GateLayer({MZ(4), H(0), H(1), H(2), H(3)}),
                        Circuit(
                            [
                                GateLayer({H(0), H(1)}),
                                GateLayer({CX(1, 2), MZ(3)}),
                                Detector([MeasurementRecord(-1)]),
                            ],
                            iterations=2,
                        ),
                        GateLayer({H(0), H(1)}),
                        GateLayer([CX(1, 2), MZ(3), MZ(4)]),
                        Detector([MeasurementRecord(-2)]),
                    ]
                ),
            ),
        ],
    )
    def test_merge_layers_with_break_repeat_block_true(
        self, input_circuit, expected_output_circuit
    ):
        assert merge_layers(input_circuit, True) == expected_output_circuit

    def test_merge_layers_with_noise_layer(self):
        layer0 = GateLayer(H(0))
        layer1 = NoiseLayer(Depolarise2(2, 3, 0.01))
        circuit = Circuit([layer0, layer1])
        message = "Layer merge cannot be carried out on a circuit with noise layers."
        with pytest.raises(ValueError, match=message):
            merge_layers(circuit)
