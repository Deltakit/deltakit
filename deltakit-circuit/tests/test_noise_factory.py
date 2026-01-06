# (c) Copyright Riverlane 2020-2025.
from copy import deepcopy

import pytest
import stim

import deltakit_circuit as sp


def test_noise_context_can_return_qubits_operated_on_by_a_one_qubit_gate_type_in_a_gate_layer():
    deltakit_circuit_circuit = sp.Circuit(sp.GateLayer([sp.gates.H(0), sp.gates.H(1)]))
    noise_context = sp.NoiseContext(
        deltakit_circuit_circuit, deltakit_circuit_circuit.layers[0]
    )
    assert list(noise_context.gate_layer_qubits(sp.gates.H)) == [
        sp.Qubit(0),
        sp.Qubit(1),
    ]


def test_noise_context_can_return_qubits_operated_on_by_a_two_qubit_gate_type_in_a_gate_layer():
    deltakit_circuit_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.CX(0, 1), sp.gates.CX(2, 3)])
    )
    noise_context = sp.NoiseContext(
        deltakit_circuit_circuit, deltakit_circuit_circuit.layers[0]
    )
    assert list(noise_context.gate_layer_qubits(sp.gates.CX)) == [
        sp.Qubit(0),
        sp.Qubit(1),
        sp.Qubit(2),
        sp.Qubit(3),
    ]


@pytest.mark.parametrize(
    ("input_layer", "gate_qubit_count", "expected_output_layers"),
    [
        (
            sp.GateLayer([sp.gates.CX(sp.SweepBit(0), 1), sp.gates.CX(2, 3)]),
            None,
            [sp.Qubit(1), sp.Qubit(2), sp.Qubit(3)],
        ),
        (
            sp.GateLayer([sp.gates.CX(sp.SweepBit(0), 1), sp.gates.CX(2, 3)]),
            2,
            [sp.Qubit(2), sp.Qubit(3)],
        ),
        (
            sp.GateLayer([sp.gates.CX(sp.SweepBit(0), 1), sp.gates.CX(2, 3)]),
            1,
            [sp.Qubit(1)],
        ),
    ],
)
def test_noise_context_can_return_qubits_selected_on_qubit_operand_count(
    input_layer, gate_qubit_count, expected_output_layers
):
    deltakit_circuit_circuit = sp.Circuit(input_layer)
    noise_context = sp.NoiseContext(
        deltakit_circuit_circuit, deltakit_circuit_circuit.layers[0]
    )
    assert (
        list(
            noise_context.gate_layer_qubits(
                gate_t=None, gate_qubit_count=gate_qubit_count
            )
        )
        == expected_output_layers
    )


@pytest.mark.parametrize(
    ("input_layer", "gate_t", "gate_qubit_count", "expected_output_layers"),
    [
        (
            sp.GateLayer(
                [
                    sp.gates.CX(sp.SweepBit(0), 1),
                    sp.gates.CX(2, 3),
                    sp.gates.H(4),
                    sp.gates.CY(5, 6),
                ]
            ),
            None,
            None,
            [sp.Qubit(i) for i in range(1, 7)],
        ),
        (
            sp.GateLayer(
                [sp.gates.CX(sp.SweepBit(0), 1), sp.gates.CX(2, 3), sp.gates.H(4)]
            ),
            (sp.gates.CX),
            1,
            [sp.Qubit(1)],
        ),
        (
            sp.GateLayer(
                [
                    sp.gates.CX(sp.SweepBit(0), 1),
                    sp.gates.CX(2, 3),
                    sp.gates.CZ(5, 6),
                    sp.gates.H(4),
                ]
            ),
            (sp.gates.CX),
            2,
            [sp.Qubit(2), sp.Qubit(3)],
        ),
        (
            sp.GateLayer(
                [
                    sp.gates.CX(sp.SweepBit(0), 1),
                    sp.gates.CX(2, 3),
                    sp.gates.CZ(5, 6),
                    sp.gates.H(4),
                ]
            ),
            (sp.gates.CX, sp.gates.H),
            1,
            [sp.Qubit(1), sp.Qubit(4)],
        ),
        (
            sp.GateLayer(
                [
                    sp.gates.CX(sp.SweepBit(0), 1),
                    sp.gates.CX(2, 3),
                    sp.gates.CZ(5, 6),
                    sp.gates.H(4),
                ]
            ),
            (sp.gates.X, sp.gates.CY),
            None,
            [],
        ),
        (
            sp.GateLayer(
                [
                    sp.gates.CX(sp.SweepBit(0), 1),
                    sp.gates.CX(2, 3),
                    sp.gates.CZ(5, 6),
                    sp.gates.H(4),
                ]
            ),
            None,
            3,
            [],
        ),
    ],
)
def test_gate_layer_qubits_selects_on_params(
    input_layer, gate_qubit_count, gate_t, expected_output_layers
):
    deltakit_circuit_circuit = sp.Circuit(input_layer)
    noise_context = sp.NoiseContext(
        deltakit_circuit_circuit, deltakit_circuit_circuit.layers[0]
    )
    assert (
        list(
            noise_context.gate_layer_qubits(
                gate_t=gate_t, gate_qubit_count=gate_qubit_count
            )
        )
        == expected_output_layers
    )


def test_can_apply_PauliXError_on_all_qubits_that_are_not_operated_on_by_a_H_gate():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.Y(qubit) for qubit in [1, 3]),
            sp.GateLayer(sp.gates.RX(qubit) for qubit in [4, 5, 8]),
            sp.GateLayer(sp.gates.H(qubit) for qubit in [2, 11, 16, 25]),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.H,
            inverse_noise_generator=sp.noise_channels.PauliXError.generator_from_prob(
                0.2
            ),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(
            [sp.noise_channels.PauliXError(qubit, 0.2) for qubit in [1, 3, 4, 5, 8]]
        )
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_can_apply_inverse_noise_in_a_two_qubit_gate_context():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.CY(1, 3)),
            sp.GateLayer(sp.gates.CZ(4, 5)),
            sp.GateLayer(sp.gates.CX.from_consecutive((2, 11, 16, 25))),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.CX,
            inverse_noise_generator=sp.noise_channels.PauliXError.generator_from_prob(
                0.2
            ),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(
            sp.noise_channels.PauliXError(qubit, 0.2) for qubit in [1, 3, 4, 5]
        )
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_can_apply_inverse_noise_in_addition_to_a_two_qubit_noise_channel_on_target_gate():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.CY(1, 3)),
            sp.GateLayer(sp.gates.CZ(4, 5)),
            sp.GateLayer(sp.gates.CX.from_consecutive((2, 11, 16, 25))),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.CX, sp.noise_channels.Depolarise2.generator_from_prob(0.2)
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(
            sp.noise_channels.Depolarise2.from_consecutive((2, 11, 16, 25), 0.2)
        )
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_can_apply_inverse_noise_given_a_multi_probability_noise_channel_context():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.CY(1, 3)),
            sp.GateLayer(sp.gates.CZ(4, 5)),
            sp.GateLayer(sp.gates.CX.from_consecutive((2, 11, 16, 25))),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.CX,
            sp.noise_channels.PauliChannel2.generator_from_prob(
                *tuple(0.02 for _ in range(15))
            ),
            sp.noise_channels.PauliChannel1.generator_from_prob(
                *tuple(0.01 for _ in range(3))
            ),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(
            list(
                sp.noise_channels.PauliChannel2.from_consecutive(
                    [2, 11, 16, 25], *(0.02 for _ in range(15))
                )
            )
            + [
                sp.noise_channels.PauliChannel1(qubit, *[0.01 for _ in range(3)])
                for qubit in [1, 3, 4, 5]
            ]
        )
    ]
    assert deltakit_circuit_circuit.noise_layers() == expected_noise_layers


def test_inverse_noise_is_not_added_to_the_wrong_qubits():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.CY(1, 3)),
            sp.GateLayer(sp.gates.CX.from_consecutive((2, 11, 16, 25))),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.CX,
            inverse_noise_generator=sp.noise_channels.PauliChannel1.generator_from_prob(
                *tuple(0.01 for _ in range(3))
            ),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_channels = [
        sp.noise_channels.PauliChannel1(qubit, *[0.01 for _ in range(3)])
        for qubit in [2, 11, 16, 15]
    ]
    assert all(
        channel not in deltakit_circuit_circuit.noise_layers()[0].noise_channels
        for channel in expected_noise_channels
    )


def test_inverse_noise_can_be_added_in_the_form_of_pauli_product_noise():
    deltakit_circuit_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.H(0), sp.gates.CX(1, 2)])
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.H,
            sp.noise_channels.ElseCorrelatedError.generator_from_prob(sp.PauliX, 0.2),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(sp.noise_channels.ElseCorrelatedError(sp.PauliX(0), 0.2))
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_applying_two_qubit_noise_channel_on_an_odd_number_of_qubits_raises_an_exception():
    deltakit_circuit_circuit = sp.Circuit(sp.GateLayer(sp.gates.H(0)))
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.H, sp.noise_channels.Depolarise2.generator_from_prob(0.2)
        )
    ]
    with pytest.raises(
        ValueError,
        match="Two qubit noise channels can only be "
        "constructed from an even number of qubits",
    ):
        deltakit_circuit_circuit.apply_gate_noise(
            noise_profile, sp.Circuit.LayerAdjacency.AFTER
        )


def test_can_apply_noise_channel_on_a_given_gate_and_another_noise_channel_on_all_other_qubits():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer(sp.gates.RX(qubit) for qubit in [1, 3, 4, 5, 8]),
            sp.GateLayer(sp.gates.H(qubit) for qubit in [2, 11, 16, 25]),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.H,
            sp.noise_channels.Depolarise1.generator_from_prob(0.1),
            sp.noise_channels.PauliXError.generator_from_prob(0.2),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(
            [sp.noise_channels.PauliXError(qubit, 0.2) for qubit in [1, 3, 4, 5, 8]]
            + [sp.noise_channels.Depolarise1(qubit, 0.1) for qubit in [2, 11, 16, 25]]
        )
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_inverted_noise_can_be_applied_to_all_qubits_that_are_not_acted_upon_in_each_gate_layer():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer(
                [sp.gates.RX(qubit) for qubit in [1, 3, 4, 5, 8]]
                + [sp.gates.CX(*qubit_pair) for qubit_pair in [(2, 0), (9, 11)]]
            ),
            sp.GateLayer(sp.gates.H(qubit) for qubit in range(15)),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            target_noise_generator=None,
            inverse_noise_generator=sp.noise_channels.PauliXError.generator_from_prob(
                0.2
            ),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(
            [
                sp.noise_channels.PauliXError(qubit, 0.2)
                for qubit in [6, 7, 10, 12, 13, 14]
            ]
        )
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_no_target_noise_generator_can_be_passed_through_to_inverted_noise_generator():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer([sp.gates.RX(qubit) for qubit in [0, 1, 2, 3, 4]]),
            sp.GateLayer(sp.gates.H(qubit) for qubit in range(15)),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.RX,
            inverse_noise_generator=sp.noise_channels.PauliXError.generator_from_prob(
                0.2
            ),
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer(
            [sp.noise_channels.PauliXError(qubit, 0.2) for qubit in range(5, 15)]
        )
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_no_inverse_noise_generator_can_be_passed_through_to_inverted_noise_generator():
    deltakit_circuit_circuit = sp.Circuit(
        [
            sp.GateLayer([sp.gates.RX(qubit) for qubit in [0, 1, 2, 3, 4]]),
            sp.GateLayer(sp.gates.H(qubit) for qubit in range(15)),
        ]
    )
    noise_profile = [
        sp.noise_profile_with_inverted_noise(
            sp.gates.RX, sp.noise_channels.PauliXError.generator_from_prob(0.2)
        )
    ]
    deltakit_circuit_circuit.apply_gate_noise(
        noise_profile, sp.Circuit.LayerAdjacency.AFTER
    )
    expected_noise_layers = [
        sp.NoiseLayer([sp.noise_channels.PauliXError(qubit, 0.2) for qubit in range(5)])
    ]
    assert expected_noise_layers == deltakit_circuit_circuit.noise_layers()


def test_adding_noise_to_all_measurement_gate_via_predefined_noise_profile():
    deltakit_circuit_circuit = sp.Circuit(
        sp.GateLayer(
            [
                sp.gates.MX(sp.Qubit(0)),
                sp.gates.MY(sp.Qubit(1)),
                sp.gates.MZ(sp.Qubit(2)),
                sp.gates.MRX(sp.Qubit(3)),
                sp.gates.MRY(sp.Qubit(4)),
                sp.gates.MRZ(sp.Qubit(5)),
                sp.gates.MPP(sp.MeasurementPauliProduct([sp.PauliZ(6), sp.PauliX(7)])),
            ]
        )
    )
    deltakit_circuit_circuit.replace_gates(sp.measurement_noise_profile(0.1))
    assert all(
        gate.probability == 0.1 for gate in deltakit_circuit_circuit.measurement_gates
    )


@pytest.mark.parametrize(
    ("inherited_gate_class", "init_data"),
    [
        (sp.gates.MX, (sp.Qubit(0),)),
        (sp.gates.MY, (sp.Qubit(0),)),
        (sp.gates.MZ, (sp.Qubit(0),)),
        (sp.gates.MRX, (sp.Qubit(0),)),
        (sp.gates.MRY, (sp.Qubit(0),)),
        (sp.gates.MRZ, (sp.Qubit(0),)),
        (sp.gates.MPP, (sp.PauliX(sp.Qubit(1)),)),
    ],
)
def test_measurement_noise_profile_uses_input_constructor(
    inherited_gate_class, init_data
):
    class GateInheritor(inherited_gate_class):
        pass

    reconstructed_gate = GateInheritor(*init_data)

    assert isinstance(
        sp.measurement_noise_profile(0.1)[inherited_gate_class](reconstructed_gate),
        GateInheritor,
    )


def test_measurement_noise_profile_does_not_change_qubit_id_type_for_single_qubit_measurement_gates():
    deltakit_circuit_circuit = sp.Circuit(
        sp.GateLayer(
            [
                sp.gates.MX(sp.Qubit(0)),
                sp.gates.MY(sp.Qubit(1)),
                sp.gates.MZ(sp.Qubit(2)),
                sp.gates.MRX(sp.Qubit(3)),
                sp.gates.MRY(sp.Qubit(4)),
                sp.gates.MRZ(sp.Qubit(5)),
            ]
        )
    )
    unchanged_deltakit_circuit_circuit = deepcopy(deltakit_circuit_circuit)
    deltakit_circuit_circuit.replace_gates(sp.measurement_noise_profile(0.1))
    for gate_layer, unchanged_gate_layer in zip(
        deltakit_circuit_circuit.gate_layers(),
        unchanged_deltakit_circuit_circuit.gate_layers(),
        strict=True,
    ):
        for gate, unchanged_gate in zip(
            gate_layer.gates, unchanged_gate_layer.gates, strict=True
        ):
            assert gate.qubit == unchanged_gate.qubit


def test_measurement_noise_profile_does_not_change_qubit_id_type_for_MPP_gate():
    deltakit_circuit_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.MPP([sp.PauliX(1), sp.PauliY(2)])])
    )
    unchanged_deltakit_circuit_circuit = deepcopy(deltakit_circuit_circuit)
    deltakit_circuit_circuit.replace_gates(sp.measurement_noise_profile(0.1))
    gate_layer = deltakit_circuit_circuit.gate_layers()[0]
    assert (
        gate_layer.qubits == unchanged_deltakit_circuit_circuit.gate_layers()[0].qubits
    )


def test_after_clifford_depolarisation_adds_a_noise_channel_for_each_clifford_gate():
    stim_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.H(0), sp.gates.CX(1, 2), sp.gates.MX(4)])
    )
    stim_circuit.apply_gate_noise(
        sp.after_clifford_depolarisation(0.1), sp.Circuit.LayerAdjacency.AFTER
    )
    assert len(stim_circuit.noise_layers()) == 1
    assert len(stim_circuit.noise_layers()[0].noise_channels) == 2


def test_after_clifford_depolarisation_adds_correct_noise_channels():
    stim_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.H(0), sp.gates.CX(1, 2), sp.gates.MX(4)])
    )
    stim_circuit.apply_gate_noise(
        sp.after_clifford_depolarisation(0.1), sp.Circuit.LayerAdjacency.AFTER
    )
    assert stim_circuit.noise_layers()[0].noise_channels == (
        sp.noise_channels.Depolarise1(0, 0.1),
        sp.noise_channels.Depolarise2(1, 2, 0.1),
    )


def test_before_measure_flip_probability_adds_an_error_for_each_measurement_operation():
    stim_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.MX(0), sp.gates.MRZ(1), sp.gates.H(5)])
    )
    stim_circuit.apply_gate_noise(
        sp.before_measure_flip_probability(0.1), sp.Circuit.LayerAdjacency.BEFORE
    )
    assert len(stim_circuit.noise_layers()) == 1
    assert len(stim_circuit.noise_layers()[0].noise_channels) == 2


def test_before_measure_flip_probability_adds_correct_errors():
    stim_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.MX(0), sp.gates.MRZ(1), sp.gates.H(5)])
    )
    stim_circuit.apply_gate_noise(
        sp.before_measure_flip_probability(0.1), sp.Circuit.LayerAdjacency.BEFORE
    )
    assert stim_circuit.noise_layers()[0].noise_channels == (
        sp.noise_channels.PauliXError(1, 0.1),
        sp.noise_channels.PauliZError(0, 0.1),
    )


def test_after_reset_flip_probability_adds_an_error_for_each_reset_operation():
    stim_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.RX(0), sp.gates.MRZ(1), sp.gates.H(4)])
    )
    stim_circuit.apply_gate_noise(
        sp.after_reset_flip_probability(0.1), sp.Circuit.LayerAdjacency.AFTER
    )
    assert len(stim_circuit.noise_layers()) == 1
    assert len(stim_circuit.noise_layers()[0].noise_channels) == 2


def test_after_reset_flip_probability_adds_correct_errors():
    stim_circuit = sp.Circuit(
        sp.GateLayer([sp.gates.RX(0), sp.gates.MRZ(1), sp.gates.H(5)])
    )
    stim_circuit.apply_gate_noise(
        sp.after_reset_flip_probability(0.1), sp.Circuit.LayerAdjacency.AFTER
    )
    assert stim_circuit.noise_layers()[0].noise_channels == (
        sp.noise_channels.PauliXError(1, 0.1),
        sp.noise_channels.PauliZError(0, 0.1),
    )


@pytest.mark.parametrize(
    ("clean_stim_circuit", "noisy_stim_circuit", "deltakit_circuit_noise_profile"),
    [
        (
            stim.Circuit.generated(
                "surface_code:rotated_memory_x", rounds=3, distance=3
            ),
            stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                rounds=3,
                distance=3,
                after_clifford_depolarization=0.333,
            ),
            (sp.after_clifford_depolarisation(0.333), sp.Circuit.LayerAdjacency.AFTER),
        ),
        (
            stim.Circuit.generated(
                "surface_code:rotated_memory_z", rounds=40, distance=40
            ),
            stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=40,
                distance=40,
                after_clifford_depolarization=0.01,
            ),
            (sp.after_clifford_depolarisation(0.01), sp.Circuit.LayerAdjacency.AFTER),
        ),
        (
            stim.Circuit.generated(
                "surface_code:unrotated_memory_z", rounds=20, distance=20
            ),
            stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                rounds=20,
                distance=20,
                after_clifford_depolarization=0.333,
            ),
            (sp.after_clifford_depolarisation(0.333), sp.Circuit.LayerAdjacency.AFTER),
        ),
        (
            stim.Circuit.generated("color_code:memory_xyz", rounds=5, distance=5),
            stim.Circuit.generated(
                "color_code:memory_xyz",
                rounds=5,
                distance=5,
                after_clifford_depolarization=0.333,
            ),
            (sp.after_clifford_depolarisation(0.333), sp.Circuit.LayerAdjacency.AFTER),
        ),
        (
            stim.Circuit.generated(
                "surface_code:rotated_memory_x", rounds=3, distance=3
            ),
            stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                rounds=3,
                distance=3,
                before_measure_flip_probability=0.333,
            ),
            (
                sp.before_measure_flip_probability(0.333),
                sp.Circuit.LayerAdjacency.BEFORE,
            ),
        ),
        (
            stim.Circuit.generated(
                "surface_code:rotated_memory_z", rounds=40, distance=40
            ),
            stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=40,
                distance=40,
                before_measure_flip_probability=0.01,
            ),
            (
                sp.before_measure_flip_probability(0.01),
                sp.Circuit.LayerAdjacency.BEFORE,
            ),
        ),
        (
            stim.Circuit.generated(
                "surface_code:unrotated_memory_z", rounds=20, distance=20
            ),
            stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                rounds=20,
                distance=20,
                before_measure_flip_probability=0.333,
            ),
            (
                sp.before_measure_flip_probability(0.333),
                sp.Circuit.LayerAdjacency.BEFORE,
            ),
        ),
        (
            stim.Circuit.generated("color_code:memory_xyz", rounds=5, distance=5),
            stim.Circuit.generated(
                "color_code:memory_xyz",
                rounds=5,
                distance=5,
                before_measure_flip_probability=0.333,
            ),
            (
                sp.before_measure_flip_probability(0.333),
                sp.Circuit.LayerAdjacency.BEFORE,
            ),
        ),
        (
            stim.Circuit.generated(
                "surface_code:rotated_memory_x", rounds=3, distance=3
            ),
            stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                rounds=3,
                distance=3,
                after_reset_flip_probability=0.333,
            ),
            (sp.after_reset_flip_probability(0.333), sp.Circuit.LayerAdjacency.AFTER),
        ),
        (
            stim.Circuit.generated("color_code:memory_xyz", rounds=5, distance=5),
            stim.Circuit.generated(
                "color_code:memory_xyz",
                rounds=5,
                distance=5,
                after_reset_flip_probability=0.333,
            ),
            (sp.after_reset_flip_probability(0.333), sp.Circuit.LayerAdjacency.AFTER),
        ),
        (
            stim.Circuit.generated("repetition_code:memory", rounds=13, distance=13),
            stim.Circuit.generated(
                "repetition_code:memory",
                rounds=13,
                distance=13,
                after_reset_flip_probability=0.01,
            ),
            (sp.after_reset_flip_probability(0.01), sp.Circuit.LayerAdjacency.AFTER),
        ),
    ],
)
def test_stim_circuits_can_be_manipulated_with_same_noise_as_exposed_in_stim(
    clean_stim_circuit, noisy_stim_circuit, deltakit_circuit_noise_profile
):
    """Stim exposes a number of types of noise in stim.Circuit.generated
    (e.g. after_clifford_depolarisation, before_measure_flip_probability).
    These tests check that deltakit_circuit can apply the noise transformations"""
    expected_deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(noisy_stim_circuit)
    deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(clean_stim_circuit)
    deltakit_circuit_circuit.apply_gate_noise(*deltakit_circuit_noise_profile)
    assert expected_deltakit_circuit_circuit == deltakit_circuit_circuit


@pytest.mark.parametrize(
    ("clean_stim_circuit", "noisy_stim_circuit", "deltakit_circuit_noise_profile"),
    [
        (
            stim.Circuit.generated(
                "surface_code:rotated_memory_x", rounds=3, distance=3
            ),
            stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                rounds=3,
                distance=3,
                after_clifford_depolarization=0.333,
                after_reset_flip_probability=0.2,
            ),
            (
                sp.after_clifford_depolarisation(0.333)
                + sp.after_reset_flip_probability(0.2),
                sp.Circuit.LayerAdjacency.AFTER,
            ),
        ),
        (
            stim.Circuit.generated("repetition_code:memory", rounds=5, distance=5),
            stim.Circuit.generated(
                "repetition_code:memory",
                rounds=5,
                distance=5,
                after_clifford_depolarization=0.333,
                after_reset_flip_probability=0.2,
            ),
            (
                sp.after_clifford_depolarisation(0.333)
                + sp.after_reset_flip_probability(0.2),
                sp.Circuit.LayerAdjacency.AFTER,
            ),
        ),
    ],
)
def test_stim_circuits_can_be_manipulated_with_multiple_types_of_noise_exposed_in_stim_simultaneously(
    clean_stim_circuit, noisy_stim_circuit, deltakit_circuit_noise_profile
):
    """Stim exposes a number of types of noise in stim.Circuit.generated
    (e.g. after_clifford_depolarisation, before_measure_flip_probability).
    These tests check that deltakit_circuit can apply multiple of these noise
    transformations simultaneously"""
    expected_deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(noisy_stim_circuit)
    deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(clean_stim_circuit)
    deltakit_circuit_circuit.apply_gate_noise(*deltakit_circuit_noise_profile)
    assert expected_deltakit_circuit_circuit == deltakit_circuit_circuit


def test_stim_circuits_can_be_manipulated_with_all_types_of_noise_exposed_in_stim_simultaneously():
    """Stim exposes a number of types of noise in stim.Circuit.generated
    (e.g. after_clifford_depolarisation, before_measure_flip_probability).
    These tests check that deltakit_circuit can apply all of these noise
    transformations simultaneously"""
    clean_stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x", rounds=3, distance=3
    )
    noisy_stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.333,
        before_measure_flip_probability=0.04,
        after_reset_flip_probability=0.2,
    )

    expected_deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(noisy_stim_circuit)
    deltakit_circuit_circuit = sp.Circuit.from_stim_circuit(clean_stim_circuit)
    deltakit_circuit_circuit.apply_gate_noise(
        sp.after_clifford_depolarisation(0.333) + sp.after_reset_flip_probability(0.2),
        sp.Circuit.LayerAdjacency.AFTER,
    )
    deltakit_circuit_circuit.apply_gate_noise(
        sp.before_measure_flip_probability(0.04), sp.Circuit.LayerAdjacency.BEFORE
    )
    assert expected_deltakit_circuit_circuit == deltakit_circuit_circuit
