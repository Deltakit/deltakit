# (c) Copyright Riverlane 2020-2025.
import deltakit_circuit as sp
import pytest
import stim
from deltakit_circuit.gates._abstract_gates import (OneQubitMeasurementGate,
                                                    TwoOperandGate)
from deltakit_decode.noise_sources import OptionedStim, StimNoise, ToyNoise


class TestStimNoise:

    @pytest.mark.parametrize("stim_circuit, target_gate, noise_channel, expected_stim_circuit", [
        (
            stim.Circuit("CX 0 1 2 8 9 4"),
            sp.gates.CX,
            sp.noise_channels.Depolarise2,
            stim.Circuit("CX 0 1 2 8 9 4\n DEPOLARIZE2(0.1) 0 1 2 8 9 4")
        ),
        (
            stim.Circuit("H 0 1 2 8 9 4"),
            sp.gates.H,
            sp.noise_channels.Depolarise1,
            stim.Circuit("H 0 1 2 4 8 9\n DEPOLARIZE1(0.1) 0 1 2 4 8 9")
        )
    ])
    def test_noise_can_be_added_after_gate_layer(
            self, stim_circuit, target_gate,
            noise_channel, expected_stim_circuit):
        noise_channel_generator = noise_channel.generator_from_prob(0.1)
        stim_noise_model = StimNoise(
            after_gate_noise_profile=lambda noise_context:
                noise_channel_generator(
                    list(noise_context.gate_layer_qubits(target_gate))))
        assert stim_noise_model.permute_stim_circuit(
            stim_circuit) == expected_stim_circuit

    @pytest.mark.parametrize("stim_circuit, target_gate, noise_channel, expected_stim_circuit", [
        (
            stim.Circuit("CX 0 1 2 8 9 4"),
            sp.gates.CX,
            sp.noise_channels.Depolarise2,
            stim.Circuit("DEPOLARIZE2(0.1) 0 1 2 8 9 4\n CX 0 1 2 8 9 4")
        ),
        (
            stim.Circuit("H 0 1 2 8 9 4"),
            sp.gates.H,
            sp.noise_channels.Depolarise1,
            stim.Circuit("DEPOLARIZE1(0.1) 0 1 2 4 8 9\n H 0 1 2 4 8 9")
        )
    ])
    def test_noise_can_be_added_before_gate_layer(
            self, stim_circuit, target_gate,
            noise_channel, expected_stim_circuit):
        noise_channel_generator = noise_channel.generator_from_prob(0.1)
        stim_noise_model = StimNoise(
            before_gate_noise_profile=lambda noise_context:
                noise_channel_generator(
                    list(noise_context.gate_layer_qubits(target_gate))))
        assert stim_noise_model.permute_stim_circuit(
            stim_circuit) == expected_stim_circuit

    @pytest.mark.parametrize("stim_circuit, replacement_policy, expected_stim_circuit", [
        (
            stim.Circuit("MX 0 1 2 8 9 4"),
            {},
            stim.Circuit("MX 0 1 2 8 9 4")
        ),
        (
            stim.Circuit("MX 0 1 2 8 9 4"),
            {sp.gates.MX: lambda gate: sp.gates.MX(gate.qubit, 0.001)},
            stim.Circuit("MX(0.001) 0 1 2 8 9 4")
        ),
        (
            stim.Circuit("MR 0 1 2 8 9 4"),
            {sp.gates.MRZ: lambda gate: sp.gates.MRZ(gate.qubit, 0.1)},
            stim.Circuit("MR(0.1) 0 1 2 8 9 4")
        )
    ])
    def test_noise_can_be_added_to_measurement_gates_of_a_given_type(
            self, stim_circuit, replacement_policy, expected_stim_circuit):
        stim_noise_model = StimNoise(gate_replacement_policy=replacement_policy)
        assert stim_noise_model.permute_stim_circuit(
            stim_circuit) == expected_stim_circuit


class TestToyNoise:

    def test_toy_noise_can_output_field_values(self):
        optioned_stim = ToyNoise(p_physical=0.02)
        assert optioned_stim.field_values() == {'noise_name': 'ToyNoise',
                                                'p_physical': 0.02}

    @pytest.mark.parametrize("stim_circuit, expected_stim_circuit", [
        (
            stim.Circuit("CX 0 1 2 8 9 4"),
            stim.Circuit("CX 0 1 2 8 9 4\n DEPOLARIZE2(0.1) 0 1 2 8 9 4")
        ),
        (
            stim.Circuit("H 0 1 2 8 9 4"),
            stim.Circuit("H 0 1 2 4 8 9\n DEPOLARIZE1(0.01) 0 1 2 4 8 9")
        ),
        (
            stim.Circuit("M 0 1 2 3"),
            stim.Circuit("M(0.1) 0 1 2 3\n "
                         "DEPOLARIZE1(0.01) 0 1 2 3")
        ),
        (
            stim.Circuit("MR 0 1 2 3"),
            stim.Circuit("MR(0.1) 0 1 2 3\n "
                         "DEPOLARIZE1(0.01) 0 1 2 3")
        ),
        (
            stim.Circuit("R 0 1 2 3"),
            stim.Circuit("R 0 1 2 3\n"
                         "DEPOLARIZE1(0.01) 0 1 2 3")
        ),
        (
            stim.Circuit("X 0 1 2 3\n H 8 9"),
            stim.Circuit("X 0 1 2 3\n TICK\n DEPOLARIZE1(0.01) 0 1 2 3 8 9 \n"
                         "H 8 9 \nDEPOLARIZE1(0.01) 8 9 0 1 2 3")
        )
    ])
    def test_noise_can_be_added_after_gate_layer(
            self, stim_circuit, expected_stim_circuit):
        stim_noise_model = ToyNoise(0.1)
        assert stim_noise_model.permute_stim_circuit(
            stim_circuit) == expected_stim_circuit

    def test_toy_noise_adds_a_noise_channel_on_every_gate_layer(self):
        stim_circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                              rounds=3,
                                              distance=3)
        stim_noise_model = ToyNoise(0.1)
        stim_circuit = stim_noise_model.permute_stim_circuit(stim_circuit)
        circuit = sp.Circuit.from_stim_circuit(stim_circuit)
        # Should be different if noise layers was inspecting inner circuits
        assert 24 == len(circuit.noise_layers())

    def test_noise_channels_target_the_correct_qubits(self):
        stim_noise_model = ToyNoise(0.1)
        stim_circuit = stim_noise_model.permute_stim_circuit(
            stim.Circuit.generated("surface_code:rotated_memory_z",
                                   rounds=3, distance=3))
        circuit = sp.Circuit.from_stim_circuit(stim_circuit)
        noise_layer_index = 0
        for gate_layer in circuit.gate_layers():
            if issubclass(type(gate_layer.gates[0]), TwoOperandGate):
                assert (circuit.qubits ==
                        (circuit.noise_layers()[noise_layer_index].qubits) |
                        (circuit.noise_layers()[noise_layer_index+1].qubits))
                noise_layer_index += 1
            elif issubclass(type(gate_layer.gates[0]), OneQubitMeasurementGate):
                assert circuit.qubits == circuit.noise_layers()[noise_layer_index].qubits
            noise_layer_index += 1


class TestOptionedStim:
    def test_optioned_stim_can_output_field_values(self):
        optioned_stim = OptionedStim(after_clifford_depolarisation=0.03)
        assert optioned_stim.field_values() == {'noise_name': 'OptionedStim',
                                                'after_clifford_depolarisation': 0.03,
                                                'before_measure_flip_probability': 0.0,
                                                'after_reset_flip_probability': 0.0}

    @pytest.fixture(params=[
        ("repetition_code:memory", 3, 3),
        ("surface_code:rotated_memory_z", 3, 3),
        ("surface_code:rotated_memory_x", 5, 4),
        ("surface_code:unrotated_memory_x", 9, 5),
        ("surface_code:unrotated_memory_z", 11, 2),
        ("color_code:memory_xyz", 9, 5),
    ])
    def code_rounds_distance(self, request):
        return request.param

    @pytest.fixture()
    def clean_stim_circuit(self, code_rounds_distance):
        code, rounds, distance = code_rounds_distance
        return stim.Circuit.generated(code,
                                      rounds=rounds, distance=distance)

    def test_stim_circuits_can_be_manipulated_with_after_clifford_depolarisation(
            self, clean_stim_circuit, code_rounds_distance):
        code, rounds, distance = code_rounds_distance
        stim_circuit = stim.Circuit.generated(code,
                                              rounds=rounds, distance=distance,
                                              after_clifford_depolarization=0.333)
        expected_circuit = sp.Circuit.from_stim_circuit(stim_circuit)
        optioned_stim = OptionedStim(after_clifford_depolarisation=0.333)
        circuit = sp.Circuit.from_stim_circuit(
            optioned_stim.permute_stim_circuit(clean_stim_circuit))
        assert expected_circuit == circuit

    def test_applying_before_round_data_depolarisation_raises_not_implemented_exception(self):
        with pytest.raises(NotImplementedError):
            OptionedStim(before_round_data_depolarisation=0.333)

    def test_stim_circuits_can_be_manipulated_with_before_measure_flip_probability(
            self, clean_stim_circuit, code_rounds_distance):
        code, rounds, distance = code_rounds_distance
        stim_circuit = stim.Circuit.generated(code,
                                              rounds=rounds, distance=distance,
                                              before_measure_flip_probability=0.03)
        expected_circuit = sp.Circuit.from_stim_circuit(stim_circuit)
        optioned_stim = OptionedStim(before_measure_flip_probability=0.03)
        circuit = sp.Circuit.from_stim_circuit(
            optioned_stim.permute_stim_circuit(clean_stim_circuit))
        assert expected_circuit == circuit

    def test_stim_circuits_can_be_manipulated_with_after_reset_flip_probability(
            self, clean_stim_circuit, code_rounds_distance):
        if clean_stim_circuit.num_qubits in {9, 26}:
            pytest.skip(
                "These circuits are logically equivalent, but equality does not (and cannot) hold")
        code, rounds, distance = code_rounds_distance
        stim_circuit = stim.Circuit.generated(code,
                                              rounds=rounds, distance=distance,
                                              after_reset_flip_probability=0.03)
        expected_circuit = sp.Circuit.from_stim_circuit(stim_circuit)
        optioned_stim = OptionedStim(after_reset_flip_probability=0.03)
        circuit = sp.Circuit.from_stim_circuit(
            optioned_stim.permute_stim_circuit(clean_stim_circuit))
        assert expected_circuit == circuit
