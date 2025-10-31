# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import Qubit
from deltakit_circuit._stim_identifiers import NoiseStimIdentifier
from deltakit_circuit.noise_channels import Depolarise1, Depolarise2


@pytest.mark.parametrize(
    "depolarising_noise, expected_string",
    [(Depolarise1, "DEPOLARIZE1"), (Depolarise2, "DEPOLARIZE2")],
)
def test_depolarising_noise_stim_string_matches_expected_string(
    depolarising_noise, expected_string
):
    assert depolarising_noise.stim_string == expected_string


class TestDepolarisingNoiseEquality:
    def test_two_identical_depolarise1_noises_are_equal(self):
        noise1 = Depolarise1(Qubit(0), 0.02)
        noise2 = Depolarise1(Qubit(0), 0.02)
        assert noise1 == noise2
        assert hash(noise1) == hash(noise2)

    def test_two_depolarise1_noises_with_probabilities_within_tolerance_are_approx_equal(
        self,
    ):
        assert Depolarise1(Qubit(0), 0.02).approx_equals(
            Depolarise1(Qubit(0), 0.020001), rel_tol=0.001
        )

    def test_depolarise1_noises_with_different_probabilities_are_not_equal(self):
        noise1 = Depolarise1(Qubit(0), 0.01)
        noise2 = Depolarise1(Qubit(0), 0.02)
        assert noise1 != noise2
        assert hash(noise1) != hash(noise2)

    def test_depolarise1_noises_with_probabilities_outside_tolerance_are_not_approx_equal(
        self,
    ):
        assert not Depolarise1(Qubit(0), 0.01).approx_equals(
            Depolarise1(Qubit(0), 0.02)
        )

    def test_depolarise1_noises_on_different_qubits_are_not_equal(self):
        noise1 = Depolarise1(Qubit(0), 0.01)
        noise2 = Depolarise1(Qubit(1), 0.01)
        assert noise1 != noise2
        assert hash(noise1) != hash(noise2)

    def test_depolarise1_noises_on_different_qubits_are_not_approx_equal(self):
        assert not Depolarise1(Qubit(0), 0.01).approx_equals(
            Depolarise1(Qubit(1), 0.01)
        )

    def test_two_identical_depolarise2_noises_are_equal(self):
        noise1 = Depolarise2(Qubit(0), Qubit(1), 0.02)
        noise2 = Depolarise2(Qubit(0), Qubit(1), 0.02)
        assert noise1 == noise2
        assert hash(noise1) == hash(noise2)

    def test_two_depolarise2_noises_with_probabilities_within_tolerance_are_approx_equal(
        self,
    ):
        assert Depolarise2(Qubit(0), Qubit(1), 0.02).approx_equals(
            Depolarise2(Qubit(0), Qubit(1), 0.02001), rel_tol=0.001
        )

    def test_two_depolarise2_noises_with_different_probabilities_are_not_equal(self):
        noise1 = Depolarise2(Qubit(0), Qubit(1), 0.01)
        noise2 = Depolarise2(Qubit(0), Qubit(1), 0.02)
        assert noise1 != noise2
        assert hash(noise1) != hash(noise2)

    def test_depolarise2_noises_with_probabilities_outside_tolerance_are_not_approx_equal(
        self,
    ):
        assert not Depolarise2(Qubit(0), Qubit(1), 0.01).approx_equals(
            Depolarise2(Qubit(0), Qubit(1), 0.02)
        )

    def test_two_depolarise2_noises_on_different_qubits_are_not_equal(self):
        noise1 = Depolarise2(Qubit(0), Qubit(1), 0.01)
        noise2 = Depolarise2(Qubit(0), Qubit(2), 0.01)
        assert noise1 != noise2
        assert hash(noise1) != hash(noise2)

    def test_two_depolarise2_noises_on_different_qubits_are_not_approx_equal(self):
        assert not Depolarise2(Qubit(0), Qubit(1), 0.01).approx_equals(
            Depolarise2(Qubit(0), Qubit(2), 0.01)
        )


def test_depolarise1_qubit_type_is_qubit_class_when_given_generic_type():
    assert isinstance(Depolarise1(0, 0.2).qubit, Qubit)


def test_depolarise2_qubit_types_are_qubit_classes_when_given_generic_type():
    depolarise2 = Depolarise2(0, 1, 0.1)
    assert isinstance(depolarise2.qubit1, Qubit)
    assert isinstance(depolarise2.qubit2, Qubit)


@pytest.mark.parametrize(
    "noise_channel", [Depolarise1(Qubit(0), 0.1), Depolarise2(Qubit(0), Qubit(2), 0.2)]
)
def test_stim_identifier_matches_expected_stim_identifier(noise_channel):
    assert noise_channel.stim_identifier == NoiseStimIdentifier(
        noise_channel.stim_string, (noise_channel.probability,)
    )


@pytest.mark.parametrize(
    "depolarising_noise, expected_repr",
    [
        (Depolarise1(Qubit(0), 0.001), "DEPOLARIZE1(Qubit(0), probability=0.001)"),
        (
            Depolarise1(Qubit(0), 0.001, tag="λ<0>"),
            "DEPOLARIZE1[λ<0>](Qubit(0), probability=0.001)",
        ),
        (
            Depolarise2(Qubit(0), Qubit(1), 0.01),
            "DEPOLARIZE2(qubit1=Qubit(0), qubit2=Qubit(1), probability=0.01)",
        ),
        (
            Depolarise2(Qubit(0), Qubit(1), 0.01, tag="λ<0>"),
            "DEPOLARIZE2[λ<0>](qubit1=Qubit(0), qubit2=Qubit(1), probability=0.01)",
        ),
    ],
)
def test_repr_of_depolarising_noise_matches_expected_representation(
    depolarising_noise, expected_repr
):
    assert repr(depolarising_noise) == expected_repr


@pytest.mark.parametrize(
    "depolarising_noise",
    [Depolarise1(Qubit(0), 0.02), Depolarise2(Qubit(0), Qubit(1), 0.01)],
)
def test_depolarising_noise_probabilities_is_tuple_of_probability(depolarising_noise):
    assert depolarising_noise.probabilities == (depolarising_noise.probability,)


@pytest.mark.parametrize("bad_probability", [-0.1, 1.2, -1.4])
@pytest.mark.parametrize(
    "noise_generator",
    [
        lambda probability: Depolarise1(Qubit(0), probability),
        lambda probability: Depolarise2(Qubit(0), Qubit(1), probability),
    ],
)
def test_error_is_raised_if_probability_is_outside_of_the_natural_range(
    bad_probability, noise_generator
):
    with pytest.raises(ValueError, match="Probability must be between zero and one."):
        noise_generator(bad_probability)


@pytest.mark.parametrize(
    "qubit1, qubit2", [(0, 0), (Qubit(3), Qubit(3)), ("a", "a"), (0, Qubit(0))]
)
def test_error_is_raised_if_arguments_to_depolarise2_are_the_same(qubit1, qubit2):
    with pytest.raises(
        ValueError, match="Qubits in two qubit noise channels must be different."
    ):
        Depolarise2(qubit1, qubit2, 0.01)


class TestQubitTransforms:
    def test_depolarise1_qubit_does_not_transform_if_qubit_id_not_in_mapping(self):
        qubit = Qubit(0)
        depolarising_noise = Depolarise1(qubit, 0.01)
        depolarising_noise.transform_qubits({})
        assert depolarising_noise.qubit is qubit

    def test_depolarise2_qubits_does_not_transform_if_qubit_id_not_in_mapping(self):
        qubit1, qubit2 = Qubit(0), Qubit(1)
        depolarising_noise = Depolarise2(qubit1, qubit2, 0.01)
        depolarising_noise.transform_qubits({})
        assert depolarising_noise.qubit1 is qubit1
        assert depolarising_noise.qubit2 is qubit2

    def test_depolarise1_qubit_transform_when_qubit_id_in_mapping(self):
        depolarising_noise = Depolarise1(Qubit(0), 0.01)
        depolarising_noise.transform_qubits({0: 1})
        assert depolarising_noise.qubit == Qubit(1)

    def test_depolarise2_qubits_transform_when_qubit_ids_in_mapping(self):
        depolarising_noise = Depolarise2(Qubit(0), Qubit(1), 0.01)
        depolarising_noise.transform_qubits({0: 2, 1: 3})
        assert depolarising_noise.qubit1 == Qubit(2)
        assert depolarising_noise.qubit2 == Qubit(3)
