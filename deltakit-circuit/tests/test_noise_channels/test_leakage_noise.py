# (c) Copyright Riverlane 2020-2025.
from typing import Type

import pytest

from deltakit_circuit import Qubit
from deltakit_circuit._qubit_identifiers import T
from deltakit_circuit._stim_identifiers import NoiseStimIdentifier
from deltakit_circuit.noise_channels import Leakage, Relax
from deltakit_circuit.noise_channels._abstract_noise_channels import (
    OneQubitOneProbabilityNoiseChannel,
)


@pytest.mark.parametrize(
    "depolarising_noise, expected_string", [(Leakage, "LEAKAGE"), (Relax, "RELAX")]
)
def test_leakage_noise_stim_string_matches_expected_string(
    depolarising_noise, expected_string
):
    assert depolarising_noise.stim_string == expected_string


@pytest.fixture(params=[Leakage, Relax])
def channel_type(request) -> Type[OneQubitOneProbabilityNoiseChannel[T]]:
    return request.param


class TestLeakageNoiseEquality:
    def test_two_identical_leakage_noises_are_equal(self, channel_type):
        noise1 = channel_type(Qubit(0), 0.02)
        noise2 = channel_type(Qubit(0), 0.02)
        assert noise1 == noise2
        assert hash(noise1) == hash(noise2)

    def test_two_leakage_noises_with_probabilities_within_tolerance_are_approx_equal(
        self, channel_type
    ):
        assert channel_type(Qubit(0), 0.02).approx_equals(
            channel_type(Qubit(0), 0.020001), rel_tol=0.001
        )

    def test_leakage_noises_with_different_probabilities_are_not_equal(
        self, channel_type
    ):
        noise1 = channel_type(Qubit(0), 0.01)
        noise2 = channel_type(Qubit(0), 0.02)
        assert noise1 != noise2
        assert hash(noise1) != hash(noise2)

    def test_leakage_noises_with_probabilities_outside_tolerance_are_not_approx_equal(
        self, channel_type
    ):
        assert not channel_type(Qubit(0), 0.01).approx_equals(
            channel_type(Qubit(0), 0.02)
        )

    def test_leakage_noises_on_different_qubits_are_not_equal(self, channel_type):
        noise1 = channel_type(Qubit(0), 0.01)
        noise2 = channel_type(Qubit(1), 0.01)
        assert noise1 != noise2
        assert hash(noise1) != hash(noise2)

    def test_leakage_noises_on_different_qubits_are_not_approx_equal(
        self, channel_type
    ):
        assert not channel_type(Qubit(0), 0.01).approx_equals(
            channel_type(Qubit(1), 0.01)
        )


def test_leakage_qubit_type_is_qubit_class_when_given_generic_type(channel_type):
    assert isinstance(channel_type(0, 0.2).qubit, Qubit)


@pytest.mark.parametrize(
    "noise_channel", [Leakage(Qubit(0), 0.1), Relax(Qubit(0), 0.2)]
)
def test_stim_identifier_matches_expected_stim_identifier(noise_channel):
    assert noise_channel.stim_identifier == NoiseStimIdentifier(
        noise_channel.stim_string, (noise_channel.probability,)
    )


@pytest.mark.parametrize(
    "leakage_noise, expected_repr",
    [
        (Leakage(Qubit(0), 0.001), "LEAKAGE(Qubit(0), probability=0.001)"),
        (Relax(Qubit(0), 0.01), "RELAX(Qubit(0), probability=0.01)"),
        (
            Leakage(Qubit(0), 0.001, tag="λ<0>"),
            "LEAKAGE[λ<0>](Qubit(0), probability=0.001)",
        ),
        (Relax(Qubit(0), 0.01, tag="λ<0>"), "RELAX[λ<0>](Qubit(0), probability=0.01)"),
    ],
)
def test_repr_of_leakage_noise_matches_expected_representation(
    leakage_noise, expected_repr
):
    assert repr(leakage_noise) == expected_repr


@pytest.mark.parametrize(
    "leakage_noise", [Leakage(Qubit(0), 0.02), Relax(Qubit(0), 0.01)]
)
def test_leakage_noise_probabilities_is_tuple_of_probability(leakage_noise):
    assert leakage_noise.probabilities == (leakage_noise.probability,)


@pytest.mark.parametrize("bad_probability", [-0.1, 1.2, -1.4])
@pytest.mark.parametrize(
    "noise_generator",
    [
        lambda probability: Leakage(Qubit(0), probability),
        lambda probability: Relax(Qubit(0), probability),
    ],
)
def test_error_is_raised_if_probability_is_outside_of_the_natural_range(
    bad_probability, noise_generator
):
    with pytest.raises(ValueError, match="Probability must be between zero and one."):
        noise_generator(bad_probability)


class TestQubitTransforms:
    def test_leakage_qubit_does_not_transform_if_qubit_id_not_in_mapping(
        self, channel_type
    ):
        qubit = Qubit(0)
        depolarising_noise = channel_type(qubit, 0.01)
        depolarising_noise.transform_qubits({})
        assert depolarising_noise.qubit is qubit

    def test_leakage_qubit_transform_when_qubit_id_in_mapping(self, channel_type):
        depolarising_noise = channel_type(Qubit(0), 0.01)
        depolarising_noise.transform_qubits({0: 1})
        assert depolarising_noise.qubit == Qubit(1)
