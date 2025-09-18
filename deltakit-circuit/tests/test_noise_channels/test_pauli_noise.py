# (c) Copyright Riverlane 2020-2025.
from itertools import combinations

import pytest
from deltakit_circuit import Qubit
from deltakit_circuit._stim_identifiers import NoiseStimIdentifier
from deltakit_circuit.noise_channels import (
    PauliChannel1,
    PauliChannel2,
    PauliXError,
    PauliYError,
    PauliZError,
)


@pytest.mark.parametrize(
    "noise_channel, expected_string",
    [
        (PauliXError, "X_ERROR"),
        (PauliYError, "Y_ERROR"),
        (PauliZError, "Z_ERROR"),
        (PauliChannel1, "PAULI_CHANNEL_1"),
        (PauliChannel2, "PAULI_CHANNEL_2"),
    ],
)
def test_pauli_noise_channels_string_matches_expected_string(
    noise_channel, expected_string
):
    assert noise_channel.stim_string == expected_string


@pytest.mark.parametrize(
    "noise_channel, bad_arguments",
    [
        (PauliXError, (4, 5)),
        (PauliYError, [1, 2, 3, 4]),
        (PauliZError, ()),
        (PauliChannel1, ("A", "B", "C")),
        (PauliChannel2, (4, 5, 6)),
    ],
)
def test_channel_generator_from_prob_raises_error_with_wrong_args(
    noise_channel, bad_arguments
):
    # The ignore on the following line is needed because the exception raised in each
    # test cases may be different, and Exception is the only common parent class of all
    # of them.
    with pytest.raises(Exception):  # noqa: B017
        noise_channel.generator_from_prob(*bad_arguments)([0, 1, 2, 4])


@pytest.mark.parametrize(
    "noise_channel",
    [
        PauliXError(Qubit(0), 0.1),
        PauliYError(Qubit(0), 0.2),
        PauliZError(Qubit(0), 0.3),
    ],
)
def test_stim_identifier_matches_expected_identifier_for_single_probability_noise(
    noise_channel,
):
    assert noise_channel.stim_identifier == NoiseStimIdentifier(
        noise_channel.stim_string, (noise_channel.probability,)
    )


@pytest.mark.parametrize(
    "noise_channel, expected_repr",
    [
        (PauliXError(Qubit(0), 0.02), "X_ERROR(Qubit(0), probability=0.02)"),
        (PauliYError(Qubit(1), 0.01), "Y_ERROR(Qubit(1), probability=0.01)"),
        (PauliZError(Qubit(2), 0.03), "Z_ERROR(Qubit(2), probability=0.03)"),
        (
            PauliChannel1(Qubit(0), 0.01, 0.02, 0.03),
            "PAULI_CHANNEL_1(Qubit(0), p_x=0.01, p_y=0.02, p_z=0.03)",
        ),
        (
            PauliChannel2(
                Qubit(0),
                Qubit(1),
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.011,
                0.012,
                0.013,
                0.014,
                0.015,
            ),
            "PAULI_CHANNEL_2(qubit1=Qubit(0), qubit2=Qubit(1), "
            "p_ix=0.001, p_iy=0.002, p_iz=0.003, "
            "p_xi=0.004, p_xx=0.005, p_xy=0.006, p_xz=0.007, "
            "p_yi=0.008, p_yx=0.009, p_yy=0.01, p_yz=0.011, "
            "p_zi=0.012, p_zx=0.013, p_zy=0.014, p_zz=0.015)",
        ),
    ],
)
def test_pauli_noise_channel_repr_matches_expected_representation(
    noise_channel, expected_repr
):
    assert repr(noise_channel) == expected_repr


class TestPauliNoiseEquality:
    @pytest.mark.parametrize("error_class", [PauliXError, PauliYError, PauliZError])
    def test_two_identical_pauli_errors_are_equal(self, error_class):
        error1 = error_class(Qubit(0), 0.01)
        error2 = error_class(Qubit(0), 0.01)
        assert error1 == error2
        assert hash(error1) == hash(error2)

    @pytest.mark.parametrize("error_class", [PauliXError, PauliYError, PauliZError])
    def test_two_pauli_errors_are_approx_equal_if_probabilities_are_within_tolerance(
        self, error_class
    ):
        assert error_class(0, 0.01).approx_equals(
            error_class(0, 0.0100001), rel_tol=0.01
        )

    @pytest.mark.parametrize("error_class", [PauliXError, PauliYError, PauliZError])
    def test_two_pauli_errors_with_different_probabilities_are_not_equal(
        self, error_class
    ):
        error1 = error_class(Qubit(0), 0.01)
        error2 = error_class(Qubit(0), 0.02)
        assert error1 != error2
        assert hash(error1) != hash(error2)

    @pytest.mark.parametrize("error_class", [PauliXError, PauliYError, PauliZError])
    def test_two_pauli_errors_with_probabilities_outside_tolerance_are_not_approx_equal(
        self, error_class
    ):
        assert not error_class(0, 0.01).approx_equals(error_class(0, 0.4))

    @pytest.mark.parametrize("error_class", [PauliXError, PauliYError, PauliZError])
    def test_two_pauli_errors_on_different_qubits_are_not_equal(self, error_class):
        error1 = error_class(Qubit(0), 0.01)
        error2 = error_class(Qubit(1), 0.01)
        assert error1 != error2
        assert hash(error1) != hash(error2)

    @pytest.mark.parametrize("error_class", [PauliXError, PauliYError, PauliZError])
    def test_two_pauli_errors_on_different_qubits_are_not_approx_equal(
        self, error_class
    ):
        assert not error_class(0, 0.01).approx_equals(error_class(1, 0.01))

    @pytest.mark.parametrize(
        "error_class1, error_class2",
        combinations([PauliXError, PauliYError, PauliZError], 2),
    )
    def test_two_different_pauli_errors_on_the_same_qubits_are_not_equal(
        self, error_class1, error_class2
    ):
        error1 = error_class1(Qubit(0), 0.01)
        error2 = error_class2(Qubit(0), 0.01)
        assert error1 != error2
        assert hash(error1) != hash(error2)

    @pytest.mark.parametrize(
        "error_class1, error_class2",
        combinations([PauliXError, PauliYError, PauliZError], 2),
    )
    def test_two_different_pauli_errors_on_the_same_qubits_are_not_approx_equal(
        self, error_class1, error_class2
    ):
        assert not error_class1(0, 0.01).approx_equals(error_class2(0, 0.01))

    def test_identical_pauli_channel_1_noises_are_equal(self):
        error1 = PauliChannel1(Qubit(0), 0.01, 0.02, 0.03)
        error2 = PauliChannel1(Qubit(0), 0.01, 0.02, 0.03)
        assert error1 == error2
        assert hash(error1) == hash(error2)

    def test_pauli_channel_1_with_probabilities_within_tolerance_are_approx_equal(self):
        error1 = PauliChannel1(Qubit(0), 0.01, 0.02, 0.03)
        error2 = PauliChannel1(Qubit(0), 0.010001, 0.020001, 0.0300001)
        assert error1.approx_equals(error2, rel_tol=0.001)

    def test_pauli_channel_1_noises_with_different_probabilities_are_not_equal(self):
        error1 = PauliChannel1(Qubit(0), 0.01, 0.02, 0.03)
        error2 = PauliChannel1(Qubit(0), 0.01, 0.02, 0.04)
        assert error1 != error2
        assert hash(error1) != hash(error2)

    def test_two_pauli_channel_1_with_probability_outside_of_tolerance_are_not_approx_equal(
        self,
    ):
        error1 = PauliChannel1(0, 0.01, 0.02, 0.03)
        error2 = PauliChannel1(0, 0.02, 0.04, 0.02)
        assert not error1.approx_equals(error2)

    def test_pauli_channel_1_noises_on_different_qubits_are_not_equal(self):
        error1 = PauliChannel1(Qubit(0), 0.01, 0.02, 0.03)
        error2 = PauliChannel1(Qubit(1), 0.01, 0.02, 0.03)
        assert error1 != error2
        assert hash(error1) != hash(error2)

    def test_pauli_channel_1_noises_on_different_qubits_are_not_approx_equal(self):
        error1 = PauliChannel1(Qubit(0), 0.01, 0.02, 0.03)
        error2 = PauliChannel1(Qubit(1), 0.01, 0.02, 0.03)
        assert not error1.approx_equals(error2)

    def test_two_identical_pauli_channel_2_noises_are_equal(self):
        error1 = PauliChannel2(Qubit(0), Qubit(1), 0.1)
        error2 = PauliChannel2(Qubit(0), Qubit(1), 0.1)
        assert error1 == error2
        assert hash(error1) == hash(error2)

    def test_two_pauli_channel_2_with_probability_within_tolerance_are_approx_equal(
        self,
    ):
        error1 = PauliChannel2(Qubit(0), Qubit(1), 0.1)
        error2 = PauliChannel2(Qubit(0), Qubit(1), 0.100001)
        assert error1.approx_equals(error2, rel_tol=0.001)

    def test_pauli_channel_2_noises_with_different_probabilities_are_not_equal(self):
        error1 = PauliChannel2(Qubit(0), Qubit(1), 0.1)
        error2 = PauliChannel2(Qubit(0), Qubit(1), 0.2)
        assert error1 != error2
        assert hash(error1) != hash(error2)

    def test_two_pauli_channel_2_with_probability_outside_of_tolerance_are_not_approx_equal(
        self,
    ):
        error1 = PauliChannel2(0, 1, 0.01)
        error2 = PauliChannel2(0, 1, 0.1)
        assert not error1.approx_equals(error2)

    def test_pauli_channel_2_noises_on_different_qubits_are_not_equal(self):
        error1 = PauliChannel2(Qubit(0), Qubit(1), 0.1)
        error2 = PauliChannel2(Qubit(0), Qubit(2), 0.1)
        assert error1 != error2
        assert hash(error1) != hash(error2)

    def test_pauli_channel_2_noises_on_different_qubits_are_not_approx_equal(self):
        error1 = PauliChannel2(0, 1, 0.01, 0.02, 0.03)
        error2 = PauliChannel2(0, 2, 0.01, 0.02, 0.03)
        assert not error1.approx_equals(error2)


@pytest.mark.parametrize(
    "noise_channel",
    [PauliChannel1(Qubit(0), 0.1, 0.2, 0.3), PauliChannel2(Qubit(0), Qubit(1), 0.2)],
)
def test_stim_identifier_matches_expected_identifier_for_multiple_probability_noise(
    noise_channel,
):
    assert noise_channel.stim_identifier == NoiseStimIdentifier(
        noise_channel.stim_string, noise_channel.probabilities
    )


@pytest.mark.parametrize(
    "noise_channel",
    [
        PauliXError(Qubit(0), 0.02),
        PauliYError(Qubit(1), 0.01),
        PauliZError(Qubit(2), 0.002),
    ],
)
def test_pauli_noise_probabilities_is_just_probability_in_tuple(noise_channel):
    assert noise_channel.probabilities == (noise_channel.probability,)


@pytest.mark.parametrize("bad_probability", [-1.0, 1.2, -0.5])
@pytest.mark.parametrize("noise_model_class", [PauliXError, PauliYError, PauliZError])
def test_error_is_raised_if_independent_pauli_probability_is_outside_natural_range(
    noise_model_class, bad_probability
):
    with pytest.raises(ValueError, match="Probability must be between zero and one."):
        noise_model_class(Qubit(0), bad_probability)


@pytest.mark.parametrize(
    "bad_probabilities",
    [
        (0.0, 0.1, 1.1),
        (-0.1, 0.2, 0.3),
        (0.1, -0.2, 0.3),
    ],
)
def test_error_is_raised_if_any_pauli_channel_1_probabilities_are_outside_natural_range(
    bad_probabilities,
):
    with pytest.raises(ValueError, match="Probability must be between zero and one."):
        PauliChannel1(Qubit(0), *bad_probabilities)


def test_error_is_raised_if_sum_of_pauli_channel_1_probabilities_is_greater_than_one():
    with pytest.raises(
        ValueError, match="The sum of probabilities cannot be greater than one."
    ):
        PauliChannel1(Qubit(0), 0.5, 0.5, 0.1)


@pytest.mark.parametrize(
    "bad_probabilities", [(-0.1,), (0.0, -0.1), (1.2,), (0.0, 1.2)]
)
def test_error_is_raised_if_any_pauli_channel_2_probabilities_are_outside_natural_range(
    bad_probabilities,
):
    with pytest.raises(ValueError, match="Probability must be between zero and one."):
        PauliChannel2(Qubit(0), Qubit(1), *bad_probabilities)


def test_error_is_raised_if_sum_of_pauli_channel_2_probabilities_is_greater_than_one():
    with pytest.raises(
        ValueError, match="The sum of probabilities cannot be greater than one."
    ):
        PauliChannel2(Qubit(0), Qubit(1), 0.4, 0.4, 0.2, 0.1)


def test_error_is_raised_if_constructing_pauli_channel_two_with_odd_number_of_qubits():
    with pytest.raises(
        ValueError,
        match="Two qubit noise channels can only "
        "be constructed from an even number "
        "of qubits",
    ):
        list(PauliChannel2.from_consecutive([1, 2, 3]))


@pytest.mark.parametrize(
    "qubit1, qubit2", [(0, 0), (Qubit(3), Qubit(3)), ("a", "a"), (0, Qubit(0))]
)
def test_error_is_raised_if_arguments_to_pauli_channel_2_are_the_same(qubit1, qubit2):
    with pytest.raises(
        ValueError, match="Qubits in two qubit noise channels must be different."
    ):
        PauliChannel2(qubit1, qubit2, 0.01)


@pytest.mark.parametrize(
    "pauli_error",
    [
        PauliXError(0, 0.1),
        PauliYError(0, 0.1),
        PauliZError(0, 0.1),
        PauliChannel1(0, 0.01, 0.02, 0.03),
    ],
)
def test_single_qubit_pauli_error_qubit_type_is_qubit_when_passed_generic_type(
    pauli_error,
):
    assert isinstance(pauli_error.qubit, Qubit)


def test_pauli_channel_2_qubit_types_are_qubits_when_passed_generic_type():
    pauli_channel_2 = PauliChannel2(0, 1, 0)
    assert isinstance(pauli_channel_2.qubit1, Qubit)
    assert isinstance(pauli_channel_2.qubit2, Qubit)


class TestQubitTransforms:
    @pytest.mark.parametrize(
        "pauli_error_class", [PauliXError, PauliYError, PauliZError]
    )
    def test_single_pauli_qubit_does_not_change_if_qubit_id_not_in_mapping(
        self, pauli_error_class
    ):
        original_qubit = Qubit(0)
        pauli_error = pauli_error_class(original_qubit, 0.01)
        pauli_error.transform_qubits({})
        assert pauli_error.qubit is original_qubit

    def test_pauli_channel_qubit_does_not_change_if_qubit_id_not_in_mapping(self):
        original_qubit = Qubit(0)
        pauli_error = PauliChannel1(original_qubit, 0.01, 0.01, 0.01)
        pauli_error.transform_qubits({})
        assert pauli_error.qubit is original_qubit

    def test_pauli_channel_2_qubits_do_not_change_if_qubit_id_not_in_mapping(self):
        qubit1, qubit2 = Qubit(0), Qubit(1)
        pauli_error = PauliChannel2(qubit1, qubit2, 0.01)
        pauli_error.transform_qubits({})
        assert pauli_error.qubit1 is qubit1
        assert pauli_error.qubit2 is qubit2

    @pytest.mark.parametrize(
        "pauli_error_class", [PauliXError, PauliYError, PauliZError]
    )
    def test_single_pauli_error_qubit_transforms_when_qubit_id_in_mapping(
        self, pauli_error_class
    ):
        pauli_error = pauli_error_class(Qubit(0), 0.01)
        pauli_error.transform_qubits({0: 1})
        assert pauli_error.qubit == Qubit(1)

    def test_pauli_channel_1_qubit_transforms_when_qubit_id_in_mapping(self):
        pauli_error = PauliChannel1(Qubit(0), 0.01, 0.02, 0.03)
        pauli_error.transform_qubits({0: 1})
        assert pauli_error.qubit == Qubit(1)

    def test_pauli_channel_2_qubit_transforms_when_qubit_id_in_mapping(self):
        pauli_error = PauliChannel2(Qubit(0), Qubit(1), 0.01)
        pauli_error.transform_qubits({0: 2, 1: 3})
        assert pauli_error.qubit1 == Qubit(2)
        assert pauli_error.qubit2 == Qubit(3)
