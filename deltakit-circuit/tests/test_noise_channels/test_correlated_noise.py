# (c) Copyright Riverlane 2020-2025.
from copy import copy, deepcopy
from itertools import permutations

import pytest
from deltakit_circuit import PauliProduct, PauliX, PauliY, PauliZ, Qubit
from deltakit_circuit._stim_identifiers import NoiseStimIdentifier
from deltakit_circuit.noise_channels import CorrelatedError, ElseCorrelatedError

EQUAL_ONE_QUBIT_PAULI_PRODUCTS = (
    PauliProduct(PauliX(Qubit(0))),
    PauliProduct([PauliX(Qubit(0))]),
    PauliX(Qubit(0)),
    [PauliX(Qubit(0))],
)


@pytest.fixture(scope="module")
def pauli_product():
    return PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1)), PauliZ(Qubit(2))])


@pytest.mark.parametrize(
    "correlated_error, expected_string",
    [
        (CorrelatedError, "CORRELATED_ERROR"),
        (ElseCorrelatedError, "ELSE_CORRELATED_ERROR"),
    ],
)
def test_correlated_error_stim_string_matches_expected_string(
    correlated_error, expected_string
):
    assert correlated_error.stim_string == expected_string


@pytest.mark.parametrize(
    "correlated_error, expected_repr",
    [
        (
            CorrelatedError(PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]), 0.001),
            "CORRELATED_ERROR([PauliX(Qubit(0)), PauliY(Qubit(1))], probability=0.001)",
        ),
        (
            CorrelatedError(PauliX(Qubit(0)), probability=0.002),
            "CORRELATED_ERROR([PauliX(Qubit(0))], probability=0.002)",
        ),
        (
            CorrelatedError(PauliX(Qubit(0)), probability=0.002, tag="λ<0>"),
            "CORRELATED_ERROR[λ<0>]([PauliX(Qubit(0))], probability=0.002)",
        ),
        (
            ElseCorrelatedError(
                PauliProduct([PauliZ(Qubit(0)), PauliY(Qubit(1))]), 0.02
            ),
            "ELSE_CORRELATED_ERROR([PauliZ(Qubit(0)), PauliY(Qubit(1))], probability=0.02)",
        ),
        (
            ElseCorrelatedError(PauliX(Qubit(0)), probability=0.002),
            "ELSE_CORRELATED_ERROR([PauliX(Qubit(0))], probability=0.002)",
        ),
        (
            ElseCorrelatedError(PauliX(Qubit(0)), probability=0.002, tag="λ<0>"),
            "ELSE_CORRELATED_ERROR[λ<0>]([PauliX(Qubit(0))], probability=0.002)",
        ),
    ],
)
def test_repr_of_correlated_noise_matches_expected_representation(
    correlated_error, expected_repr
):
    assert repr(correlated_error) == expected_repr


@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_identical_correlated_errors_are_equal(error_class, pauli_product):
    probability = 0.01
    assert error_class(pauli_product, probability) == error_class(
        pauli_product, probability
    )


def test_correlated_errors_hash(pauli_product):
    correlated = CorrelatedError(pauli_product, 0.01)
    else_correlated = CorrelatedError(pauli_product, 0.02)
    assert hash(correlated) == hash(correlated)
    assert hash(correlated) == hash(copy(correlated))
    assert hash(correlated) == hash(deepcopy(correlated))
    assert hash(correlated) != hash(else_correlated)


@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_correlated_errors_with_probabilities_within_tolerance_are_approx_equal(
    error_class, pauli_product
):
    assert error_class(pauli_product, 0.01).approx_equals(
        error_class(pauli_product, 0.01000001), rel_tol=0.001
    )


@pytest.mark.parametrize(
    "pauli_prod_1, pauli_prod_2", permutations(EQUAL_ONE_QUBIT_PAULI_PRODUCTS, 2)
)
@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_correlated_errors_with_equivalent_one_qubit_input_pauli_products_are_equal(
    pauli_prod_1, pauli_prod_2, error_class
):
    assert error_class(pauli_prod_1, 0.1) == error_class(pauli_prod_2, 0.1)


@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_correlated_errors_with_equivalent_two_qubit_input_pauli_products_are_equal(
    error_class,
):
    pauli_prod_1 = PauliProduct([PauliX(0), PauliY(1)])
    pauli_prod_2 = [PauliX(0), PauliY(1)]
    assert error_class(pauli_prod_1, 0.1) == error_class(pauli_prod_2, 0.1)


@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_correlated_errors_that_do_not_have_the_same_probability_are_not_equal(
    error_class, pauli_product
):
    assert error_class(pauli_product, 0.01) != error_class(pauli_product, 0.02)


@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_correlated_errors_that_have_probability_outside_tolerance_are_not_approx_equal(
    error_class, pauli_product
):
    assert not error_class(pauli_product, 0.01).approx_equals(
        error_class(pauli_product, 0.02)
    )


@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_correlated_errors_that_have_different_pauli_products_are_not_equal(
    error_class,
):
    probability = 0.01
    pauli_product1 = PauliProduct([PauliX(Qubit(0)), PauliX(Qubit(1))])
    pauli_product2 = PauliProduct([PauliX(Qubit(0)), PauliX(Qubit(2))])
    assert error_class(pauli_product1, probability) != error_class(
        pauli_product2, probability
    )


@pytest.mark.parametrize("error_class", [CorrelatedError, ElseCorrelatedError])
def test_correlated_errors_that_have_different_pauli_products_are_not_approx_equal(
    error_class,
):
    probability = 0.01
    pauli_product1 = PauliProduct([PauliX(Qubit(0)), PauliX(Qubit(1))])
    pauli_product2 = PauliProduct([PauliX(Qubit(0)), PauliX(Qubit(2))])
    assert not error_class(pauli_product1, probability).approx_equals(
        error_class(pauli_product2, probability)
    )


@pytest.mark.parametrize(
    "noise_channel",
    [
        CorrelatedError(PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]), 0.1),
        ElseCorrelatedError(PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]), 0.1),
        CorrelatedError([PauliX(Qubit(0)), PauliY(Qubit(1))], 0.1),
        ElseCorrelatedError([PauliX(Qubit(0)), PauliY(Qubit(1))], 0.1),
        CorrelatedError(PauliProduct(PauliY(Qubit(1))), 0.1),
        ElseCorrelatedError(PauliProduct(PauliX(Qubit(0))), 0.1),
        CorrelatedError(PauliY(Qubit(1)), 0.1),
        ElseCorrelatedError(PauliX(Qubit(0)), 0.1),
    ],
)
def test_stim_identifier_matches_expected_identifier(noise_channel):
    assert noise_channel.stim_identifier == NoiseStimIdentifier(
        noise_channel.stim_string, (noise_channel.probability,)
    )


@pytest.mark.parametrize(
    "correlated_noise",
    [CorrelatedError(PauliX(0), 0.01), ElseCorrelatedError(PauliZ(0), 0.03)],
)
def test_correlated_noise_probabilities_is_tuple_of_probability(correlated_noise):
    assert correlated_noise.probabilities == (correlated_noise.probability,)


@pytest.mark.parametrize("bad_probability", [2, -1, -0.5])
@pytest.mark.parametrize("noise_model_class", [CorrelatedError, ElseCorrelatedError])
def test_error_is_raised_if_probability_is_out_of_bounds(
    bad_probability, noise_model_class
):
    with pytest.raises(ValueError, match="Probability must be between zero and one."):
        noise_model_class(
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]), bad_probability
        )


@pytest.mark.parametrize("noise_model_class", [CorrelatedError, ElseCorrelatedError])
def test_error_is_raised_if_constructing_noise_acting_on_same_qubit(noise_model_class):
    with pytest.raises(
        ValueError, match="Pauli product cannot contain duplicate qubits."
    ):
        noise_model_class((PauliX(0), PauliZ(0)), 0.01)


@pytest.mark.parametrize("noise_channel_class", [CorrelatedError, ElseCorrelatedError])
class TestQubitTransforms:
    def test_correlated_noise_pauli_product_does_not_change_if_qubit_id_not_in_mapping(
        self, noise_channel_class
    ):
        pauli_product = PauliProduct([PauliX(0), PauliY(1)])
        noise_channel = noise_channel_class(pauli_product, 0.01)
        noise_channel.transform_qubits({})
        assert noise_channel.pauli_product == pauli_product

    def test_correlated_noise_pauli_product_changes_if_qubit_ids_in_mapping(
        self, noise_channel_class
    ):
        noise_channel = noise_channel_class(PauliProduct([PauliX(0), PauliY(1)]), 0.01)
        noise_channel.transform_qubits({0: 2, 1: 3})
        assert noise_channel.pauli_product == PauliProduct([PauliX(2), PauliY(3)])

    def test_correlated_noise_qubits_change_if_qubit_ids_in_mapping(
        self, noise_channel_class
    ):
        noise_channel = noise_channel_class(PauliProduct([PauliX(0), PauliY(1)]), 0.01)
        noise_channel.transform_qubits({0: 2, 1: 3})
        assert noise_channel.qubits == (Qubit(2), Qubit(3))
