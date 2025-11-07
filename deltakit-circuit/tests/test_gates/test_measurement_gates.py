# (c) Copyright Riverlane 2020-2025.
from itertools import permutations
import itertools

import pytest
from deltakit_circuit import (
    InvertiblePauliX,
    InvertiblePauliY,
    InvertiblePauliZ,
    MeasurementPauliProduct,
    PauliX,
    PauliY,
    PauliZ,
    Qubit,
    gates,
)


@pytest.mark.parametrize(
    "measurement_gate, expected_string",
    [
        (gates.MZ, "MZ"),
        (gates.MX, "MX"),
        (gates.MY, "MY"),
        (gates.MRZ, "MRZ"),
        (gates.MRX, "MRX"),
        (gates.MRY, "MRY"),
        (gates.MPP, "MPP"),
        (gates.HERALD_LEAKAGE_EVENT, "HERALD_LEAKAGE_EVENT"),
    ],
)
def test_all_measurement_gate_stim_string_match_expected_string(
    measurement_gate, expected_string
):
    assert measurement_gate.stim_string == expected_string


@pytest.mark.parametrize(
    "measurement_gate, expected_basis",
    [
        (gates.MZ, gates.PauliBasis.Z),
        (gates.MX, gates.PauliBasis.X),
        (gates.MY, gates.PauliBasis.Y),
        (gates.MRZ, gates.PauliBasis.Z),
        (gates.MRX, gates.PauliBasis.X),
        (gates.MRY, gates.PauliBasis.Y),
        (gates.HERALD_LEAKAGE_EVENT, None),
    ],
)
def test_one_qubit_measurement_gate_bases_match_expected_basis(
    measurement_gate, expected_basis
):
    assert measurement_gate.basis == expected_basis


@pytest.mark.parametrize(
    "measurement_gate_class,tag,probability",
    itertools.product(
        gates.MEASUREMENT_GATES - {gates.MPP},
        [None, "", "sjkdhf", "λ", "leaky<0>"],
        [0.0, 0.02],
    ),
)
def test_repr_of_non_inverted_one_qubit_measurement_gate_matches_expected_representation(
    measurement_gate_class, tag: str | None, probability: float
) -> None:
    tag_repr = f"[{tag}]" if tag is not None else ""
    assert (
        repr(measurement_gate_class(Qubit(3), probability, False, tag=tag))
        == f"{measurement_gate_class.stim_string}{tag_repr}(Qubit(3), probability={probability})"
    )


@pytest.mark.parametrize(
    "measurement_gate_class,tag",
    itertools.product(
        gates.MEASUREMENT_GATES - {gates.MPP},
        [None, "", "sjkdhf", "λ", "leaky<0>"],
    ),
)
def test_repr_of_inverted_one_qubit_measurement_gate_matches_expected_representation(
    measurement_gate_class, tag: str | None
) -> None:
    tag_repr = f"[{tag}]" if tag is not None else ""
    assert (
        repr(measurement_gate_class(Qubit(3), 0.1, True, tag=tag))
        == f"!{measurement_gate_class.stim_string}{tag_repr}(Qubit(3), probability=0.1)"
    )


@pytest.mark.parametrize(
    "mpp_gate, expected_repr",
    [
        (
            gates.MPP(PauliX(Qubit(0)), 0.01),
            "MPP([PauliX(Qubit(0))], probability=0.01)",
        ),
        (
            gates.MPP(MeasurementPauliProduct([PauliY(0), PauliZ(1)]), 0.02),
            "MPP([PauliY(Qubit(0)), PauliZ(Qubit(1))], probability=0.02)",
        ),
        (
            gates.MPP(
                MeasurementPauliProduct([PauliY(0), PauliZ(1)]), 0.02, tag="λ<0>"
            ),
            "MPP[λ<0>]([PauliY(Qubit(0)), PauliZ(Qubit(1))], probability=0.02)",
        ),
    ],
)
def test_repr_of_mpp_gate_matches_expected_representation(
    mpp_gate: gates.MPP, expected_repr: str
):
    assert repr(mpp_gate) == expected_repr


class TestSingleQubitMeasurements:
    SINGLE_QUBIT_MEASUREMENTS = gates.MEASUREMENT_GATES - {gates.MPP}

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    @pytest.mark.parametrize("bad_probability", [1.1, -0.2, 5])
    def test_error_is_raised_if_probability_is_out_of_bounds_for_single_qubit_measurements(
        self, measurement_gate, bad_probability
    ):
        with pytest.raises(
            ValueError, match="Probability must be between zero and one."
        ):
            measurement_gate(Qubit(0), bad_probability)

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    @pytest.mark.parametrize("probability", [0.0, 0.2])
    def test_single_qubit_measurements_on_the_same_qubit_are_equal(
        self, measurement_gate, probability
    ):
        assert measurement_gate(Qubit(0), probability) == measurement_gate(
            Qubit(0), probability
        )

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    @pytest.mark.parametrize("probability", [0.0, 0.2])
    def test_single_qubit_measurements_on_the_same_qubit_have_the_same_hash(
        self, measurement_gate, probability
    ):
        assert hash(measurement_gate(Qubit(0), probability)) == hash(
            measurement_gate(Qubit(0), probability)
        )

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    def test_single_qubit_measurements_on_different_qubits_are_not_equal(
        self, measurement_gate
    ):
        assert measurement_gate(Qubit(0)) != measurement_gate(Qubit(1))

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    def test_single_qubit_measurements_with_different_probabilities_are_not_equal(
        self, measurement_gate
    ):
        assert measurement_gate(Qubit(0), 0.1) != measurement_gate(Qubit(0), 0.2)

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    def test_single_qubit_measurements_with_different_invert_are_not_equal(
        self, measurement_gate
    ):
        assert measurement_gate(Qubit(0), 0.1, invert=False) != measurement_gate(
            Qubit(0), 0.1, invert=True
        )

    @pytest.mark.parametrize(
        "measurement_gate1, measurement_gate2",
        permutations(SINGLE_QUBIT_MEASUREMENTS, 2),
    )
    def test_different_single_qubit_measurements_on_same_qubit_are_not_equal(
        self, measurement_gate1, measurement_gate2
    ):
        assert measurement_gate1(Qubit(0)) != measurement_gate2(Qubit(0))

    @pytest.mark.parametrize("measurement_gate_class", SINGLE_QUBIT_MEASUREMENTS)
    def test_measurement_gates_are_approximately_equal_if_probabilities_within_tolerance(
        self, measurement_gate_class
    ):
        gate1 = measurement_gate_class(0, 0.01)
        gate2 = measurement_gate_class(0, 0.01001)
        assert gate1.approx_equals(gate2, rel_tol=0.001)

    @pytest.mark.parametrize(
        "measurement_gate1, measurement_gate2",
        permutations(SINGLE_QUBIT_MEASUREMENTS, 2),
    )
    def test_equal_measurement_gates_except_gate_type_are_not_approximately_equal(
        self, measurement_gate1, measurement_gate2
    ):
        assert not measurement_gate1(Qubit(0), 0.1).approx_equals(
            measurement_gate2(Qubit(0), 0.1)
        )

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    def test_same_measurement_gate_except_qubit_are_not_approximately_equal(
        self, measurement_gate
    ):
        assert not measurement_gate(Qubit(0), 0.1).approx_equals(
            measurement_gate(Qubit(1), 0.1)
        )

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    def test_same_measurement_gate_with_probabilities_outside_tolerance_are_not_approx_equal(
        self, measurement_gate
    ):
        assert not measurement_gate(Qubit(0), 0.1).approx_equals(
            measurement_gate(Qubit(0), 0.2)
        )

    @pytest.mark.parametrize("measurement_gate", SINGLE_QUBIT_MEASUREMENTS)
    def test_same_measurement_gate_except_inverted_are_not_approximately_equal(
        self, measurement_gate
    ):
        assert not measurement_gate(Qubit(0), 0.1, invert=False).approx_equals(
            measurement_gate(Qubit(0), 0.1, invert=True)
        )

    @pytest.mark.parametrize(
        "measurement_gate", (gate(Qubit(0), 0.01) for gate in SINGLE_QUBIT_MEASUREMENTS)
    )
    def test_inverting_single_qubit_measurement_sets_inverted_flag(
        self, measurement_gate
    ):
        assert (~measurement_gate).is_inverted

    @pytest.mark.parametrize(
        "measurement_gate", (gate(Qubit(0), 0.01) for gate in SINGLE_QUBIT_MEASUREMENTS)
    )
    def test_inverse_of_an_inverted_measurement_gate_is_equal_to_non_inverted_gate(
        self, measurement_gate
    ):
        assert ~(~measurement_gate) == measurement_gate

    @pytest.mark.parametrize("measurement_gate_class", SINGLE_QUBIT_MEASUREMENTS)
    def test_qubit_property_of_gates_is_qubit_type_when_passed_generic_type(
        self, measurement_gate_class
    ):
        assert isinstance(measurement_gate_class(0).qubit, Qubit)


class TestMPPGate:
    EQUAL_MPP_QUBIT_IDENTIFIERS_ONE_QUBIT = (
        PauliY(Qubit(0)),
        InvertiblePauliY(Qubit(0)),
        [PauliY(Qubit(0))],
        [InvertiblePauliY(Qubit(0))],
        MeasurementPauliProduct(PauliY(Qubit(0))),
        MeasurementPauliProduct(InvertiblePauliY(Qubit(0))),
    )
    EQUAL_MPP_QUBIT_IDENTIFIERS_TWO_QUBIT = (
        MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
        MeasurementPauliProduct(
            [InvertiblePauliX(Qubit(0)), InvertiblePauliY(Qubit(1))]
        ),
        [PauliX(Qubit(0)), PauliY(Qubit(1))],
        [InvertiblePauliX(Qubit(0)), InvertiblePauliY(Qubit(1))],
    )
    MPP_QUBIT_IDENTIFIERS = (
        PauliX(Qubit(0)),
        InvertiblePauliX(Qubit(1)),
        [PauliX(Qubit(2))],
        [InvertiblePauliX(Qubit(3))],
        MeasurementPauliProduct(PauliX(Qubit(4))),
        MeasurementPauliProduct(InvertiblePauliX(Qubit(5))),
        ~InvertiblePauliX(Qubit(1)),
        [~InvertiblePauliX(Qubit(3))],
        MeasurementPauliProduct(~InvertiblePauliX(Qubit(5))),
        MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
        [PauliX(Qubit(2)), PauliY(Qubit(3))],
        MeasurementPauliProduct([PauliZ(Qubit(0)), InvertiblePauliX(Qubit(1))]),
        [PauliZ(Qubit(2)), InvertiblePauliX(Qubit(3))],
        MeasurementPauliProduct([PauliX(Qubit(0)), ~InvertiblePauliY(Qubit(1))]),
        [PauliX(Qubit(2)), ~InvertiblePauliY(Qubit(3))],
    )

    @pytest.mark.parametrize("qubit_identifier", MPP_QUBIT_IDENTIFIERS)
    @pytest.mark.parametrize("bad_probability", [1.1, -0.2, 5])
    def test_error_is_raised_if_probability_is_out_of_bounds_for_pauli_product_measurement(
        self, bad_probability, qubit_identifier
    ):
        with pytest.raises(
            ValueError, match="Probability must be between zero and one."
        ):
            gates.MPP(qubit_identifier, bad_probability)

    @pytest.mark.parametrize("qubit_identifier", MPP_QUBIT_IDENTIFIERS)
    @pytest.mark.parametrize("probability", [0.0, 0.2])
    def test_mpp_gates_on_same_qubit_identifier_are_equal(
        self, qubit_identifier, probability
    ):
        assert gates.MPP(qubit_identifier, probability) == gates.MPP(
            qubit_identifier, probability
        )

    @pytest.mark.parametrize("qubit_identifier", MPP_QUBIT_IDENTIFIERS)
    @pytest.mark.parametrize("probability", [0.0, 0.2])
    def test_mpp_gates_on_the_same_qubit_identifiers_have_the_same_hash(
        self, qubit_identifier, probability
    ):
        assert hash(gates.MPP(qubit_identifier, probability)) == hash(
            gates.MPP(qubit_identifier, probability)
        )

    @pytest.mark.parametrize("qubit_identifier", MPP_QUBIT_IDENTIFIERS)
    def test_mpp_gates_with_different_probabilities_are_not_equal(
        self, qubit_identifier
    ):
        assert gates.MPP(qubit_identifier, 0.1) != gates.MPP(qubit_identifier, 0.2)

    @pytest.mark.parametrize(
        "qubit_identifier1, qubit_identifier2", permutations(MPP_QUBIT_IDENTIFIERS, 2)
    )
    def test_mpp_gate_on_different_qubit_identifiers_are_not_equal(
        self, qubit_identifier1, qubit_identifier2
    ):
        assert gates.MPP(qubit_identifier1, 0.2) != gates.MPP(qubit_identifier2, 0.2)

    @pytest.mark.parametrize(
        "qubit_identifier1, qubit_identifier2",
        permutations(EQUAL_MPP_QUBIT_IDENTIFIERS_ONE_QUBIT, 2),
    )
    def test_mpp_gate_with_equivalent_one_qubit_identifiers_are_equal(
        self, qubit_identifier1, qubit_identifier2
    ):
        assert gates.MPP(qubit_identifier1, 0.2) == gates.MPP(qubit_identifier2, 0.2)

    @pytest.mark.parametrize(
        "qubit_identifier1, qubit_identifier2",
        permutations(EQUAL_MPP_QUBIT_IDENTIFIERS_TWO_QUBIT, 2),
    )
    def test_mpp_gate_with_equivalent_two_qubit_identifiers_are_equal(
        self, qubit_identifier1, qubit_identifier2
    ):
        assert gates.MPP(qubit_identifier1, 0.2) == gates.MPP(qubit_identifier2, 0.2)

    @pytest.mark.parametrize("qubit_identifier", MPP_QUBIT_IDENTIFIERS)
    def test_mpp_gates_are_approximately_equal_if_probabilities_are_within_tolerance(
        self, qubit_identifier
    ):
        assert gates.MPP(qubit_identifier, 0.2).approx_equals(
            gates.MPP(qubit_identifier, 0.200001), rel_tol=0.0001
        )

    @pytest.mark.parametrize("qubit_identifier", MPP_QUBIT_IDENTIFIERS)
    def test_mpp_gates_with_probabilities_outside_tolerance_are_not_approx_equal(
        self, qubit_identifier
    ):
        assert not gates.MPP(qubit_identifier, 0.1).approx_equals(
            gates.MPP(qubit_identifier, 0.2)
        )

    @pytest.mark.parametrize(
        "qubit_identifier1, qubit_identifier2", permutations(MPP_QUBIT_IDENTIFIERS, 2)
    )
    def test_mpp_gate_on_different_qubit_identifiers_are_not_approx_equal(
        self, qubit_identifier1, qubit_identifier2
    ):
        assert not gates.MPP(qubit_identifier1, 0.2).approx_equals(
            gates.MPP(qubit_identifier2, 0.2)
        )

    @pytest.mark.parametrize(
        "qubit_identifier, expected_pauli_product",
        [
            (MeasurementPauliProduct(PauliX(0)), MeasurementPauliProduct(PauliX(0))),
            (
                MeasurementPauliProduct([InvertiblePauliY(0)]),
                MeasurementPauliProduct(PauliY(0)),
            ),
            (PauliZ(4), MeasurementPauliProduct(PauliZ(4))),
            (InvertiblePauliZ(4), MeasurementPauliProduct(PauliZ(4))),
            (~InvertiblePauliZ(4), MeasurementPauliProduct(~InvertiblePauliZ(4))),
            ([PauliX(3)], MeasurementPauliProduct(PauliX(3))),
            ([InvertiblePauliX(3)], MeasurementPauliProduct(PauliX(3))),
            ((PauliX(3),), MeasurementPauliProduct(PauliX(3))),
            ((InvertiblePauliX(3),), MeasurementPauliProduct(PauliX(3))),
            (
                MeasurementPauliProduct([PauliY(0), InvertiblePauliY(1)]),
                MeasurementPauliProduct([PauliY(0), PauliY(1)]),
            ),
            (
                [PauliX(0), InvertiblePauliZ(1)],
                MeasurementPauliProduct([PauliX(0), PauliZ(1)]),
            ),
            (
                (PauliX(0), InvertiblePauliZ(1)),
                MeasurementPauliProduct([PauliX(0), PauliZ(1)]),
            ),
        ],
    )
    def test_mpp_gate_pauli_product_is_correct(
        self, qubit_identifier, expected_pauli_product
    ):
        assert gates.MPP(qubit_identifier).pauli_product == expected_pauli_product

    def test_error_is_raised_when_acting_gate_on_same_qubits(self):
        with pytest.raises(
            ValueError, match="Pauli product cannot contain duplicate qubits"
        ):
            gates.MPP((PauliX(0), InvertiblePauliY(0)))


class TestQubitTransforms:
    @pytest.mark.parametrize(
        "measurement_gate_class", gates.MEASUREMENT_GATES - {gates.MPP}
    )
    def test_one_qubit_measurement_gate_qubit_does_not_change_if_id_not_in_mapping(
        self, measurement_gate_class
    ):
        qubit = Qubit(0)
        measurement_gate = measurement_gate_class(qubit)
        measurement_gate.transform_qubits({})
        assert measurement_gate.qubit is qubit

    @pytest.mark.parametrize(
        "measurement_gate_class", gates.MEASUREMENT_GATES - {gates.MPP}
    )
    def test_one_qubit_measurement_gate_qubit_changes_if_id_in_mapping(
        self, measurement_gate_class
    ):
        measurement_gate = measurement_gate_class(Qubit(0))
        measurement_gate.transform_qubits({0: 1})
        assert measurement_gate.qubit == Qubit(1)

    def test_mpp_pauli_product_does_not_change_if_ids_not_in_mapping(self):
        pauli_product = MeasurementPauliProduct([PauliX(0), InvertiblePauliY(1)])
        measurement_gate = gates.MPP(pauli_product)
        measurement_gate.transform_qubits({})
        assert measurement_gate.pauli_product == pauli_product

    def test_mpp_pauli_product_transforms_if_ids_in_mapping(self):
        measurement_gate = gates.MPP([PauliX(0), InvertiblePauliZ(1)])
        measurement_gate.transform_qubits({0: 2, 1: 3})
        assert measurement_gate.pauli_product == MeasurementPauliProduct(
            [PauliX(2), InvertiblePauliZ(3)]
        )
