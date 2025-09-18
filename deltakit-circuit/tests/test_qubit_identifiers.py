# (c) Copyright Riverlane 2020-2025.
from copy import deepcopy

import pytest
import stim

import deltakit_circuit as sp
from deltakit_circuit import (
    Coordinate,
    InvertiblePauliX,
    InvertiblePauliY,
    InvertiblePauliZ,
    MeasurementPauliProduct,
    MeasurementRecord,
    PauliProduct,
    PauliX,
    PauliY,
    PauliZ,
    Qubit,
    SweepBit,
)
from deltakit_circuit._qubit_identifiers import PauliGate

IDENTICAL_QUBIT_PAIRS = [
    (Qubit(2), Qubit(2)),
    (Qubit(Coordinate(0, 1, 2)), Qubit(Coordinate(0, 1, 2))),
    (SweepBit(3), SweepBit(3)),
    (InvertiblePauliX(Qubit(2)), PauliX(Qubit(2))),
    (PauliX(Qubit(2)), PauliX(Qubit(2))),
    (PauliY(Qubit(2)), PauliY(Qubit(2))),
    (PauliZ(Qubit(2)), PauliZ(Qubit(2))),
    (PauliX(Qubit(2)), InvertiblePauliX(Qubit(2))),
    (PauliY(Qubit(2)), InvertiblePauliY(Qubit(2))),
    (PauliZ(Qubit(2)), InvertiblePauliZ(Qubit(2))),
    (~(~InvertiblePauliX(Qubit(2))), InvertiblePauliX(Qubit(2))),
    (~(~InvertiblePauliY(Qubit(2))), InvertiblePauliY(Qubit(2))),
    (~(~InvertiblePauliZ(Qubit(2))), InvertiblePauliZ(Qubit(2))),
    (
        MeasurementPauliProduct([PauliX(Qubit(0)), PauliZ(Qubit(1))]),
        MeasurementPauliProduct([PauliX(Qubit(0)), PauliZ(Qubit(1))]),
    ),
    (
        MeasurementPauliProduct([PauliX(Qubit(0)), InvertiblePauliY(Qubit(1))]),
        MeasurementPauliProduct([PauliX(Qubit(0)), InvertiblePauliY(Qubit(1))]),
    ),
    (
        MeasurementPauliProduct([InvertiblePauliX(Qubit(0)), PauliY(Qubit(1))]),
        MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
    ),
    (
        MeasurementPauliProduct([PauliZ(Qubit(0))]),
        MeasurementPauliProduct([PauliZ(Qubit(0))]),
    ),
    (
        MeasurementPauliProduct([InvertiblePauliX(Qubit(0))]),
        MeasurementPauliProduct([PauliX(Qubit(0))]),
    ),
    (
        PauliProduct([PauliX(Qubit(0)), PauliX(Qubit(1))]),
        PauliProduct([PauliX(Qubit(0)), PauliX(Qubit(1))]),
    ),
    (InvertiblePauliX(Qubit(Coordinate(2, 0))), PauliX(Qubit(Coordinate(2, 0)))),
    (~(~InvertiblePauliZ(Coordinate(4, 6))), InvertiblePauliZ(Coordinate(4, 6))),
    (
        PauliProduct([PauliX(Coordinate(0, 0)), PauliX(Coordinate(1, 1))]),
        PauliProduct([PauliX(Coordinate(0, 0)), PauliX(Coordinate(1, 1))]),
    ),
]


@pytest.mark.parametrize("qubit1, qubit2", IDENTICAL_QUBIT_PAIRS)
def test_two_identical_qubits_are_equal(qubit1, qubit2):
    assert qubit1 == qubit2


@pytest.mark.parametrize("qubit1, qubit2", IDENTICAL_QUBIT_PAIRS)
def test_two_equal_qubits_have_the_same_hash(qubit1, qubit2):
    assert hash(qubit1) == hash(qubit2)


def test_warning_is_raised_if_calling_pairs_from_consecutive_method():
    with pytest.warns(DeprecationWarning):
        list(Qubit.pairs_from_consecutive([0, 1]))


def test_error_is_raised_if_calling_pairs_from_consecutive_with_odd_sequence():
    with pytest.raises(
        ValueError, match="Pairs cannot be constructed from an odd number of IDs."
    ):
        list(Qubit.pairs_from_consecutive([0, 1, 2]))


def test_pairs_from_consecutive_returns_correct_qubit_pairs():
    assert list(Qubit.pairs_from_consecutive([0, 1, 2, 3])) == [
        (Qubit(0), Qubit(1)),
        (Qubit(2), Qubit(3)),
    ]


@pytest.mark.parametrize(
    "qubit1, qubit2",
    [
        (Qubit(2), Qubit(3)),
        (Qubit(Coordinate(0, 1, 2)), Qubit((0, 1, 2))),
        (Coordinate(0, 1, 2), (0, 1, 2)),
        (Qubit(0), Qubit((0, 1))),
        (Qubit(1), 3),
        (SweepBit(3), SweepBit(2)),
        (PauliX(Qubit(2)), PauliX(Qubit(4))),
        (PauliY(Qubit(2)), PauliY(Qubit(4))),
        (PauliZ(Qubit(2)), PauliZ(Qubit(4))),
        (PauliX(Qubit(2)), PauliY(Qubit(2))),
        (PauliX(Qubit(4)), PauliZ(Qubit(4))),
        (PauliY(Qubit(3)), PauliZ(Qubit(3))),
        (PauliY(Qubit((3, 2))), PauliZ(Coordinate(3, 2))),
        (InvertiblePauliX(Qubit(2)), InvertiblePauliX(Qubit(3))),
        (InvertiblePauliY(Qubit(2)), InvertiblePauliY(Qubit(3))),
        (InvertiblePauliZ(Qubit(2)), InvertiblePauliZ(Qubit(3))),
        (InvertiblePauliX(Qubit(3)), InvertiblePauliY(Qubit(3))),
        (InvertiblePauliX(Qubit(3)), InvertiblePauliZ(Qubit(3))),
        (InvertiblePauliY(Qubit(3)), InvertiblePauliZ(Qubit(3))),
        (~InvertiblePauliX(Qubit(2)), PauliX(Qubit(2))),
        (PauliZ(Qubit(1)), ~InvertiblePauliZ(Qubit(1))),
        (
            MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
            MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(2))]),
        ),
        (
            MeasurementPauliProduct([PauliX(Qubit(0)), PauliZ(Qubit(1))]),
            MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
        ),
        (
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(2))]),
        ),
        (
            PauliProduct([PauliX(Qubit(0)), PauliZ(Qubit(1))]),
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
        ),
        (
            MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
        ),
    ],
)
def test_different_qubits_are_not_equal(qubit1, qubit2):
    assert qubit1 != qubit2


@pytest.mark.parametrize(
    "qubit, expected_representation",
    [
        (Qubit(4), "Qubit(4)"),
        (Qubit((0, 2)), "Qubit((0, 2))"),
        (SweepBit(5), "SweepBit(5)"),
        (MeasurementRecord(-3), "MeasurementRecord(-3)"),
        (PauliX(Qubit(1)), "PauliX(Qubit(1))"),
        (PauliY(Qubit(0)), "PauliY(Qubit(0))"),
        (PauliZ(Qubit(4)), "PauliZ(Qubit(4))"),
        (InvertiblePauliX(Qubit(1)), "PauliX(Qubit(1))"),
        (InvertiblePauliY(Qubit(0)), "PauliY(Qubit(0))"),
        (InvertiblePauliZ(Qubit(4)), "PauliZ(Qubit(4))"),
        (~InvertiblePauliX(Qubit(1)), "!PauliX(Qubit(1))"),
        (~InvertiblePauliY(Qubit(0)), "!PauliY(Qubit(0))"),
        (~InvertiblePauliZ(Qubit(4)), "!PauliZ(Qubit(4))"),
        (
            MeasurementPauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]),
            "[PauliX(Qubit(0)), PauliY(Qubit(1))]",
        ),
        (
            MeasurementPauliProduct([~InvertiblePauliZ(Qubit(0)), PauliX(Qubit(2))]),
            "[!PauliZ(Qubit(0)), PauliX(Qubit(2))]",
        ),
        (
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(3))]),
            "[PauliX(Qubit(0)), PauliY(Qubit(3))]",
        ),
    ],
)
def test_representation_of_qubit_matches_expected_repr(qubit, expected_representation):
    assert repr(qubit) == expected_representation


def test_qubit_unique_identifier_matches_expected_identifier():
    assert Qubit(4).unique_identifier == 4


def test_sweep_bit_bit_index_is_the_same_as_given_index():
    assert SweepBit(2).bit_index == 2


def test_error_is_raised_if_sweep_bit_index_is_negative():
    with pytest.raises(
        ValueError, match="Sweep bit index cannot be a negative number."
    ):
        SweepBit(-2)


class TestPauliProducts:
    @pytest.mark.parametrize(
        "pauli_gate, expected_string",
        [
            (PauliX, "X"),
            (PauliY, "Y"),
            (PauliZ, "Z"),
            (InvertiblePauliX, "X"),
            (InvertiblePauliY, "Y"),
            (InvertiblePauliZ, "Z"),
        ],
    )
    def test_pauli_gate_stim_identifier_matches_expected_identifier(
        self, pauli_gate, expected_string
    ):
        assert pauli_gate.stim_identifier == expected_string

    @pytest.mark.parametrize(
        "pauli_gate_type",
        [
            PauliX,
            PauliY,
            PauliZ,
            InvertiblePauliX,
            InvertiblePauliY,
            InvertiblePauliZ,
        ],
    )
    def test_type_of_qubit_for_pauli_gates_is_qubit_when_given_plain_type(
        self, pauli_gate_type
    ):
        assert isinstance(pauli_gate_type(1).qubit, Qubit)

    @pytest.mark.parametrize(
        "invertible_gate",
        [
            InvertiblePauliX(Qubit(3)),
            InvertiblePauliY(Qubit(2)),
            InvertiblePauliZ(Qubit(5)),
        ],
    )
    def test_inverted_gate_has_the_same_qubit_as_non_inverted_gate(
        self, invertible_gate
    ):
        assert invertible_gate.qubit == (~invertible_gate).qubit

    @pytest.mark.parametrize(
        "pauli_product_class", [PauliProduct, MeasurementPauliProduct]
    )
    def test_error_is_raised_if_zero_in_pauli_product(self, pauli_product_class):
        with pytest.raises(
            ValueError,
            match="There must be at least one Pauli gate in a Pauli product.",
        ):
            pauli_product_class([])

    @pytest.mark.parametrize(
        "pauli_product",
        [
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1)), PauliZ(Qubit(2))]),
            MeasurementPauliProduct(
                [PauliX(Qubit(0)), PauliY(Qubit(1)), PauliZ(Qubit(2))]
            ),
            MeasurementPauliProduct(
                [
                    InvertiblePauliX(Qubit(0)),
                    InvertiblePauliY(Qubit(1)),
                    InvertiblePauliZ(Qubit(2)),
                ]
            ),
        ],
    )
    def test_all_items_in_gate_target_are_stim_gate_targets(self, pauli_product):
        qubit_mapping = {
            qubit: qubit.unique_identifier for qubit in pauli_product.qubits
        }
        assert all(
            isinstance(target, stim.GateTarget)
            for target in pauli_product.stim_targets(qubit_mapping)
        )

    def test_all_odd_items_in_measurement_pauli_product_stim_targets_are_target_combiners(
        self,
    ):
        pauli_product = MeasurementPauliProduct(
            [PauliX(Qubit(0)), InvertiblePauliY(Qubit(1)), PauliZ(Qubit(2))]
        )
        qubit_mapping = {
            qubit: qubit.unique_identifier for qubit in pauli_product.qubits
        }
        assert all(
            target == stim.target_combiner()
            for target in pauli_product.stim_targets(qubit_mapping)[1::2]
        )

    def test_target_combiner_isnt_in_the_pauli_product_stim_targets(self):
        pauli_product = PauliProduct(
            [PauliX(Qubit(0)), PauliY(Qubit(1)), PauliZ(Qubit(2))]
        )
        qubit_mapping = {
            qubit: qubit.unique_identifier for qubit in pauli_product.qubits
        }
        assert stim.target_combiner() not in pauli_product.stim_targets(qubit_mapping)


@pytest.mark.parametrize("pauli_product_class", [PauliProduct, MeasurementPauliProduct])
def test_error_is_raised_if_any_qubits_in_pauli_product_are_equal(pauli_product_class):
    with pytest.raises(
        ValueError, match="Pauli product cannot contain duplicate qubits."
    ):
        pauli_product_class((PauliX(0), PauliZ(0)))


def test_coordinate_qubits_are_given_a_declaration_when_converted_into_stim(
    empty_circuit,
):
    qubit = Qubit(Coordinate(5, 0))
    qubit.permute_stim_circuit(empty_circuit, qubit_mapping={qubit: 4})
    assert empty_circuit == stim.Circuit("QUBIT_COORDS(5, 0) 4")


def test_qubit_not_parametrized_by_coordinates_has_an_empty_stim_circuit(empty_circuit):
    Qubit((0, 1, 2)).permute_stim_circuit(empty_circuit, {})
    assert empty_circuit == stim.Circuit()


def test_deltakit_circuit_coordinate_qubits_can_propagate_1_dimensional_coordinates_to_stim(
    empty_circuit,
):
    qubit = Qubit(Coordinate(0))
    qubit.permute_stim_circuit(empty_circuit, qubit_mapping={qubit: 0})
    assert empty_circuit == stim.Circuit("QUBIT_COORDS(0) 0")


def test_qubit_coordinates_are_not_output_to_stim_file_repeat_blocks():
    stim_circuit = sp.Circuit(
        sp.Circuit(sp.GateLayer(sp.gates.X(sp.Qubit(Coordinate(0, 0)))), 2)
    ).as_stim_circuit()
    for instruction in stim_circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            assert "QUBIT_COORDS" not in str(instruction.body_copy())


def test_deepcopying_coordinate_returns_equal_coordinate():
    coord = Coordinate(0, 1, 2, 3)
    assert deepcopy(coord) == coord


def test_deepcopy_of_coordinate_does_not_return_same_object():
    coord = Coordinate(0, 1, 2, 3)
    assert deepcopy(coord) is not coord


@pytest.mark.parametrize(
    "pauli_product",
    [
        PauliProduct(gate(i) for i, gate in enumerate([PauliX, PauliY, PauliZ])),
        MeasurementPauliProduct(
            gate(i)
            for i, gate in enumerate(
                [
                    PauliX,
                    PauliY,
                    PauliZ,
                    InvertiblePauliX,
                    InvertiblePauliY,
                    InvertiblePauliZ,
                ]
            )
        ),
    ],
)
def test_iterating_over_pauli_product_gives_the_pauli_gates_in_that_pauli_product(
    pauli_product,
):
    assert all(isinstance(pauli_gate, PauliGate) for pauli_gate in pauli_product)


def test_identifier_is_not_a_generator():
    with pytest.raises(TypeError):
        Qubit(i for i in [1, 2, 3])


def test_error_for_accessing_not_set_stim_id():
    with pytest.raises(ValueError, match=r".* has no stim identifier."):
        # ruff reports "useless attribute access" on the line below but the attribute
        # access is not really useless, as the goal is to call it and see if it raises
        # an exception, so this check is ignored for that line.
        Qubit((2, 34)).stim_identifier  # noqa: B018


def test_accessing_stim_id_when_set():
    assert Qubit(Coordinate(0, 1, 2, 3), 7).stim_identifier == 7


def test_accessing_stim_id_when_unique_id_can_be_used():
    assert Qubit(7).stim_identifier == 7


@pytest.mark.parametrize("rec", list(range(-1, -10, -1)))
def test_measurement_record_hash(rec: int):
    assert hash(MeasurementRecord(rec)) == hash(MeasurementRecord(rec))
