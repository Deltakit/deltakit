# (c) Copyright Riverlane 2020-2025.
import pytest

from deltakit_circuit._qubit_identifiers import Qubit
from deltakit_circuit._qubit_mapping import default_qubit_mapping


def test_default_qubit_mapping_for_integer_qubits_returns_the_unique_identifier():
    qubits = [Qubit(i) for i in range(10)]
    assert default_qubit_mapping(qubits) == {Qubit(i): i for i in range(10)}


def test_default_qubit_mapping_for_non_integer_qubits_is_position_in_list():
    qubits = [Qubit(letter) for letter in "abcdefg"]
    assert default_qubit_mapping(qubits) == {
        qubit: index for index, qubit in enumerate(qubits)
    }


def test_default_qubit_mapping_raises_exception_type_of_qubit_uids_is_not_always_the_same():
    qubits = [Qubit((0, 0)), Qubit(0)]
    with pytest.raises(
        TypeError, match="All Qubit.unique_identifier fields must be of the same type"
    ):
        default_qubit_mapping(qubits)
