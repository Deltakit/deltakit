# (c) Copyright Riverlane 2020-2025.
from itertools import permutations
import itertools

import pytest
import stim
from deltakit_circuit import gates
from deltakit_circuit._qubit_identifiers import Qubit


@pytest.mark.parametrize(
    "reset_gate, expected_string",
    [
        (gates.RZ, "RZ"),
        (gates.RX, "RX"),
        (gates.RY, "RY"),
    ],
)
def test_reset_gate_stim_string_matches_expected_string(reset_gate, expected_string):
    assert reset_gate.stim_string == expected_string


@pytest.mark.parametrize(
    "reset_gate, expected_basis",
    [
        (gates.RZ, gates.PauliBasis.Z),
        (gates.RX, gates.PauliBasis.X),
        (gates.RY, gates.PauliBasis.Y),
    ],
)
def test_reset_gate_basis_is_expected_basis(reset_gate, expected_basis):
    assert reset_gate.basis == expected_basis


@pytest.mark.parametrize(
    "reset_gate,tag",
    itertools.product(gates.RESET_GATES, [None, "", "sjkdhf", "λ", "leaky<0>"]),
)
def test_repr_of_reset_gate_matches_the_expected_representation(
    reset_gate, tag: str | None
) -> None:
    tag_repr = f"[{tag}]" if tag is not None else ""
    assert (
        repr(reset_gate(Qubit(3), tag=tag))
        == f"{reset_gate.stim_string}{tag_repr}(Qubit(3))"
    )


@pytest.mark.parametrize("reset_gate", gates.RESET_GATES)
def test_reset_gates_on_same_qubit_are_equal(reset_gate):
    assert reset_gate(Qubit(0)) == reset_gate(Qubit(0))


@pytest.mark.parametrize("reset_gate", gates.RESET_GATES)
def test_reset_gates_on_same_qubit_have_the_same_hash(reset_gate):
    assert hash(reset_gate(Qubit(0))) == hash(reset_gate(Qubit(0)))


@pytest.mark.parametrize("reset_gate", gates.RESET_GATES)
def test_reset_gates_on_different_qubits_are_not_equal(reset_gate):
    assert reset_gate(Qubit(0)) != reset_gate(Qubit(1))


@pytest.mark.parametrize("reset_gate1, reset_gate2", permutations(gates.RESET_GATES, 2))
def test_different_reset_gates_on_same_qubit_are_not_equal(reset_gate1, reset_gate2):
    assert reset_gate1(Qubit(0)) != reset_gate2(Qubit(0))


@pytest.mark.parametrize("reset_gate_class", gates.RESET_GATES)
def test_qubit_property_of_gates_is_qubit_type_when_passed_generic_type(
    reset_gate_class,
):
    assert isinstance(reset_gate_class(0).qubit, Qubit)


@pytest.mark.parametrize("reset_gate_class", gates.RESET_GATES)
def test_stim_targets_are_stim_gate_target_instances(reset_gate_class):
    gate = reset_gate_class(Qubit(0))
    assert isinstance(gate.stim_targets({Qubit(0): 0})[0], stim.GateTarget)


@pytest.mark.parametrize("reset_gate_class", gates.ONE_QUBIT_GATES)
def test_reset_gate_qubit_does_not_change_if_qubit_not_in_id_mapping(reset_gate_class):
    qubit = Qubit(0)
    gate = reset_gate_class(qubit)
    gate.transform_qubits({})
    assert gate.qubit is qubit


@pytest.mark.parametrize("reset_gate_class", gates.ONE_QUBIT_GATES)
def test_reset_gate_qubit_changes_if_qubit_in_id_mapping(reset_gate_class):
    gate = reset_gate_class(Qubit(0))
    gate.transform_qubits({0: 1})
    assert gate.qubit == Qubit(1)
