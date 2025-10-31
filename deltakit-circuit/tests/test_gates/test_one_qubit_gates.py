# (c) Copyright Riverlane 2020-2025.
from itertools import permutations
import itertools

import pytest
import stim
from deltakit_circuit import gates
from deltakit_circuit._qubit_identifiers import Qubit


@pytest.mark.parametrize(
    "one_qubit_gate, expected_string",
    [
        (gates.I, "I"),
        (gates.X, "X"),
        (gates.Y, "Y"),
        (gates.Z, "Z"),
        (gates.H, "H"),
        (gates.H_XY, "H_XY"),
        (gates.H_YZ, "H_YZ"),
        (gates.C_XYZ, "C_XYZ"),
        (gates.C_ZYX, "C_ZYX"),
        (gates.S, "S"),
        (gates.S_DAG, "S_DAG"),
        (gates.SQRT_X, "SQRT_X"),
        (gates.SQRT_X_DAG, "SQRT_X_DAG"),
        (gates.SQRT_Y, "SQRT_Y"),
        (gates.SQRT_Y_DAG, "SQRT_Y_DAG"),
    ],
)
def test_one_qubit_gate_stim_string_matches_expected_string(
    one_qubit_gate, expected_string
):
    assert one_qubit_gate.stim_string == expected_string


@pytest.mark.parametrize(
    "one_qubit_gate,tag",
    itertools.product(gates.ONE_QUBIT_GATES, [None, "", "sjkdhf", "λ", "leaky<0>"]),
)
def test_one_qubit_gates_repr_matches_expected_representation(
    one_qubit_gate, tag: str | None
) -> None:
    tag_repr = f"[{tag}]" if tag is not None else ""
    assert (
        repr(one_qubit_gate(Qubit(0), tag=tag))
        == f"{one_qubit_gate.stim_string}{tag_repr}(Qubit(0))"
    )


@pytest.mark.parametrize("one_qubit_gate", gates.ONE_QUBIT_GATES)
def test_one_qubit_gates_on_the_same_qubit_are_equal(one_qubit_gate):
    assert one_qubit_gate(Qubit(0)) == one_qubit_gate(Qubit(0))


@pytest.mark.parametrize("one_qubit_gate", gates.ONE_QUBIT_GATES)
def test_one_qubit_gates_on_the_same_qubit_have_the_same_hash(one_qubit_gate):
    assert hash(one_qubit_gate(Qubit(0))) == hash(one_qubit_gate(Qubit(0)))


@pytest.mark.parametrize("one_qubit_gate", gates.ONE_QUBIT_GATES)
def test_one_qubit_gates_on_different_qubits_are_not_equal(one_qubit_gate):
    assert one_qubit_gate(Qubit(0)) != one_qubit_gate(Qubit(1))


@pytest.mark.parametrize(
    "one_qubit_gate1, one_qubit_gate2", permutations(gates.ONE_QUBIT_GATES, 2)
)
def test_different_one_qubit_gates_on_same_qubit_are_different(
    one_qubit_gate1, one_qubit_gate2
):
    assert one_qubit_gate1(Qubit(0)) != one_qubit_gate2(Qubit(0))


@pytest.mark.parametrize("one_qubit_gate_class", gates.ONE_QUBIT_GATES)
def test_qubit_property_of_gates_is_qubit_type_when_passed_generic_type(
    one_qubit_gate_class,
):
    assert isinstance(one_qubit_gate_class(1).qubit, Qubit)


@pytest.mark.parametrize("one_qubit_gate_class", gates.ONE_QUBIT_GATES)
def test_stim_targets_for_one_qubit_gate_are_stim_gate_target_instances(
    one_qubit_gate_class,
):
    gate = one_qubit_gate_class(Qubit(0))
    assert isinstance(gate.stim_targets({Qubit(0): 0})[0], stim.GateTarget)


@pytest.mark.parametrize("reset_gate_type", [gates.RX, gates.RY, gates.RZ])
def test_reset_gates_are_not_identified_as_clifford(reset_gate_type):
    assert not isinstance(reset_gate_type, gates.OneQubitCliffordGate)


@pytest.mark.parametrize("one_qubit_gate_class", gates.ONE_QUBIT_GATES)
def test_one_qubit_gate_qubit_does_not_change_if_qubit_not_in_id_mapping(
    one_qubit_gate_class,
):
    qubit = Qubit(0)
    gate = one_qubit_gate_class(qubit)
    gate.transform_qubits({})
    assert gate.qubit is qubit


@pytest.mark.parametrize("one_qubit_gate_class", gates.ONE_QUBIT_GATES)
def test_one_qubit_gate_qubit_changes_if_qubit_in_id_mapping(one_qubit_gate_class):
    gate = one_qubit_gate_class(Qubit(0))
    gate.transform_qubits({0: 1})
    assert gate.qubit == Qubit(1)


def test_gates_are_not_produced_by_a_generator():
    with pytest.raises(TypeError):
        gates.X(Qubit(i) for i in [1, 2, 3])
