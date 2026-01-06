# (c) Copyright Riverlane 2020-2025.
import itertools

import pytest
from deltakit_circuit import PauliX, PauliY, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D
from deltakit_circuit.gates import PauliBasis

from deltakit_explorer.codes._stabiliser import Stabiliser

paulis_and_ancilla_qubits_pair_examples = [
    (
        (PauliX(Coord2D(3, 3)), PauliZ(Coord2D(3, 1))),
        Qubit(Coord2D(2, 2)),
    ),
    (
        (
            PauliZ(Coord2D(3, 1)),
            None,
            PauliX(Coord2D(3, 3)),
        ),
        None,
    ),
    (
        (
            PauliX(Coord2D(3, 3)),
            PauliZ(Coord2D(3, 1)),
            None,
            PauliX(Coord2D(1, 1)),
            PauliX(Coord2D(1, 3)),
        ),
        Qubit(Coord2D(2, 2)),
    ),
    (
        (
            PauliX(Coord2D(3, 3)),
            PauliZ(Coord2D(3, 1)),
            PauliX(Coord2D(1, 1)),
            PauliX(Coord2D(1, 3)),
        ),
        None,
    ),
]


class TestStabiliserClass:
    """Class for testing the Stabiliser class."""

    @pytest.mark.parametrize("other", [(1,), (PauliBasis.X)])
    def test_raises_error_if_multiplied_with_other_type(self, other):
        with pytest.raises(TypeError):
            Stabiliser(paulis=(PauliX(0), PauliY(1))) * other

    def test_raises_error_when_data_qubits_are_not_unique(self):
        with pytest.raises(
            ValueError, match="Data qubits given in paulis should be unique."
        ):
            Stabiliser(
                paulis=(PauliX((1, 1)), PauliX((1, 1))),
            )

    def test_raises_error_without_data_qubits(self):
        with pytest.raises(
            ValueError, match="Stabiliser was initialised without Pauli terms."
        ):
            Stabiliser(
                paulis=[],
            )

    @pytest.mark.parametrize("ancilla", [(1, 1), Qubit((1, 1))])
    def test_raises_error_when_ancilla_is_a_data_qubit(self, ancilla):
        with pytest.raises(
            ValueError, match="Ancilla qubit should be different from the data qubits."
        ):
            Stabiliser(
                paulis=(PauliX((1, 1)), PauliX((1, 3))),
                ancilla_qubit=ancilla,
            )

    @pytest.mark.parametrize(
        ("stabiliser", "expected_dqs"),
        [
            (
                Stabiliser(paulis=(PauliX(0), PauliY(1))),
                {Qubit(0), Qubit(1)},
            ),
            (
                Stabiliser(paulis=(PauliX(0), None, PauliY(1))),
                {Qubit(0), Qubit(1)},
            ),
            (
                Stabiliser(
                    paulis=(PauliX(0), PauliY(1)),
                    ancilla_qubit=2,
                ),
                {Qubit(0), Qubit(1)},
            ),
            (
                Stabiliser(
                    paulis=(PauliX(0), PauliY(1)),
                    ancilla_qubit=Qubit(2),
                ),
                {Qubit(0), Qubit(1)},
            ),
            (
                Stabiliser(
                    paulis=(PauliX(Qubit(0)), PauliY(Qubit(1))),
                    ancilla_qubit=Qubit(2),
                ),
                {Qubit(0), Qubit(1)},
            ),
            (
                Stabiliser(paulis=(PauliX(Coord2D(0, 0)), PauliX(Coord2D(1, 1)))),
                {Qubit(Coord2D(0, 0)), Qubit(Coord2D(1, 1))},
            ),
            (
                Stabiliser(
                    paulis=(
                        None,
                        PauliX(Coord2D(0, 0)),
                        PauliX(Coord2D(1, 1)),
                    )
                ),
                {Qubit(Coord2D(0, 0)), Qubit(Coord2D(1, 1))},
            ),
            (
                Stabiliser(
                    paulis=(PauliX(Coord2D(0, 0)), PauliX(Coord2D(1, 1))),
                    ancilla_qubit=Coord2D(2, 2),
                ),
                {Qubit(Coord2D(0, 0)), Qubit(Coord2D(1, 1))},
            ),
            (
                Stabiliser(
                    paulis=(PauliX(Qubit(Coord2D(0, 0))), PauliX(Qubit(Coord2D(1, 1)))),
                    ancilla_qubit=Qubit(Coord2D(2, 2)),
                ),
                {Qubit(Coord2D(0, 0)), Qubit(Coord2D(1, 1))},
            ),
        ],
    )
    def test_data_qubits_set_correctly(self, stabiliser, expected_dqs):
        assert stabiliser.data_qubits == expected_dqs

    @pytest.mark.parametrize(
        ("stabiliser", "expected_paulis"),
        [
            (
                Stabiliser(paulis=[PauliX(0), PauliY(1)]),
                (PauliX(0), PauliY(1)),
            ),
            (
                Stabiliser(paulis=[PauliX(Coord2D(0, 0)), PauliX(Coord2D(1, 1))]),
                (PauliX(Coord2D(0, 0)), PauliX(Coord2D(1, 1))),
            ),
            (
                Stabiliser(paulis=[PauliX(0), None, PauliY(1)]),
                (PauliX(0), None, PauliY(1)),
            ),
            (
                Stabiliser(
                    paulis=[
                        PauliX(Coord2D(0, 0)),
                        PauliX(Coord2D(1, 1)),
                        None,
                        None,
                    ]
                ),
                (
                    PauliX(Coord2D(0, 0)),
                    PauliX(Coord2D(1, 1)),
                    None,
                    None,
                ),
            ),
        ],
    )
    def test_paulis_set_correctly(self, stabiliser, expected_paulis):
        assert stabiliser.paulis == expected_paulis

    @pytest.mark.parametrize(("paulis", "ancilla"), paulis_and_ancilla_qubits_pair_examples)
    def test__eq__and__hash__work_correctly_for_equal_paulis(self, paulis, ancilla):
        stab_0 = Stabiliser(paulis=paulis, ancilla_qubit=ancilla)
        stab_1 = Stabiliser(paulis=paulis, ancilla_qubit=ancilla)
        assert stab_0 == stab_1
        assert hash(stab_0) == hash(stab_1)

    @pytest.mark.parametrize(
        ("paulis_ancilla_0", "paulis_ancilla_1"),
        itertools.combinations(paulis_and_ancilla_qubits_pair_examples, r=2),
    )
    def test__neq__and__hash__work_correctly_for_different_paulis(
        self, paulis_ancilla_0, paulis_ancilla_1
    ):
        stab_0 = Stabiliser(
            paulis=paulis_ancilla_0[0], ancilla_qubit=paulis_ancilla_0[1]
        )
        stab_1 = Stabiliser(
            paulis=paulis_ancilla_1[0], ancilla_qubit=paulis_ancilla_1[1]
        )
        assert stab_0 != stab_1
        assert hash(stab_0) != hash(stab_1)

    @pytest.mark.parametrize(
        ("pauli_1", "pauli_2", "expected_product"),
        [
            (
                Stabiliser(
                    paulis=(PauliX(Coord2D(1, 1)), PauliZ(Coord2D(1, 3))),
                    ancilla_qubit=Coord2D(0, 2),
                ),
                Stabiliser(
                    paulis=(PauliX(Coord2D(3, 1)), PauliZ(Coord2D(3, 3))),
                    ancilla_qubit=Coord2D(4, 2),
                ),
                Stabiliser(
                    paulis=(
                        PauliX(Coord2D(1, 1)),
                        PauliZ(Coord2D(1, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliZ(Coord2D(3, 3)),
                    ),
                ),
            ),
            (
                Stabiliser(
                    paulis=(
                        PauliX(Coord2D(1, 1)),
                        None,
                        PauliZ(Coord2D(1, 3)),
                    ),
                    ancilla_qubit=Coord2D(0, 2),
                ),
                Stabiliser(
                    paulis=(
                        PauliX(Coord2D(3, 1)),
                        None,
                        PauliZ(Coord2D(3, 3)),
                    ),
                    ancilla_qubit=Coord2D(4, 2),
                ),
                Stabiliser(
                    paulis=(
                        PauliX(Coord2D(1, 1)),
                        PauliZ(Coord2D(1, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliZ(Coord2D(3, 3)),
                    ),
                ),
            ),
        ],
    )
    def test__mul__works_correctly(self, pauli_1, pauli_2, expected_product):
        assert pauli_1 * pauli_2 == expected_product

    @pytest.mark.parametrize(
        ("stabiliser", "expected_operator_repr", "expected_repr"),
        [
            (
                Stabiliser(
                    paulis=(PauliX(1), None, PauliZ(2)),
                    ancilla_qubit=3,
                ),
                {PauliX(1), PauliZ(2)},
                'Stabiliser((PauliX(Qubit(1)), None, PauliZ(Qubit(2))), Qubit(3))',
            ),
            (
                Stabiliser(
                    paulis=(
                        PauliX(Coord2D(1, 1)),
                        None,
                        PauliZ(Coord2D(1, 3)),
                    ),
                    ancilla_qubit=Coord2D(0, 2),
                ),
                {PauliX(Coord2D(1, 1)), PauliZ(Coord2D(1, 3))},
                'Stabiliser((PauliX(Qubit(Coord2D(1, 1))), None, PauliZ(Qubit(Coord2D(1, 3)))), Qubit(Coord2D(0, 2)))',
            ),
        ],
    )
    def test_operator_repr(self, stabiliser, expected_operator_repr, expected_repr):
        assert stabiliser.operator_repr == expected_operator_repr
        assert repr(stabiliser) == expected_repr
