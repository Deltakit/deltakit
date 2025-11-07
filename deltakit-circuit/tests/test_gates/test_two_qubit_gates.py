# (c) Copyright Riverlane 2020-2025.
from itertools import chain, permutations, product
import itertools

import pytest
import stim

from deltakit_circuit import gates
from deltakit_circuit._qubit_identifiers import MeasurementRecord, Qubit, SweepBit

CONTROLLED_GATES = (
    gates.CX,
    gates.CXSWAP,
    gates.CY,
    gates.CZ,
    gates.CZSWAP,
    gates.XCX,
    gates.XCY,
    gates.XCZ,
    gates.YCX,
    gates.YCY,
    gates.YCZ,
)

UNCONTROLLED_GATES = (
    gates.ISWAP,
    gates.ISWAP_DAG,
    gates.SQRT_XX,
    gates.SQRT_XX_DAG,
    gates.SQRT_YY,
    gates.SQRT_YY_DAG,
    gates.SQRT_ZZ,
    gates.SQRT_ZZ_DAG,
    gates.SWAP,
)

SYMMETRIC_GATES = (
    gates.CZ,
    gates.CZSWAP,
    gates.SWAP,
    gates.ISWAP,
    gates.ISWAP_DAG,
    gates.SQRT_XX,
    gates.SQRT_XX_DAG,
    gates.SQRT_YY,
    gates.SQRT_YY_DAG,
    gates.SQRT_ZZ,
    gates.SQRT_ZZ_DAG,
    gates.XCX,
    gates.YCY,
)


@pytest.mark.parametrize(
    "two_qubit_gate, expected_string",
    [
        (gates.CX, "CX"),
        (gates.CXSWAP, "CXSWAP"),
        (gates.CY, "CY"),
        (gates.CZ, "CZ"),
        (gates.CZSWAP, "CZSWAP"),
        (gates.SWAP, "SWAP"),
        (gates.ISWAP, "ISWAP"),
        (gates.ISWAP_DAG, "ISWAP_DAG"),
        (gates.SQRT_XX, "SQRT_XX"),
        (gates.SQRT_XX_DAG, "SQRT_XX_DAG"),
        (gates.SQRT_YY, "SQRT_YY"),
        (gates.SQRT_YY_DAG, "SQRT_YY_DAG"),
        (gates.SQRT_ZZ, "SQRT_ZZ"),
        (gates.SQRT_ZZ_DAG, "SQRT_ZZ_DAG"),
        (gates.XCX, "XCX"),
        (gates.XCY, "XCY"),
        (gates.XCZ, "XCZ"),
        (gates.YCX, "YCX"),
        (gates.YCY, "YCY"),
        (gates.YCZ, "YCZ"),
    ],
)
def test_two_qubit_gate_stim_string_matches_expected_string(
    two_qubit_gate, expected_string
):
    assert two_qubit_gate.stim_string == expected_string


class TestFromConsecutive:
    @pytest.mark.parametrize("two_qubit_gate_type", gates.TWO_QUBIT_GATES)
    @pytest.mark.parametrize(
        "iterable, expected_qubit_pairs",
        [
            (
                range(6),
                [(Qubit(0), Qubit(1)), (Qubit(2), Qubit(3)), (Qubit(4), Qubit(5))],
            )
        ],
    )
    def test_two_qubit_gate_gates_from_consecutive_qubits_gives_expected_qubits(
        self, two_qubit_gate_type, iterable, expected_qubit_pairs
    ):
        qubit_pairs = [
            gate.qubits for gate in two_qubit_gate_type.from_consecutive(iterable)
        ]
        assert qubit_pairs == expected_qubit_pairs

    @pytest.mark.parametrize("two_qubit_gate_type", [gates.CX, gates.CY, gates.CZ])
    @pytest.mark.parametrize(
        "iterable", [[SweepBit(0), Qubit(3)], [MeasurementRecord(-1), Qubit(2)]]
    )
    def test_classically_controlled_gate_from_consecutive_gives_expected_control(
        self, two_qubit_gate_type, iterable
    ):
        assert isinstance(
            next(two_qubit_gate_type.from_consecutive(iterable)).control,
            type(iterable[0]),
        )

    @pytest.mark.parametrize("two_qubit_gate_type", [gates.XCZ, gates.YCZ])
    @pytest.mark.parametrize(
        "iterable", [[Qubit(3), SweepBit(0)], [Qubit(2), MeasurementRecord(-1)]]
    )
    def test_classically_targeted_gate_from_consecutive_gives_expected_target(
        self, two_qubit_gate_type, iterable
    ):
        assert isinstance(
            next(two_qubit_gate_type.from_consecutive(iterable)).target,
            type(iterable[1]),
        )


@pytest.mark.parametrize(
    "two_qubit_gate,tag",
    itertools.product(CONTROLLED_GATES, [None, "", "sjkdhf", "λ", "leaky<0>"]),
)
def test_repr_of_controlled_gates_matches_expected_representation(
    two_qubit_gate, tag: str | None
) -> None:
    tag_repr = f"[{tag}]" if tag is not None else ""
    assert (
        repr(two_qubit_gate(Qubit(1), Qubit(2), tag=tag))
        == f"{two_qubit_gate.stim_string}{tag_repr}(control=Qubit(1), target=Qubit(2))"
    )


@pytest.mark.parametrize(
    "two_qubit_gate,tag",
    itertools.product(UNCONTROLLED_GATES, [None, "", "sjkdhf", "λ", "leaky<0>"]),
)
def test_repr_of_uncontrolled_gate_matches_expected_representation(
    two_qubit_gate, tag: str | None
) -> None:
    tag_repr = f"[{tag}]" if tag is not None else ""
    assert (
        repr(two_qubit_gate(Qubit(1), Qubit(0), tag=tag))
        == f"{two_qubit_gate.stim_string}{tag_repr}(Qubit(1), Qubit(0))"
    )


@pytest.mark.parametrize("two_qubit_gate", gates.TWO_QUBIT_GATES)
def test_error_is_raised_if_consecutive_data_has_an_odd_number_of_elements(
    two_qubit_gate,
):
    with pytest.raises(
        ValueError,
        match="Two qubit gates can only be constructed from an even number of qubits",
    ):
        list(two_qubit_gate.from_consecutive([1, 2, 3]))


@pytest.mark.parametrize(
    "two_qubit_gate_class, operands",
    chain(
        product(
            gates.TWO_QUBIT_GATES, [(Qubit(0), Qubit(0)), (0, Qubit(0)), ("a", "a")]
        ),
        product(
            CONTROLLED_GATES,
            [
                (SweepBit(0), SweepBit(0)),
                (MeasurementRecord(-1), MeasurementRecord(-1)),
            ],
        ),
    ),
)
def test_error_is_raised_if_operands_of_two_qubit_gates_are_equal(
    two_qubit_gate_class, operands
):
    operand1, operand2 = operands
    with pytest.raises(
        ValueError, match="Operands for two qubit gates must be different."
    ):
        two_qubit_gate_class(operand1, operand2)


class TestEquality:
    @pytest.mark.parametrize("two_qubit_gate", [gates.CX, gates.CY, gates.CZ])
    @pytest.mark.parametrize("control", [Qubit(0), SweepBit(0), MeasurementRecord(-1)])
    def test_classically_controlled_gates_with_same_arguments_are_equal(
        self, two_qubit_gate, control
    ):
        gate1 = two_qubit_gate(control, Qubit(1))
        gate2 = two_qubit_gate(control, Qubit(1))
        assert gate1 == gate2
        assert hash(gate1) == hash(gate2)

    @pytest.mark.parametrize("two_qubit_gate", [gates.XCZ, gates.YCZ])
    @pytest.mark.parametrize("target", [Qubit(1), SweepBit(0), MeasurementRecord(-1)])
    def test_classically_targeted_gates_with_same_arguments_are_equal(
        self, two_qubit_gate, target
    ):
        gate1 = two_qubit_gate(Qubit(0), target)
        gate2 = two_qubit_gate(Qubit(0), target)
        assert gate1 == gate2
        assert hash(gate1) == hash(gate2)

    @pytest.mark.parametrize("two_qubit_gate", UNCONTROLLED_GATES)
    def test_uncontrolled_gates_with_same_arguments_are_equal(self, two_qubit_gate):
        gate1 = two_qubit_gate(Qubit(0), Qubit(1))
        gate2 = two_qubit_gate(Qubit(0), Qubit(1))
        assert gate1 == gate2
        assert hash(gate1) == hash(gate2)

    @pytest.mark.parametrize("two_qubit_gate", SYMMETRIC_GATES)
    def test_symmetric_gates_are_equal_regardless_of_the_order_of_qubits(
        self, two_qubit_gate
    ):
        gate1 = two_qubit_gate(Qubit(0), Qubit(1))
        gate2 = two_qubit_gate(Qubit(1), Qubit(0))
        assert gate1 == gate2
        assert hash(gate1) == hash(gate2)

    @pytest.mark.parametrize("two_qubit_gate", gates.TWO_QUBIT_GATES)
    def test_gates_with_different_qubits_are_not_equal(self, two_qubit_gate):
        assert two_qubit_gate(Qubit(0), Qubit(1)) != two_qubit_gate(Qubit(2), Qubit(3))

    @pytest.mark.parametrize(
        "two_qubit_gate1, two_qubit_gate2", permutations(gates.TWO_QUBIT_GATES, 2)
    )
    def test_different_two_qubit_gates_on_same_qubits_are_not_equal(
        self, two_qubit_gate1, two_qubit_gate2
    ):
        assert two_qubit_gate1(Qubit(0), Qubit(1)) != two_qubit_gate2(
            Qubit(0), Qubit(1)
        )

    @pytest.mark.parametrize("two_qubit_gate", [gates.CX, gates.CY, gates.CZ])
    @pytest.mark.parametrize(
        "control1, control2",
        permutations([Qubit(0), SweepBit(0), MeasurementRecord(-1)], 2),
    )
    def test_classically_controlled_gates_are_not_equal_if_control_is_different(
        self, two_qubit_gate, control1, control2
    ):
        assert two_qubit_gate(control1, Qubit(1)) != two_qubit_gate(control2, Qubit(1))

    @pytest.mark.parametrize("two_qubit_gate", [gates.XCZ, gates.YCZ])
    @pytest.mark.parametrize(
        "target1, target2",
        permutations([Qubit(0), SweepBit(0), MeasurementRecord(-1)], 2),
    )
    def test_classically_targeted_gates_are_not_equal_if_target_is_different(
        self, two_qubit_gate, target1, target2
    ):
        assert two_qubit_gate(Qubit(1), target1) != two_qubit_gate(Qubit(1), target2)


@pytest.mark.parametrize("two_qubit_gate", gates.TWO_QUBIT_GATES)
def test_stim_targets_method_returns_stim_gate_targets_when_input_is_qubits(
    two_qubit_gate,
):
    gate = two_qubit_gate(Qubit(0), Qubit(1))
    qubit_mapping = {Qubit(0): 0, Qubit(1): 1}
    assert all(
        isinstance(target, stim.GateTarget)
        for target in gate.stim_targets(qubit_mapping)
    )


@pytest.mark.parametrize(
    "two_qubit_gate, sweep_bit_index",
    [(gates.CX, 0), (gates.CY, 0), (gates.CZ, 0), (gates.XCZ, 1), (gates.YCZ, 1)],
)
def test_stim_targets_are_sweep_bits_when_given_to_gate(
    two_qubit_gate, sweep_bit_index
):
    gate = (
        two_qubit_gate(SweepBit(0), Qubit(0))
        if sweep_bit_index == 0
        else two_qubit_gate(Qubit(0), SweepBit(0))
    )
    qubit_mapping = {Qubit(0): 0}
    assert gate.stim_targets(qubit_mapping)[sweep_bit_index].is_sweep_bit_target


@pytest.mark.parametrize(
    "two_qubit_gate, record_index",
    [(gates.CX, 0), (gates.CY, 0), (gates.CZ, 0), (gates.XCZ, 1), (gates.YCZ, 1)],
)
def test_stim_targets_are_measurement_records_when_given_to_gate(
    two_qubit_gate, record_index
):
    gate = (
        two_qubit_gate(MeasurementRecord(-1), Qubit(0))
        if record_index == 0
        else two_qubit_gate(Qubit(0), MeasurementRecord(-1))
    )
    qubit_mapping = {Qubit(0): 0}
    assert gate.stim_targets(qubit_mapping)[record_index].is_measurement_record_target


class TestQubitTransforms:
    @pytest.mark.parametrize("two_qubit_gate_class", gates.TWO_QUBIT_GATES)
    def test_two_qubit_gate_qubits_do_not_change_if_id_not_in_mapping(
        self, two_qubit_gate_class
    ):
        qubit0, qubit1 = Qubit(0), Qubit(1)
        gate = two_qubit_gate_class(qubit0, qubit1)
        gate.transform_qubits({})
        assert gate.qubits[0] is qubit0
        assert gate.qubits[1] is qubit1

    @pytest.mark.parametrize("two_qubit_gate_class", gates.TWO_QUBIT_GATES)
    def test_two_qubit_gate_qubits_change_if_id_in_mapping(self, two_qubit_gate_class):
        gate = two_qubit_gate_class(Qubit(0), Qubit(1))
        gate.transform_qubits({0: 2, 1: 3})
        assert gate.qubits[0] == Qubit(2)
        assert gate.qubits[1] == Qubit(3)
