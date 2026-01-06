# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import PauliX, PauliZ, Qubit

from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._css._stabiliser_code import StabiliserCode


class StabiliserCodeForTesting(StabiliserCode):
    def __init__(
        self,
        stabilisers,
        x_logical_operators,
        z_logical_operators,
        use_ancilla_qubits,
    ):
        super().__init__(
            stabilisers=stabilisers,
            x_logical_operators=x_logical_operators,
            z_logical_operators=z_logical_operators,
            use_ancilla_qubits=use_ancilla_qubits,
        )

    def encode_logical_zeroes(self) -> CSSStage:
        return CSSStage()

    def encode_logical_pluses(self) -> CSSStage:
        return CSSStage()

    def measure_x_logicals(self) -> CSSStage:
        return CSSStage()

    def measure_z_logicals(self) -> CSSStage:
        return CSSStage()


class TestLogicalOperators:
    @pytest.mark.parametrize(
        ("x_logical_operators", "z_logical_operators"),
        [
            ((frozenset({}),), (frozenset({}),)),
            ((frozenset({PauliX(Qubit(0))}),), (frozenset({PauliZ(Qubit(0))}),)),
            (
                (frozenset({PauliX(Qubit(0)), PauliX(Qubit(1))}),),
                (frozenset({PauliZ(Qubit(0)), PauliZ(Qubit(1))}),),
            ),
            (
                (frozenset({PauliX(Qubit(0))}), frozenset({PauliX(Qubit(1))})),
                (frozenset({PauliZ(Qubit(0))}), frozenset({PauliZ(Qubit(1))})),
            ),
            (
                (
                    frozenset({PauliX(Qubit(0)), PauliX(Qubit(1))}),
                    frozenset({PauliX(Qubit(2)), PauliX(Qubit(3))}),
                ),
                (
                    frozenset({PauliZ(Qubit(0)), PauliZ(Qubit(1))}),
                    frozenset({PauliZ(Qubit(2)), PauliZ(Qubit(3))}),
                ),
            ),
        ],
    )
    class TestLogicalOperatorProperty:
        def test_x_logical_operators_returns_correct_value(
            self, x_logical_operators, z_logical_operators
        ):
            css_code = StabiliserCodeForTesting(
                stabilisers=[],
                x_logical_operators=x_logical_operators,
                z_logical_operators=z_logical_operators,
                use_ancilla_qubits=False,
            )
            assert css_code.x_logical_operators == x_logical_operators

        def test_z_logical_operators_returns_correct_value(
            self, x_logical_operators, z_logical_operators
        ):
            css_code = StabiliserCodeForTesting(
                stabilisers=[],
                x_logical_operators=x_logical_operators,
                z_logical_operators=z_logical_operators,
                use_ancilla_qubits=False,
            )
            assert css_code.z_logical_operators == z_logical_operators
