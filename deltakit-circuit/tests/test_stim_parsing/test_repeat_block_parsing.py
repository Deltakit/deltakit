# (c) Copyright Riverlane 2020-2025.
import pytest
import stim

import deltakit_circuit as sp


def test_parsing_empty_repeat_block_gives_empty_circuit():
    assert sp.Circuit.from_stim_circuit(stim.Circuit("REPEAT 5 {}")) == sp.Circuit(
        [], 5
    )


@pytest.mark.parametrize("repeat_num", [0, -1, -50, -100])
def test_repeat_block_must_specified_in_deltakit_circuit_to_be_repeated_at_least_1_time(
    repeat_num,
):
    with pytest.raises(
        ValueError,
        match="Stim does not allow repeat blocks to be repeated less than 1 time. "
        f"Requested repeats was {repeat_num}.",
    ):
        sp.Circuit(
            [
                sp.GateLayer(sp.gates.H(sp.Qubit(0))),
            ],
            iterations=repeat_num,
        )


def test_parsing_repeat_block_with_one_repeat_flattens_the_circuit():
    stim_circuit = stim.Circuit("""
        H 0
        REPEAT 1 {
            CX 0 1
            CZ 1 2
        }
    """)
    assert sp.Circuit.from_stim_circuit(stim_circuit) == sp.Circuit(
        [
            sp.GateLayer(sp.gates.H(sp.Qubit(0))),
            sp.GateLayer(sp.gates.CX(0, 1)),
            sp.GateLayer(sp.gates.CZ(1, 2)),
        ]
    )


def test_parsing_stim_circuit_which_is_just_a_repeat_block():
    stim_circuit = stim.Circuit("REPEAT 5 {\nX 0\n}")
    assert sp.Circuit.from_stim_circuit(stim_circuit) == sp.Circuit(
        sp.GateLayer(sp.gates.X(sp.Qubit(0))), iterations=5
    )


def test_parsing_stim_circuit_with_repeat_block_creates_expected_deltakit_circuit_circuit():
    stim_circuit = stim.Circuit("H 0\nREPEAT 4 {\nCX 0 1\nCZ 1 2\n}")
    assert sp.Circuit.from_stim_circuit(stim_circuit) == sp.Circuit(
        [
            sp.GateLayer(sp.gates.H(sp.Qubit(0))),
            sp.Circuit(
                [sp.GateLayer(sp.gates.CX(0, 1)), sp.GateLayer(sp.gates.CZ(1, 2))],
                iterations=4,
            ),
        ]
    )


def test_parsing_stim_circuit_with_multiple_repeat_blocks():
    stim_circuit = stim.Circuit("""
        H 0 3
        REPEAT 5 {
            CX 0 1
            CZ 1 2
        }
        REPEAT 4 {
            CX 2 3
            CZ 3 4
        }
    """)
    assert sp.Circuit.from_stim_circuit(stim_circuit) == sp.Circuit(
        [
            sp.GateLayer(sp.gates.H(sp.Qubit(i)) for i in (0, 3)),
            sp.Circuit(
                [sp.GateLayer(sp.gates.CX(0, 1)), sp.GateLayer(sp.gates.CZ(1, 2))],
                iterations=5,
            ),
            sp.Circuit(
                [sp.GateLayer(sp.gates.CX(2, 3)), sp.GateLayer(sp.gates.CZ(3, 4))],
                iterations=4,
            ),
        ]
    )


def test_parsing_stim_circuit_nested_repeat_blocks():
    stim_circuit = stim.Circuit("""
        REPEAT 5 {
            H 0
            REPEAT 3 {
                CX 0 1
                CZ 1 2
            }
        }
    """)
    assert sp.Circuit.from_stim_circuit(stim_circuit) == sp.Circuit(
        [
            sp.GateLayer(sp.gates.H(sp.Qubit(0))),
            sp.Circuit(
                [sp.GateLayer(sp.gates.CX(0, 1)), sp.GateLayer(sp.gates.CZ(1, 2))],
                iterations=3,
            ),
        ],
        iterations=5,
    )


def test_parsing_stim_circuit_that_contains_nested_repeat_blocks():
    stim_circuit = stim.Circuit("""
        X 0 1
        REPEAT 5 {
            H 0
            REPEAT 3 {
                CX 0 1
                CZ 1 2
            }
        }
    """)
    assert sp.Circuit.from_stim_circuit(stim_circuit) == sp.Circuit(
        [
            sp.GateLayer(sp.gates.X(sp.Qubit(i)) for i in (0, 1)),
            sp.Circuit(
                [
                    sp.GateLayer(sp.gates.H(sp.Qubit(0))),
                    sp.Circuit(
                        [
                            sp.GateLayer(sp.gates.CX(0, 1)),
                            sp.GateLayer(sp.gates.CZ(1, 2)),
                        ],
                        iterations=3,
                    ),
                ],
                iterations=5,
            ),
        ]
    )
