import random
import re
from copy import deepcopy
from functools import reduce
from operator import add, mul

import numpy as np
import pytest
import stim
from deltakit_circuit import (Circuit, Detector, GateLayer, MeasurementRecord,
                              NoiseLayer, Observable, PauliX, Qubit,
                              ShiftCoordinates)
from deltakit_circuit._basic_types import Coord2D
from deltakit_circuit.gates import (CX, CXSWAP, CY, CZ, CZSWAP, ISWAP,
                                    ISWAP_DAG, MPP, MRX, MRY, MRZ, MX, MY, MZ,
                                    RX, RY, RZ, S_DAG, SQRT_X, SQRT_X_DAG,
                                    SQRT_XX, SQRT_XX_DAG, SQRT_Y, SQRT_Y_DAG,
                                    SQRT_YY, SQRT_YY_DAG, SQRT_ZZ, SQRT_ZZ_DAG,
                                    SWAP, XCX, XCY, XCZ, YCX, YCY, YCZ, Gate,
                                    H, I, PauliBasis, S, X, Y, Z)
from deltakit_circuit.noise_channels import Depolarise2
from deltakit_explorer.codes._css._css_code_experiment_circuit import \
    css_code_memory_circuit
from deltakit_explorer.codes._planar_code import (RotatedPlanarCode,
                                                  UnrotatedPlanarCode)
from deltakit_explorer.qpu._circuits._tableau_compile_functions import (
    _compile_measurement_to_native_gates_plus_unitaries,
    _compile_or_exchange_unitary_block,
    _compile_reset_and_meas_to_native_gates,
    _compile_reset_to_native_gates_plus_unitaries,
    _compile_two_qubit_gate_to_target,
    _compile_two_qubit_gates_to_native_gates, compile_circuit_to_native_gates)
from deltakit_explorer.qpu._circuits._tableau_functions import (
    CZ_TO_GATE_DICT, CZSWAP_TO_GATE_DICT, GATE_TO_CZ_DICT, GATE_TO_CZSWAP_DICT,
    MEAS_COMPILATION_LOOKUP_DICT, RESET_COMPILATION_LOOKUP_DICT,
    CompilationData, _create_circuit_from_compilation_data,
    _extract_structure_from_circuit, _get_compilation_dict,
    _get_tableau_from_sequence_of_1q_gates,
    _get_single_qubits_tableau_key_from_two_qubit_tableau,
    _get_relevant_dict_to_update,
    _get_compilation_with_measurement_after_unitaries,
    _get_compilation_with_projectors_before_unitaries,
    _get_compilation_with_two_qubit_gates, _is_identity_like)
from deltakit_explorer.qpu._native_gate_set import NativeGateSet

# define set of compilation dictionaries for testing
compilation_dict0 = {("+X", "+Z"): ()}
compilation_dict1 = {("+X", "+Z"): (), ("-X", "+Z"): ("Z",)}
compilation_dict2 = {("+X", "+Z"): (), ("+X", "-Z"): ("SQRT_X", "SQRT_X")}
compilation_dict3 = {("+X", "+Z"): (), ("+X", "-Z"): ("X",)}
compilation_dict4 = {("+X", "+Z"): (), ("+X", "-Z"): ("X",), ("+Z", "-X"): ("X", "H")}
compilation_dict5 = {("+X", "+Z"): (), ("+X", "-Z"): ("X",), ("+Z", "+X"): ("H")}
compilation_dict6 = {("+X", "+Z"): (), ("+Z", "-X"): ("SQRT_X", "SQRT_X", "H")}
compilation_dict7 = {
    ("+X", "+Z"): (),
    ("+Z", "-Y"): ("S", "SQRT_X"),
    ("+Z", "+X"): ("H",),
}
compilation_dict8 = {("+X", "+Z"): (), ("-Z", "+X"): ("SQRT_Y",)}
compilation_dict9 = {
    ("+X", "+Z"): (),
    ("+Z", "-Y"): ("S", "SQRT_X"),
    ("+Y", "+X"): ("SQRT_X", "S"),
}
compilation_dict10 = {("+X", "+Z"): (), ("+Z", "+X"): ("S", "SQRT_X", "S")}
compilation_dict11 = {("+X", "+Z"): (), ("-X", "-Z"): ("X", "Z"), ("-X", "+Z"): ("Z",)}
compilation_dict12 = {("+X", "+Z"): (), ("+Z", "+X"): ("H", "H", "H")}
compilation_dict13 = {("+X", "+Z"): (), ("+Z", "+X"): ("H",)}
compilation_dict14 = {("+X", "+Z"): (), ("+Z", "+X"): ("H",), ("-Z", "+X"): ("H", "X")}
compilation_dict15 = {("+X", "+Z"): (), ("+Y", "+Z"): ("S",)}
compilation_dict16 = {("+X", "+Z"): (), ("-X", "-Z"): ("X", "Z")}
compilation_dict17 = {
    ("+X", "+Z"): (),
    ("+Z", "+X"): ("H",),
    ("+Z", "-X"): ("H", "Z"),
    ("-Z", "+X"): ("H", "X"),
}
compilation_dict18 = {
    ("+X", "+Z"): (),
    ("-Z", "+X"): ("SQRT_Y",),
    ("-Z", "-X"): ("X", "SQRT_Y"),
}
compilation_dict19 = {("+X", "+Z"): (), ("-Y", "-X"): ("SQRT_Y_DAG", "SQRT_X")}
compilation_dict20 = {
    ("+X", "+Z"): (),
    ("-Z", "+Y"): ("SQRT_Y", "S"),
}
compilation_dict21 = {
    ("+X", "+Z"): (),
    ("-X", "+Z"): ("Z",),
    ("+X", "-Z"): ("X",),
    ("+Y", "+Z"): ("S",),
    ("-X", "-Z"): ("Y",),
    ("+Z", "+X"): ("H",),
    ("+X", "-Y"): ("SQRT_X",),
    ("-Y", "+Z"): ("S_DAG",),
    ("+Z", "-X"): ("H", "Z"),
    ("-Z", "+X"): ("H", "X"),
}
compilation_dict22 = {
    ("+X", "+Z"): (),
    ("-X", "+Z"): ("Z",),
    ("+Y", "+Z"): ("S",),
    ("-Y", "+Z"): ("S", "S"),
}
compilation_dict23 = {
    ("+X", "+Z"): (),
    ("-X", "+Z"): ("Z",),
    ("+X", "-Z"): ("X",),
    ("+Y", "+Z"): ("S",),
    ("-X", "-Z"): ("Y",),
    ("+Z", "+X"): ("H",),
    ("+X", "-Y"): ("SQRT_X",),
    ("-Y", "+Z"): ("S_DAG",),
    ("+Z", "-X"): ("H", "Z"),
    ("-Z", "+X"): ("H", "X"),
    ("+Z", "+Y"): ("H", "S"),
    ("-Y", "+X"): ("S", "H"),
    ("-Z", "+Y"): ("S", "H", "S_DAG", "H"),
}
compilation_dict_full = {
    ("+X", "+Z"): (),
    ("+Y", "+Z"): ("S",),
    ("-X", "+Z"): ("Z",),
    ("+X", "-Z"): ("X",),
    ("+Z", "+X"): ("H",),
    ("-Y", "+Z"): ("S", "Z"),
    ("-Y", "-Z"): ("S", "X"),
    ("-Y", "+X"): ("S", "H"),
    ("-X", "-Z"): ("Z", "X"),
    ("-Z", "+X"): ("Z", "H"),
    ("+Y", "-Z"): ("X", "S"),
    ("+Z", "-X"): ("X", "H"),
    ("+Z", "+Y"): ("H", "S"),
    ("+Y", "+X"): ("S", "Z", "H"),
    ("+Y", "-X"): ("S", "X", "H"),
    ("+X", "+Y"): ("S", "H", "S"),
    ("-Z", "-X"): ("Z", "X", "H"),
    ("-Z", "+Y"): ("Z", "H", "S"),
    ("-Y", "-X"): ("X", "S", "H"),
    ("+Z", "-Y"): ("X", "H", "S"),
    ("-Z", "-Y"): ("H", "S", "X"),
    ("+X", "-Y"): ("H", "S", "H"),
    ("-X", "+Y"): ("S", "Z", "H", "S"),
    ("-X", "-Y"): ("S", "X", "H", "S"),
}
gate_exchange_dict_full = {
    ("+Y", "+Z"): [[S]],
    ("-X", "+Z"): [[Z]],
    ("+X", "-Z"): [[X]],
    ("+Z", "+X"): [[H]],
    ("-Y", "+Z"): [[S, Z], [Z, S]],
    ("-Y", "-Z"): [[S, X]],
    ("-Y", "+X"): [[S, H]],
    ("-X", "-Z"): [[Z, X], [X, Z]],
    ("-Z", "+X"): [[Z, H], [H, X]],
    ("+Y", "-Z"): [[X, S]],
    ("+Z", "-X"): [[X, H], [H, Z]],
    ("+Z", "+Y"): [[H, S]],
    ("+Y", "+X"): [[S, Z, H], [S, H, X]],
    ("+Y", "-X"): [[S, X, H], [S, H, Z]],
    ("+X", "+Y"): [[S, H, S]],
    ("-Z", "-X"): [[Z, X, H], [Z, H, Z], [X, H, X]],
    ("-Z", "+Y"): [[Z, H, S]],
    ("-Y", "-X"): [[X, S, H]],
    ("+Z", "-Y"): [[X, H, S], [H, S, Z]],
    ("-Z", "-Y"): [[H, S, X]],
    ("+X", "-Y"): [[H, S, H]],
    ("-X", "+Y"): [[S, Z, H, S], [H, S, X, H], [H, S, H, Z]],
    ("-X", "-Y"): [[S, X, H, S], [S, H, S, Z], [Z, H, S, H]],
}

compilation_dict0_nosign = {("X", "Z"): ()}
compilation_dict1_nosign = {("X", "Z"): (), ("Z", "X"): ("H", "H", "H")}
compilation_dict2_nosign = {("X", "Z"): (), ("Z", "X"): ("H",)}
compilation_dict3_nosign = {("X", "Z"): (), ("Z", "X"): ("S", "SQRT_X", "S")}
compilation_dict4_nosign = {("X", "Z"): (), ("Y", "Z"): ("S",)}
compilation_dict5_nosign = {("X", "Z"): (), ("Z", "X"): ("X", "Z")}
compilation_dict6_nosign = {("X", "Z"): (), ("Z", "X"): ("H", "X")}
compilation_dict7_nosign = {("X", "Z"): (), ("Z", "X"): ("SQRT_X", "SQRT_X", "H")}
compilation_dict8_nosign = {("X", "Z"): (), ("Z", "Y"): ("S", "SQRT_X")}
compilation_dict9_nosign = {("X", "Z"): (), ("Z", "X"): ("H", "Z")}
compilation_dict10_nosign = {("X", "Z"): (), ("Y", "X"): ("SQRT_Y_DAG", "SQRT_X")}
compilation_dict11_nosign = {
    ("X", "Z"): (),
    ("Y", "Z"): ("S",),
    ("Z", "X"): ("H",),
    ("X", "Y"): ("SQRT_X",),
}
compilation_dict12_nosign = {
    ("X", "Z"): (),
    ("Y", "Z"): ("S",),
    ("Z", "X"): ("H",),
    ("X", "Y"): ("SQRT_X",),
    ("Z", "Y"): ("H", "S"),
    ("Y", "X"): ("S", "H"),
}


class TestGetCompilationDict:
    def test_get_compilation_dict_returns_only_identity_dict_if_native_gate_set_empty(
        self,
    ):
        assert _get_compilation_dict(NativeGateSet(one_qubit_gates={}))[0] == {
            ("+X", "+Z"): ()
        }

    def test_get_compilation_dict_returns_only_identity_equivalent_tableau_dict_if_native_gate_set_empty(
        self,
    ):
        assert _get_compilation_dict(NativeGateSet(one_qubit_gates={}))[1] == {}

    def test_get_compilation_dict_returns_only_identity_dict_if_weight_0(self):
        assert _get_compilation_dict(NativeGateSet(one_qubit_gates={X}), max_length=0)[
            0
        ] == {("+X", "+Z"): ()}

    def test_get_compilation_dict_returns_only_identity_equivalent_tableau_dict_if_weight_0(
        self,
    ):
        assert (
            _get_compilation_dict(NativeGateSet(one_qubit_gates={X}), max_length=0)[1]
            == {}
        )

    @pytest.mark.parametrize("native_gate_set", [{}, {X}, {H}, {S}, {SQRT_X}])
    def test_get_compilation_dict_returns_same_output_for_int_max_length_and_None_max_length_in_trivial_cases(
        self, native_gate_set
    ):
        # check that setting max_length=None does not change the return value for cases
        # where we expect the loop to terminate early.
        assert (
            _get_compilation_dict(
                NativeGateSet(one_qubit_gates=native_gate_set), max_length=4
            )[0]
            == _get_compilation_dict(NativeGateSet(one_qubit_gates=native_gate_set))[0]
        )

    @pytest.mark.parametrize("native_gate_set", [{}, {X}, {H}, {S}, {SQRT_X}])
    def test_get_compilation_dict_returns_same_output_for_int_and_None_max_len_with_up_to_Paulis_True(
        self, native_gate_set
    ):
        # check that setting max_length=None does not change the return value for cases
        # where we expect the loop to terminate early.
        assert (
            _get_compilation_dict(
                NativeGateSet(one_qubit_gates=native_gate_set),
                max_length=4,
                up_to_paulis=True,
            )[0]
            == _get_compilation_dict(
                NativeGateSet(one_qubit_gates=native_gate_set), up_to_paulis=True
            )[0]
        )

    @pytest.mark.parametrize(
        "native_gate_set, weight, possible_dicts",
        [
            [
                {},
                1,
                [
                    {
                        ("+X", "+Z"): (),
                    }
                ],
            ],
            [{X}, 1, [{("+X", "+Z"): (), ("+X", "-Z"): ("X",)}]],
            [
                {X, H},
                1,
                [{("+X", "+Z"): (), ("+X", "-Z"): ("X",), ("+Z", "+X"): ("H",)}],
            ],
            [
                {X, SQRT_X},
                2,
                [
                    {
                        ("+X", "-Z"): ("X",),
                        ("+X", "+Z"): (),
                        ("+X", "+Y"): ("SQRT_X", "X"),
                        ("+X", "-Y"): ("SQRT_X",),
                    },
                    {
                        ("+X", "-Z"): ("X",),
                        ("+X", "+Z"): (),
                        ("+X", "+Y"): ("X", "SQRT_X"),
                        ("+X", "-Y"): ("SQRT_X",),
                    },
                ],
            ],
            [{X}, 2, [{("+X", "-Z"): ("X",), ("+X", "+Z"): ()}]],
            [{X, X}, 2, [{("+X", "-Z"): ("X",), ("+X", "+Z"): ()}]],
            [
                {H, S},
                2,
                [
                    {
                        ("+Z", "+X"): ("H",),
                        ("+X", "+Z"): (),
                        ("-Y", "+X"): ("S", "H"),
                        ("+Y", "+Z"): ("S",),
                        ("-X", "+Z"): ("S", "S"),
                        ("+Z", "+Y"): ("H", "S"),
                    }
                ],
            ],
            [
                {H, S, X},
                2,
                [
                    {
                        ("+X", "+Z"): (),
                        ("+Z", "+X"): ("H",),
                        ("-Y", "+X"): ("S", "H"),
                        ("+Y", "+Z"): ("S",),
                        ("-X", "+Z"): ("S", "S"),
                        ("+Z", "+Y"): ("H", "S"),
                        ("+X", "-Z"): ("X",),
                        ("-Z", "+X"): ("H", "X"),
                        ("-Y", "-Z"): ("S", "X"),
                        ("+Z", "-X"): ("X", "H"),
                        ("+Y", "-Z"): ("X", "S"),
                    },
                ],
            ],
            [{X}, 3, [{("+X", "-Z"): ("X",), ("+X", "+Z"): ()}]],
            [{X}, 4, [{("+X", "-Z"): ("X",), ("+X", "+Z"): ()}]],
            [
                {S, S_DAG},
                2,
                [
                    {
                        ("+Y", "+Z"): ("S",),
                        ("-Y", "+Z"): ("S_DAG",),
                        ("-X", "+Z"): ("S", "S"),
                        ("+X", "+Z"): (),
                    },
                    {
                        ("+Y", "+Z"): ("S",),
                        ("-Y", "+Z"): ("S_DAG",),
                        ("-X", "+Z"): ("S_DAG", "S_DAG"),
                        ("+X", "+Z"): (),
                    },
                    {
                        ("+Y", "+Z"): ("S",),
                        ("-Y", "+Z"): ("S_DAG",),
                        ("-X", "+Z"): ("S_DAG", "S_DAG"),
                        ("+X", "+Z"): (),
                    },
                ],
            ],
            [
                {S, SQRT_X, H},
                2,
                [
                    {
                        ("+Y", "+Z"): ("S",),
                        ("-X", "+Z"): ("S", "S"),
                        ("+Y", "+X"): ("SQRT_X", "S"),
                        ("+X", "-Y"): ("SQRT_X",),
                        ("+X", "-Z"): ("SQRT_X", "SQRT_X"),
                        ("+Z", "-Y"): ("S", "SQRT_X"),
                        ("-Y", "+X"): ("H", "SQRT_X"),
                        ("+Z", "+X"): ("H",),
                        ("+X", "+Z"): (),
                        ("+Z", "+Y"): ("SQRT_X", "H"),
                    },
                    {
                        ("+Y", "+Z"): ("S",),
                        ("-X", "+Z"): ("S", "S"),
                        ("+Y", "+X"): ("SQRT_X", "S"),
                        ("+X", "-Y"): ("SQRT_X",),
                        ("+X", "-Z"): ("SQRT_X", "SQRT_X"),
                        ("+Z", "-Y"): ("S", "SQRT_X"),
                        ("+Z", "+X"): ("H",),
                        ("+X", "+Z"): (),
                        ("-Y", "+X"): ("S", "H"),
                        ("+Z", "+Y"): ("SQRT_X", "H"),
                    },
                    {
                        ("+Y", "+Z"): ("S",),
                        ("-X", "+Z"): ("S", "S"),
                        ("+Y", "+X"): ("SQRT_X", "S"),
                        ("+Z", "+Y"): ("H", "S"),
                        ("+X", "-Y"): ("SQRT_X",),
                        ("+X", "-Z"): ("SQRT_X", "SQRT_X"),
                        ("+Z", "-Y"): ("S", "SQRT_X"),
                        ("-Y", "+X"): ("H", "SQRT_X"),
                        ("+Z", "+X"): ("H",),
                        ("+X", "+Z"): (),
                    },
                    {
                        ("+Y", "+Z"): ("S",),
                        ("-X", "+Z"): ("S", "S"),
                        ("+Y", "+X"): ("SQRT_X", "S"),
                        ("+Z", "+Y"): ("H", "S"),
                        ("+X", "-Y"): ("SQRT_X",),
                        ("+X", "-Z"): ("SQRT_X", "SQRT_X"),
                        ("+Z", "-Y"): ("S", "SQRT_X"),
                        ("+Z", "+X"): ("H",),
                        ("+X", "+Z"): (),
                        ("-Y", "+X"): ("S", "H"),
                    },
                ],
            ],
        ],
    )
    def test_get_compilation_dict_returns_correct_values_when_up_to_paulis_is_False(
        self, native_gate_set, weight, possible_dicts
    ):
        # need to specify multiple since the `product` function to enumerate combinations of gates
        # is non-deterministic in its order, and when two sets of gates evaluate to the same tableau,
        # it will pick one at random. e.g, +X,+Y can be SQRT_X*X or X*SQRT_X.
        assert any(
            (
                _get_compilation_dict(
                    NativeGateSet(one_qubit_gates=native_gate_set), max_length=weight
                )[0]
                == expected_dict
                for expected_dict in possible_dicts
            )
        )

    @pytest.mark.parametrize(
        "native_gate_set, weight, possible_dicts",
        [
            [
                {},
                1,
                [
                    {
                        ("X", "Z"): (),
                    }
                ],
            ],
            [{X}, 1, [{("X", "Z"): ()}]],
            [
                {X, H},
                1,
                [{("X", "Z"): (), ("Z", "X"): ("H",)}],
            ],
            [
                {X, SQRT_X},
                2,
                [
                    {
                        ("X", "Z"): (),
                        ("X", "Y"): ("SQRT_X",),
                    },
                ],
            ],
            [{X}, 2, [{("X", "Z"): ()}]],
            [{X, X}, 2, [{("X", "Z"): ()}]],
            [
                {H, S},
                2,
                [
                    {
                        ("Z", "X"): ("H",),
                        ("X", "Z"): (),
                        ("Y", "X"): ("S", "H"),
                        ("Y", "Z"): ("S",),
                        ("Z", "Y"): ("H", "S"),
                    }
                ],
            ],
            [
                {H, S, X},
                2,
                [
                    {
                        ("Z", "X"): ("H",),
                        ("Y", "X"): ("S", "H"),
                        ("Y", "Z"): ("S",),
                        ("Z", "Y"): ("H", "S"),
                        ("X", "Z"): (),
                    },
                ],
            ],
            [{X}, 3, [{("X", "Z"): ()}]],
            [{X}, 4, [{("X", "Z"): ()}]],
            [
                {S, S_DAG},
                2,
                [
                    {
                        ("Y", "Z"): ("S_DAG",),
                        ("X", "Z"): (),
                    },
                    {
                        ("Y", "Z"): ("S",),
                        ("X", "Z"): (),
                    },
                ],
            ],
            [
                {S, SQRT_X, H},
                2,
                [
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("S", "SQRT_X"),
                        ("Y", "X"): ("SQRT_X", "S"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("S", "SQRT_X"),
                        ("Y", "X"): ("H", "SQRT_X"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("S", "SQRT_X"),
                        ("Y", "X"): ("S", "H"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("SQRT_X", "H"),
                        ("Y", "X"): ("SQRT_X", "S"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("SQRT_X", "H"),
                        ("Y", "X"): ("H", "SQRT_X"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("SQRT_X", "H"),
                        ("Y", "X"): ("S", "H"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("H", "S"),
                        ("Y", "X"): ("SQRT_X", "S"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("H", "S"),
                        ("Y", "X"): ("H", "SQRT_X"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("H", "S"),
                        ("Y", "X"): ("S", "H"),
                    },
                ],
            ],
        ],
    )
    def test_get_compilation_dict_returns_correct_values_when_up_to_paulis_is_True(
        self, native_gate_set, weight, possible_dicts
    ):
        # need to specify multiple since the `product` function to enumerate combinations of gates
        # is non-deterministic in its order, and when two sets of gates evaluate to the same tableau,
        # it will pick one at random. e.g, X,Y can be SQRT_X*X or X*SQRT_X.
        assert any(
            (
                _get_compilation_dict(
                    NativeGateSet(one_qubit_gates=native_gate_set),
                    max_length=weight,
                    up_to_paulis=True,
                )[0]
                == expected_dict
                for expected_dict in possible_dicts
            )
        )

    @pytest.mark.parametrize(
        "native_gate_set, possible_dicts",
        [
            [
                {},
                [
                    {
                        ("X", "Z"): (),
                    }
                ],
            ],
            [{X}, [{("X", "Z"): ()}]],
            [{X, X}, [{("X", "Z"): ()}]],
            [
                {X, H},
                [{("X", "Z"): (), ("Z", "X"): ("H",)}],
            ],
            [
                {X, SQRT_X},
                [
                    {
                        ("X", "Z"): (),
                        ("X", "Y"): ("SQRT_X",),
                    },
                ],
            ],
            [{X, X}, [{("X", "Z"): ()}]],
            [
                {H, S},
                [
                    {
                        ("Z", "X"): ("H",),
                        ("X", "Z"): (),
                        ("Y", "X"): ("S", "H"),
                        ("Y", "Z"): ("S",),
                        ("Z", "Y"): ("H", "S"),
                        ("X", "Y"): ("H", "S", "H"),
                    },
                    {
                        ("Z", "X"): ("H",),
                        ("X", "Z"): (),
                        ("Y", "X"): ("S", "H"),
                        ("Y", "Z"): ("S",),
                        ("Z", "Y"): ("H", "S"),
                        ("X", "Y"): ("S", "H", "S"),
                    },
                ],
            ],
            [
                {H, S, X},
                [
                    {
                        ("Z", "X"): ("H",),
                        ("X", "Z"): (),
                        ("Y", "X"): ("S", "H"),
                        ("Y", "Z"): ("S",),
                        ("Z", "Y"): ("H", "S"),
                        ("X", "Y"): ("H", "S", "H"),
                    },
                    {
                        ("Z", "X"): ("H",),
                        ("X", "Z"): (),
                        ("Y", "X"): ("S", "H"),
                        ("Y", "Z"): ("S",),
                        ("Z", "Y"): ("H", "S"),
                        ("X", "Y"): ("S", "H", "S"),
                    },
                ],
            ],
            [
                {S, S_DAG},
                [
                    {
                        ("Y", "Z"): ("S_DAG",),
                        ("X", "Z"): (),
                    },
                    {
                        ("Y", "Z"): ("S",),
                        ("X", "Z"): (),
                    },
                ],
            ],
            [
                {S, SQRT_X, H},
                [
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("S", "SQRT_X"),
                        ("Y", "X"): ("SQRT_X", "S"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("S", "SQRT_X"),
                        ("Y", "X"): ("H", "SQRT_X"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("S", "SQRT_X"),
                        ("Y", "X"): ("S", "H"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("SQRT_X", "H"),
                        ("Y", "X"): ("SQRT_X", "S"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("SQRT_X", "H"),
                        ("Y", "X"): ("H", "SQRT_X"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("SQRT_X", "H"),
                        ("Y", "X"): ("S", "H"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("H", "S"),
                        ("Y", "X"): ("SQRT_X", "S"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("H", "S"),
                        ("Y", "X"): ("H", "SQRT_X"),
                    },
                    {
                        ("X", "Z"): (),
                        ("Y", "Z"): ("S",),
                        ("Z", "X"): ("H",),
                        ("X", "Y"): ("SQRT_X",),
                        ("Z", "Y"): ("H", "S"),
                        ("Y", "X"): ("S", "H"),
                    },
                ],
            ],
        ],
    )
    def test_get_compilation_dict_returns_correct_values_when_up_to_paulis_is_True_and_max_weight_None(
        self, native_gate_set, possible_dicts
    ):
        # need to specify multiple since the `product` function to enumerate combinations of gates
        # is non-deterministic in its order, and when two sets of gates evaluate to the same tableau,
        # it will pick one at random. e.g, X,Y can be SQRT_X*X or X*SQRT_X.
        assert any(
            (
                _get_compilation_dict(
                    NativeGateSet(one_qubit_gates=native_gate_set),
                    up_to_paulis=True,
                )[0]
                == expected_dict
                for expected_dict in possible_dicts
            )
        )

    @pytest.mark.parametrize(
        "gate",
        [
            # since there is only one way of expressing a single gate, dict will be empty
            I,
            X,
            Y,
            Z,
            H,
            S,
            S_DAG,
            SQRT_X,
            SQRT_X_DAG,
            SQRT_Y,
            SQRT_Y_DAG,
        ],
    )
    def test_get_compilation_dict_returns_equivalent_tableau_dict_and_equiv_dict_for_single_native_gates_with_only_one_gate_in_values(
        self,
        gate,
    ):
        comp_dict, equiv_dict = _get_compilation_dict(
            NativeGateSet(one_qubit_gates={gate})
        )
        # change format, add identity
        equiv_dict = {
            k: [tuple([g.stim_string for g in equiv]) for equiv in v][0]
            for k, v in equiv_dict.items()
        }
        equiv_dict[("+X", "+Z")] = ()
        assert comp_dict == equiv_dict

    @pytest.mark.parametrize(
        "native_gate_set, possible_dicts",
        [
            [
                # just these small examples because the possibilities blow up quickly
                {H, X},
                [
                    {("-X", "-Z"): [("X", "H", "X", "H")]},
                    {("-X", "-Z"): [("H", "X", "H", "X")]},
                ],
            ],
            [
                {H, Z},
                [
                    {("-X", "-Z"): [("Z", "H", "Z", "H")]},
                    {("-X", "-Z"): [("H", "Z", "H", "Z")]},
                ],
            ],
        ],
    )
    def test_get_compilation_dict_equivalent_tableau_dict_contents_for_realistic_native_gates(
        self, native_gate_set, possible_dicts
    ):
        equiv_dict = _get_compilation_dict(
            NativeGateSet(one_qubit_gates=native_gate_set)
        )[1]
        assert any(
            (
                (equiv_dict[k] == v for k, v in possible_dict)
                for possible_dict in possible_dicts
            )
        )

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize(
        "native_gate_set",
        [
            {S, SQRT_X},
            {H, Z, X},
            {S, H, X},
            {S_DAG, H, X},
            {S, H, SQRT_X},
            {S_DAG, H, SQRT_X},
            {S_DAG, H, SQRT_X_DAG},
            {S, H, SQRT_X_DAG},
            {SQRT_Y, H, S},
            {SQRT_X, H, S_DAG},
        ],
    )
    def test_get_compilation_dict_returns_valid_equivalent_tableau_dict_for_realistic_native_gates(
        self, native_gate_set, up_to_paulis
    ):
        comp_dict, equiv_tab_dict = _get_compilation_dict(
            NativeGateSet(one_qubit_gates=native_gate_set), up_to_paulis=up_to_paulis
        )
        # test expected parameters, since the number of possible equiv_tab_dicts is huge
        assert len(equiv_tab_dict.keys()) <= len(comp_dict.keys())
        assert all(key in comp_dict for key in equiv_tab_dict)
        assert all(len(val) >= 1 for val in equiv_tab_dict.values())

    @pytest.mark.parametrize(
        "native_gate_set, possible_dicts",
        [
            [
                {H, X},
                [
                    # these are empty because up to Pauli there is a unique expression
                    {},
                ],
            ],
            [
                {H, Z},
                [
                    {},
                ],
            ],
            [
                {H, Z, X},
                [{}],
            ],
            [
                {S, SQRT_X},
                [
                    {("Z", "X"): [("S", "SQRT_X", "S")]},
                    {("Z", "X"): [("SQRT_X", "S", "SQRT_X")]},
                ],
            ],
            [
                {S, H, X},
                [{("X", "Y"): [("S", "H", "S")]}, {("X", "Y"): [("H", "S", "H")]}],
            ],
            [
                {S_DAG, H, X},
                [
                    {("X", "Y"): [("H", "S_DAG", "H")]},
                    {("X", "Y"): [("S_DAG", "H", "S_DAG")]},
                ],
            ],
        ],
    )
    def test_get_compilation_dict_equivalent_tableau_dict_contents_for_realistic_native_gates_up_to_paulis_True(
        self, native_gate_set, possible_dicts
    ):
        equiv_dict = _get_compilation_dict(
            NativeGateSet(one_qubit_gates=native_gate_set), up_to_paulis=True
        )[1]
        assert any(
            (
                (equiv_dict[k] == v for k, v in possible_dict)
                for possible_dict in possible_dicts
            )
        )


class TestGetCompilationWithProjectors:
    def test_get_compilation_with_projectors_returns_identity_if_unitary_block_empty(
        self,
    ):
        assert (
            _get_compilation_with_projectors_before_unitaries(
                {("+X", "+Z"): ()}, [], MZ
            )
            == ()
        )

    @pytest.mark.parametrize("projector_before_unitaries", [MZ, RZ])
    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0, [I], ()],
            [compilation_dict1, [I], ()],
            [compilation_dict1, [Z], ()],
            [compilation_dict1, [Z, Z, Z], ()],
            [compilation_dict2, [X], ("X",)],
            [compilation_dict3, [X, Z], ("X",)],
            [compilation_dict4, [X, H], ("X", "H")],
            [compilation_dict5, [H, X], ("H")],
            [compilation_dict6, [X, H], ("X", "H")],
            [compilation_dict7, [H, X], ("H",)],
            [compilation_dict8, [H, X], ("SQRT_Y",)],
            [compilation_dict7, [S, SQRT_X, S], ("H",)],
            [compilation_dict9, [X, H, Z], ("SQRT_X", "S")],
            [compilation_dict10, [H, X], ("H", "X")],
        ],
    )
    def test_get_compilation_with_projectors_returns_correct_output_z_basis(
        self,
        compilation_dict,
        unitary_block,
        expected_output,
        projector_before_unitaries,
    ):
        assert (
            _get_compilation_with_projectors_before_unitaries(
                compilation_dict, unitary_block, projector_before_unitaries
            )
            == expected_output
        )

    @pytest.mark.parametrize("projector_before_unitaries", [MX, RX])
    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0, [I], ()],
            [compilation_dict1, [I], ()],
            [compilation_dict1, [Z], ("Z",)],
            [compilation_dict1, [Z, Z, Z], ("Z",)],
            [compilation_dict1, [X], ()],
            [compilation_dict16, [X, Z], ("X", "Z")],
            [compilation_dict11, [X, Z], ("Z",)],
            [compilation_dict12, [X, H], ("X", "H")],
            [compilation_dict13, [X, H], ("H",)],
            [compilation_dict13, [H, S], ("H",)],
            [compilation_dict14, [X, H, X], ("H", "X")],
            [compilation_dict13, [S, SQRT_X, S], ("H",)],
            [compilation_dict12, [X, H, Z], ("X", "H", "Z")],
            [compilation_dict13, [X, H, Z], ("H",)],
            [compilation_dict10, [X, H], ("X", "H")],
        ],
    )
    def test_get_compilation_with_projectors_returns_correct_output_x_basis(
        self,
        compilation_dict,
        unitary_block,
        expected_output,
        projector_before_unitaries,
    ):
        assert (
            _get_compilation_with_projectors_before_unitaries(
                compilation_dict, unitary_block, projector_before_unitaries
            )
            == expected_output
        )

    @pytest.mark.parametrize("projector_before_unitaries", [MY, RY])
    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0, [I], ()],
            [compilation_dict1, [I], ()],
            [compilation_dict1, [Z], ("Z",)],
            [compilation_dict1, [Z, Z, Z], ("Z",)],
            [compilation_dict1, [X], ("X",)],
            [compilation_dict16, [X, Z], ()],
            [compilation_dict12, [H, X], ()],
            [compilation_dict13, [H, X], ()],
            [compilation_dict13, [S, SQRT_X, S], ("H",)],
            [compilation_dict13, [X, H, Z], ("H",)],
            [compilation_dict10, [H, X], ()],
            [compilation_dict15, [S], ("S",)],
            [compilation_dict15, [S, S_DAG], ()],
            [compilation_dict_full, [S, H, S], ("S", "H")],
        ],
    )
    def test_get_compilation_with_projectors_returns_correct_output_y_basis(
        self,
        compilation_dict,
        unitary_block,
        expected_output,
        projector_before_unitaries,
    ):
        comp = _get_compilation_with_projectors_before_unitaries(
            compilation_dict, unitary_block, projector_before_unitaries
        )
        assert comp == expected_output

    @pytest.mark.parametrize("projector_before_unitaries", [MX, RX])
    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0_nosign, [I], ()],
            [compilation_dict0_nosign, [Z], ()],
            [compilation_dict0_nosign, [Z, Z, Z], ()],
            [compilation_dict0_nosign, [X], ()],
            [compilation_dict0_nosign, [SQRT_X], ()],
            [compilation_dict1_nosign, [H, X], ("H", "X")],
            [compilation_dict1_nosign, [SQRT_X], ()],
            [compilation_dict2_nosign, [H, X], ("H",)],
            [compilation_dict2_nosign, [S, SQRT_X, S_DAG], ("H",)],
            [compilation_dict2_nosign, [S, SQRT_X], ("H",)],
            [compilation_dict2_nosign, [SQRT_X], ()],
            [compilation_dict3_nosign, [X, H, Z], ("X", "H", "Z")],
            [compilation_dict3_nosign, [SQRT_X, H, S, Z], ("S", "SQRT_X", "S")],
            [compilation_dict3_nosign, [SQRT_X], ()],
            [compilation_dict2_nosign, [H, Z, H, X, H], ("H",)],
        ],
    )
    def test_get_compilation_with_projectors_returns_correct_output_x_basis_with_up_to_paulis_True(
        self,
        compilation_dict,
        unitary_block,
        expected_output,
        projector_before_unitaries,
    ):
        assert (
            _get_compilation_with_projectors_before_unitaries(
                compilation_dict, unitary_block, projector_before_unitaries, True
            )
            == expected_output
        )

    @pytest.mark.parametrize("projector_before_unitaries", [MZ, RZ])
    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0_nosign, [I], ()],
            [compilation_dict0_nosign, [S], ()],
            [compilation_dict1_nosign, [S], ()],
            [compilation_dict2_nosign, [S], ()],
            [compilation_dict2_nosign, [S, SQRT_Y], ("H",)],
            [compilation_dict3_nosign, [S], ()],
            [compilation_dict4_nosign, [S], ()],
            [compilation_dict5_nosign, [X, Z], ()],
            [compilation_dict5_nosign, [S], ()],
            [compilation_dict6_nosign, [H, X], ("H", "X")],
            [compilation_dict6_nosign, [S], ()],
            [compilation_dict7_nosign, [H, X], ("H", "X")],
            [compilation_dict7_nosign, [H, SQRT_X, SQRT_X], ("H", "SQRT_X", "SQRT_X")],
            [compilation_dict7_nosign, [S], ()],
            [compilation_dict8_nosign, [S, SQRT_X], ("S", "SQRT_X")],
            [compilation_dict8_nosign, [S, SQRT_X, SQRT_Y], ("S", "SQRT_X")],
            [compilation_dict8_nosign, [S], ()],
            [compilation_dict2_nosign, [S, SQRT_X, S], ("H",)],
            [compilation_dict2_nosign, [X, H, Z], ("H",)],
            [compilation_dict4_nosign, [S], ()],
        ],
    )
    def test_get_compilation_with_projectors_returns_correct_output_z_basis_with_up_to_paulis_True(
        self,
        compilation_dict,
        unitary_block,
        expected_output,
        projector_before_unitaries,
    ):
        assert (
            _get_compilation_with_projectors_before_unitaries(
                compilation_dict,
                unitary_block,
                projector_before_unitaries,
                up_to_paulis=True,
            )
            == expected_output
        )

    @pytest.mark.parametrize("projector_before_unitaries", [MY, RY])
    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0_nosign, [I], ()],
            [compilation_dict0_nosign, [X], ()],
            [compilation_dict0_nosign, [Z], ()],
            [compilation_dict0_nosign, [SQRT_Y], ()],
            [compilation_dict0_nosign, [S, S_DAG], ()],
            [compilation_dict1_nosign, [SQRT_Y], ()],
            [compilation_dict2_nosign, [SQRT_Y], ()],
            [compilation_dict3_nosign, [SQRT_Y], ()],
            [compilation_dict4_nosign, [SQRT_Y], ()],
            [compilation_dict4_nosign, [S, SQRT_X], ("S",)],
            [compilation_dict5_nosign, [SQRT_Y], ()],
            [compilation_dict6_nosign, [SQRT_Y], ()],
            [compilation_dict7_nosign, [SQRT_Y], ()],
            [compilation_dict8_nosign, [SQRT_Y], ()],
        ],
    )
    def test_get_compilation_with_projectors_returns_correct_output_y_basis_with_up_to_paulis_True(
        self,
        compilation_dict,
        unitary_block,
        expected_output,
        projector_before_unitaries,
    ):
        assert (
            _get_compilation_with_projectors_before_unitaries(
                compilation_dict,
                unitary_block,
                projector_before_unitaries,
                up_to_paulis=True,
            )
            == expected_output
        )

    @pytest.mark.parametrize("projector_before_unitaries", [MZ, RZ])
    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, up_to_paulis",
        [
            [{}, [I], False],
            [compilation_dict0, [X], False],
            [compilation_dict0, [X, Z], False],
            [compilation_dict0, [SQRT_X, SQRT_X], False],
            [{}, [I], True],
            [compilation_dict0, [H], True],
            [compilation_dict0, [H, X], True],
        ],
    )
    def test_get_compilation_with_projectors_throws_KeyError_if_unitary_block_tableau_not_in_compilation_dict(
        self, compilation_dict, unitary_block, projector_before_unitaries, up_to_paulis
    ):
        with pytest.raises(
            KeyError,
            match=(
                "unitary_block's tableau is not in the compilation_dictionary."
                " This means the output of this function may include gates not"
                " in the native gate set. Try compiling unitary_block to the"
                " native gate set first."
            ),
        ):
            _get_compilation_with_projectors_before_unitaries(
                compilation_dict,
                unitary_block,
                projector_before_unitaries,
                up_to_paulis,
            )

    @pytest.mark.parametrize("non_projector", [X, Y, Z, CX])
    def test_get_compilation_with_projectors_throws_NotImplementedError_if_unrecognised_projector_used(
        self, non_projector
    ):
        with pytest.raises(
            NotImplementedError,
            match=f"{non_projector.stim_string} is not a recognised projector",
        ):
            _get_compilation_with_projectors_before_unitaries({}, [], non_projector)

    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0, [I], ()],
            [compilation_dict2, [X], ("X",)],
            [compilation_dict3, [X], ("X",)],
            [compilation_dict1, [Z], ()],
            [compilation_dict3, [Z], ()],
            [compilation_dict1, [S], ()],
            [compilation_dict3, [S], ()],
            [compilation_dict3, [S, S_DAG, S], ()],
            [compilation_dict3, [S_DAG, S, S_DAG], ()],
            [compilation_dict9, [SQRT_X, S], ("SQRT_X", "S")],
            [compilation_dict9, [S, SQRT_X, S, S_DAG], ("S", "SQRT_X")],
            [compilation_dict2, [SQRT_X, SQRT_X], ("SQRT_X", "SQRT_X")],
            [compilation_dict4, [X], ("X",)],
            [compilation_dict4, [SQRT_X, SQRT_X], ("X",)],
            [compilation_dict14, [H, X], ("H", "X")],
            [compilation_dict14, [H, SQRT_X, SQRT_X], ("H", "X")],
            [compilation_dict14, [H], ("H",)],
            [compilation_dict19, [S_DAG, X, H, Z, S], ("SQRT_Y_DAG", "SQRT_X")],
            [
                compilation_dict20,
                [SQRT_Y_DAG, S_DAG, SQRT_X, S, SQRT_Y],
                ("SQRT_Y", "S"),
            ],
            [compilation_dict13, [Z, H, X, H, S, SQRT_X], ("H",)],
            [compilation_dict_full, [S, H, S], ("S", "H")],
        ],
    )
    def test_get_compilation_with_measurement_after_unitaries_gives_correct_output_z_basis(
        self, compilation_dict, unitary_block, expected_output
    ):
        assert (
            _get_compilation_with_measurement_after_unitaries(
                compilation_dict, unitary_block, MZ
            )
            == expected_output
        )

    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0, [I], ()],
            [compilation_dict2, [X], ()],
            [compilation_dict3, [X], ()],
            [compilation_dict1, [Z], ("Z",)],
            [compilation_dict15, [S], ("S",)],
            [compilation_dict0, [SQRT_X, SQRT_X], ()],
            [compilation_dict1, [SQRT_X, SQRT_X], ()],
            [compilation_dict2, [SQRT_X, SQRT_X], ()],
            [compilation_dict3, [SQRT_X, SQRT_X], ()],
            [compilation_dict1, [S, S], ("Z",)],
            [compilation_dict0, [X, SQRT_X], ()],
            [compilation_dict0, [S_DAG, S, S, S_DAG], ()],
            [compilation_dict17, [H], ("H",)],
            [compilation_dict17, [H, X], ("H",)],
            [compilation_dict17, [H, Z], ("H", "Z")],
            [compilation_dict17, [H, H], ()],
            [compilation_dict16, [X, Z], ("X", "Z")],
            [compilation_dict1, [S_DAG, X, H, Z, S], ("Z",)],
            [
                compilation_dict18,
                [SQRT_Y_DAG, S_DAG, SQRT_X, S, SQRT_Y],
                ("SQRT_Y",),
            ],
            [compilation_dict7, [Z, H, X, H, S, SQRT_X], ("S", "SQRT_X")],
        ],
    )
    def test_get_compilation_with_measurement_after_unitaries_gives_correct_output_x_basis(
        self, compilation_dict, unitary_block, expected_output
    ):
        assert (
            _get_compilation_with_measurement_after_unitaries(
                compilation_dict, unitary_block, MX
            )
            == expected_output
        )

    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0, [I], ()],
            [compilation_dict2, [Y], ()],
            [compilation_dict3, [Y], ()],
            [compilation_dict1, [Z], ("Z",)],
            [compilation_dict3, [X], ("X",)],
            [compilation_dict15, [S], ("S",)],
            [compilation_dict13, [H], ("H",)],
            [compilation_dict0, [X, Z], ()],
            [compilation_dict9, [SQRT_X], ("SQRT_X",)],
            [compilation_dict9, [S, SQRT_X], ("S", "SQRT_X")],
            [compilation_dict8, [SQRT_Y], ()],
            [compilation_dict18, [X, SQRT_Y], ("X", "SQRT_Y")],
            [compilation_dict20, [S_DAG, X, H, Z, S], ("SQRT_Y", "S")],
            [
                compilation_dict0,
                [SQRT_Y_DAG, S_DAG, SQRT_X, S, SQRT_Y],
                (),
            ],
            [compilation_dict9, [Z, H, X, H, S, SQRT_X], ("S", "SQRT_X")],
        ],
    )
    def test_get_compilation_with_measurement_after_unitaries_gives_correct_output_y_basis(
        self, compilation_dict, unitary_block, expected_output
    ):
        assert (
            _get_compilation_with_measurement_after_unitaries(
                compilation_dict, unitary_block, MY
            )
            == expected_output
        )

    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0_nosign, [I], ()],
            [compilation_dict0_nosign, [X], ()],
            [compilation_dict0_nosign, [Z], ()],
            [compilation_dict2_nosign, [H], ("H",)],
            [compilation_dict2_nosign, [H, H], ()],
            [compilation_dict0_nosign, [S], ()],
            [compilation_dict0_nosign, [S_DAG], ()],
            [compilation_dict4_nosign, [S], ()],
            [compilation_dict4_nosign, [S_DAG], ()],
            [compilation_dict9_nosign, [H, X], ("H", "X")],
            [compilation_dict9_nosign, [H, SQRT_X, SQRT_X], ("H", "Z")],
            [compilation_dict10_nosign, [S_DAG, X, H, Z, S], ("SQRT_Y_DAG", "SQRT_X")],
            [
                compilation_dict2_nosign,
                [SQRT_Y_DAG, S_DAG, SQRT_X, S, SQRT_Y],
                ("H",),
            ],
            [compilation_dict2_nosign, [Z, H, X, H, S, SQRT_X], ("H",)],
        ],
    )
    def test_get_compilation_with_measurement_after_unitaries_gives_correct_output_z_basis_with_up_to_paulis_True(
        self, compilation_dict, unitary_block, expected_output
    ):
        assert (
            _get_compilation_with_measurement_after_unitaries(
                compilation_dict, unitary_block, MZ, True
            )
            == expected_output
        )

    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0_nosign, [I], ()],
            [compilation_dict0_nosign, [X], ()],
            [compilation_dict0_nosign, [Z], ()],
            [compilation_dict2_nosign, [H], ("H",)],
            [compilation_dict2_nosign, [H, H], ()],
            [compilation_dict4_nosign, [S], ("S",)],
            [compilation_dict4_nosign, [S_DAG], ("S_DAG",)],
            [compilation_dict9_nosign, [H, X], ("H", "X")],
            [compilation_dict9_nosign, [H, SQRT_X, SQRT_X], ("H", "Z")],
            [compilation_dict3_nosign, [S, SQRT_X, S], ("S", "SQRT_X", "S")],
            [compilation_dict4_nosign, [X, S_DAG], ("S",)],
            [compilation_dict0_nosign, [S_DAG, X, H, Z, S], ()],
            [
                compilation_dict2_nosign,
                [SQRT_Y_DAG, S_DAG, SQRT_X, S, SQRT_Y],
                ("H",),
            ],
            [compilation_dict4_nosign, [Z, H, X, H, S, SQRT_X], ("S",)],
        ],
    )
    def test_get_compilation_with_measurement_after_unitaries_gives_correct_output_x_basis_with_up_to_paulis_True(
        self, compilation_dict, unitary_block, expected_output
    ):
        assert (
            _get_compilation_with_measurement_after_unitaries(
                compilation_dict, unitary_block, MX, True
            )
            == expected_output
        )

    @pytest.mark.parametrize(
        "compilation_dict, unitary_block, expected_output",
        [
            [compilation_dict0_nosign, [I], ()],
            [compilation_dict0_nosign, [X], ()],
            [compilation_dict0_nosign, [Z], ()],
            [compilation_dict0_nosign, [Y], ()],
            [compilation_dict2_nosign, [H], ()],
            [compilation_dict2_nosign, [H, H], ()],
            [compilation_dict4_nosign, [S], ("S",)],
            [compilation_dict4_nosign, [S_DAG], ("S_DAG",)],
            [compilation_dict9_nosign, [H, X], ()],
            [compilation_dict9_nosign, [H, SQRT_X, SQRT_X], ()],
            [compilation_dict3_nosign, [S, SQRT_X, S], ()],
            [compilation_dict4_nosign, [X, S_DAG], ("S",)],
            [compilation_dict8_nosign, [S_DAG, X, H, Z, S], ("S", "SQRT_X")],
            [
                compilation_dict2_nosign,
                [SQRT_Y_DAG, S_DAG, SQRT_X, S, SQRT_Y],
                (),
            ],
            [compilation_dict8_nosign, [Z, H, X, H, S, SQRT_X], ("S", "SQRT_X")],
        ],
    )
    def test_get_compilation_with_measurement_after_unitaries_gives_correct_output_y_basis_with_up_to_paulis_True(
        self, compilation_dict, unitary_block, expected_output
    ):
        assert (
            _get_compilation_with_measurement_after_unitaries(
                compilation_dict, unitary_block, MY, True
            )
            == expected_output
        )


class TestGetCompilationWithTwoQubitGates:
    def test__get_compilation_with_two_qubit_gates_valid_for_trivial_case(self):
        assert _get_compilation_with_two_qubit_gates(CX, {}, [], [], [], []) == (
            [],
            [],
            [],
            [],
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [CX, compilation_dict21, [[Z], [], [], []], ([], [Z], [], [])],
            [CX, compilation_dict21, [[], [], [X], []], ([], [], [], [X])],
            [CX, compilation_dict21, [[S], [], [], []], ([], [S], [], [])],
            [CZ, compilation_dict21, [[Z], [], [], []], ([], [Z], [], [])],
            [CZ, compilation_dict21, [[], [], [Z], []], ([], [], [], [Z])],
            [CZ, compilation_dict21, [[Z], [], [Z], []], ([], [Z], [], [Z])],
            [CZ, compilation_dict21, [[S], [], [], []], ([], [S], [], [])],
            [CZ, compilation_dict21, [[], [], [S], []], ([], [], [], [S])],
            [ISWAP, compilation_dict21, [[Z], [], [], []], ([], [], [], [Z])],
            [ISWAP, compilation_dict21, [[], [], [Z], []], ([], [Z], [], [])],
            [ISWAP, compilation_dict21, [[Z], [], [Z], []], ([], [Z], [], [Z])],
            [ISWAP, compilation_dict21, [[S], [], [], []], ([], [], [], [S])],
            [ISWAP, compilation_dict21, [[], [], [S], []], ([], [S], [], [])],
            [ISWAP, compilation_dict21, [[S_DAG], [], [], []], ([], [], [], [S_DAG])],
            [ISWAP, compilation_dict21, [[], [], [S_DAG], []], ([], [S_DAG], [], [])],
            [SWAP, compilation_dict21, [[Z], [], [], []], ([], [], [], [Z])],
            [SWAP, compilation_dict21, [[X], [], [], []], ([], [], [], [X])],
            [SWAP, compilation_dict21, [[Y], [], [], []], ([], [], [], [Y])],
            [SWAP, compilation_dict21, [[], [], [Z], []], ([], [Z], [], [])],
            [SWAP, compilation_dict21, [[], [], [X], []], ([], [X], [], [])],
            [SWAP, compilation_dict21, [[], [], [Y], []], ([], [Y], [], [])],
            [SWAP, compilation_dict21, [[S], [], [], []], ([], [], [], [S])],
            [SWAP, compilation_dict21, [[], [], [S], []], ([], [S], [], [])],
            [SWAP, compilation_dict21, [[SQRT_X], [], [], []], ([], [], [], [SQRT_X])],
            [SWAP, compilation_dict21, [[], [], [SQRT_X], []], ([], [SQRT_X], [], [])],
            [SQRT_XX, compilation_dict21, [[X], [], [], []], ([], [X], [], [])],
            [SQRT_XX, compilation_dict21, [[], [], [X], []], ([], [], [], [X])],
            [
                SQRT_XX,
                compilation_dict21,
                [[SQRT_X], [], [], []],
                ([], [SQRT_X], [], []),
            ],
            [
                SQRT_XX,
                compilation_dict21,
                [[], [], [SQRT_X], []],
                ([], [], [], [SQRT_X]),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_correct_for_pulling_through_case(
        self, two_qubit_gate, comp_dict, unitaries_before, unitaries_after
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate, comp_dict, *unitaries_before
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [CX, compilation_dict22, [[S, S], [], [], []], ([], [Z], [], [])],
            [CX, compilation_dict21, [[S, S, S], [], [], []], ([], [S_DAG], [], [])],
            [CX, compilation_dict21, [[S, S, S, S], [], [], []], ([], [], [], [])],
            [
                CX,
                compilation_dict21,
                [[S, S_DAG, S_DAG, S], [], [], []],
                ([], [], [], []),
            ],
            [
                CX,
                compilation_dict21,
                [[S_DAG, S, S, S_DAG], [], [], []],
                ([], [], [], []),
            ],
            [
                CX,
                compilation_dict21,
                [[S, SQRT_X, S], [], [], []],
                ([S, SQRT_X], [S], [], []),
            ],
            [CX, compilation_dict22, [[Z, Z], [], [], []], ([], [], [], [])],
            [CX, compilation_dict22, [[Z, Z, Z], [], [], []], ([], [Z], [], [])],
            [CZ, compilation_dict22, [[Z, Z], [], [], []], ([], [], [], [])],
            [CZ, compilation_dict22, [[Z, Z, Z], [], [], []], ([], [Z], [], [])],
            [CX, compilation_dict22, [[S, S], [], [], []], ([], [Z], [], [])],
            [CZ, compilation_dict22, [[S, S], [], [], []], ([], [Z], [], [])],
            [CX, compilation_dict21, [[S, S, S], [], [], []], ([], [S_DAG], [], [])],
            [CZ, compilation_dict21, [[S, S, S], [], [], []], ([], [S_DAG], [], [])],
            [SWAP, compilation_dict22, [[Z, Z], [], [], []], ([], [], [], [])],
            [SWAP, compilation_dict22, [[Z, Z, Z], [], [], []], ([], [], [], [Z])],
            [SWAP, compilation_dict22, [[], [], [Z, Z], []], ([], [], [], [])],
            [SWAP, compilation_dict22, [[], [], [Z, Z, Z], []], ([], [Z], [], [])],
            [SWAP, compilation_dict21, [[X, Z], [], [], []], ([], [], [], [Y])],
            [SWAP, compilation_dict21, [[Z, X], [], [], []], ([], [], [], [Y])],
            [SWAP, compilation_dict21, [[], [], [X, Z], []], ([], [Y], [], [])],
            [SWAP, compilation_dict21, [[], [], [Z, X], []], ([], [Y], [], [])],
            [SWAP, compilation_dict22, [[S, S], [], [], []], ([], [], [], [Z])],
            [SWAP, compilation_dict22, [[S, S], [], [], []], ([], [], [], [Z])],
            [ISWAP, compilation_dict21, [[Z, Z], [], [], []], ([], [], [], [])],
            [ISWAP, compilation_dict21, [[Z, Z, Z], [], [], []], ([], [], [], [Z])],
            [ISWAP, compilation_dict21, [[], [], [Z, Z], []], ([], [], [], [])],
            [ISWAP, compilation_dict21, [[], [], [Z, Z, Z], []], ([], [Z], [], [])],
            [ISWAP, compilation_dict21, [[Z, Z], [], [Z, Z], []], ([], [], [], [])],
            [
                ISWAP,
                compilation_dict21,
                [[Z, Z, Z], [], [Z, Z, Z], []],
                ([], [Z], [], [Z]),
            ],
            [ISWAP, compilation_dict21, [[S, S], [], [], []], ([], [], [], [Z])],
            [ISWAP, compilation_dict21, [[], [], [S, S], []], ([], [Z], [], [])],
            [ISWAP, compilation_dict21, [[S, S, S], [], [], []], ([], [], [], [S_DAG])],
            [ISWAP, compilation_dict21, [[], [], [S, S, S], []], ([], [S_DAG], [], [])],
            [ISWAP, compilation_dict21, [[S_DAG, S], [], [], []], ([], [], [], [])],
            [ISWAP, compilation_dict21, [[], [], [S_DAG, S], []], ([], [], [], [])],
            [ISWAP, compilation_dict21, [[S, S_DAG], [], [], []], ([], [], [], [])],
            [ISWAP, compilation_dict21, [[], [], [S, S_DAG], []], ([], [], [], [])],
            [SQRT_XX, compilation_dict21, [[X, X], [], [], []], ([], [], [], [])],
            [SQRT_XX, compilation_dict21, [[], [], [X, X], []], ([], [], [], [])],
            [
                SQRT_XX,
                compilation_dict21,
                [[SQRT_X, SQRT_X], [], [], []],
                ([], [X], [], []),
            ],
            [
                SQRT_XX,
                compilation_dict21,
                [[], [], [SQRT_X, SQRT_X], []],
                ([], [], [], [X]),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_correct_for_multiple_unitaries_on_left_to_pull_through(
        self, two_qubit_gate, comp_dict, unitaries_before, unitaries_after
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate, comp_dict, *unitaries_before
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [CX, compilation_dict21, [[X], [], [], []], ([], [X], [], [X])],
            [CX, compilation_dict21, [[], [], [Z], []], ([], [Z], [], [Z])],
            [CX, compilation_dict21, [[Y], [], [], []], ([], [Y], [], [X])],
            [CX, compilation_dict21, [[], [], [Y], []], ([], [Z], [], [Y])],
            [CZ, compilation_dict21, [[X], [], [], []], ([], [X], [], [Z])],
            [CZ, compilation_dict21, [[], [], [X], []], ([], [Z], [], [X])],
            [CZ, compilation_dict21, [[Y], [], [], []], ([], [Y], [], [Z])],
            [CZ, compilation_dict21, [[], [], [Y], []], ([], [Z], [], [Y])],
            [ISWAP, compilation_dict21, [[X], [], [], []], ([], [Z], [], [Y])],
            [ISWAP, compilation_dict21, [[], [], [X], []], ([], [Y], [], [Z])],
            [ISWAP, compilation_dict21, [[Y], [], [], []], ([], [Z], [], [X])],
            [ISWAP, compilation_dict21, [[], [], [Y], []], ([], [X], [], [Z])],
            [SQRT_XX, compilation_dict21, [[Z], [], [], []], ([], [Y], [], [X])],
            [SQRT_XX, compilation_dict21, [[], [], [Z], []], ([], [X], [], [Y])],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_correct_for_case_when_pulling_through_creates_extra_term(
        self, two_qubit_gate, comp_dict, unitaries_before, unitaries_after
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate, comp_dict, *unitaries_before
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after, gate_exchange_dict",
        [
            [
                CX,
                compilation_dict21,
                [[X, H], [], [], []],
                ([H], [Z], [], []),
                {("+Z", "-X"): [[H, Z]]},
            ],
            [
                CX,
                compilation_dict21,
                [[], [], [X, H], []],
                ([], [Z], [H], [Z]),
                {("+Z", "-X"): [[H, Z]]},
            ],
            [
                CX,
                compilation_dict21,
                [[Z, H], [], [], []],
                ([H], [X], [], [X]),
                {("-Z", "+X"): [[H, X]]},
            ],
            [
                CX,
                compilation_dict21,
                [[], [], [Z, H], []],
                ([], [], [H], [X]),
                {("-Z", "+X"): [[H, X]]},
            ],
            [
                CX,
                compilation_dict21,
                [[SQRT_X, S, SQRT_X], [], [], []],
                ([S, SQRT_X], [S], [], []),
                {("+Z", "+X"): [[S, SQRT_X, S]]},
            ],
            [
                CX,
                compilation_dict21,
                [[SQRT_X, S, SQRT_X, Z], [], [], []],
                ([S, SQRT_X], [S_DAG], [], []),
                {("+Z", "+X"): [[S, SQRT_X, S]]},
            ],
            [
                CX,
                compilation_dict21,
                [[SQRT_X, S_DAG, SQRT_X], [], [], []],
                ([S_DAG, SQRT_X], [S_DAG], [], []),
                {("-Z", "-X"): [[S_DAG, SQRT_X, S_DAG]]},
            ],
            [
                CZ,
                compilation_dict21,
                [[SQRT_X, S, SQRT_X], [], [], []],
                ([S, SQRT_X], [S], [], []),
                {("+Z", "+X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict21,
                [[], [], [SQRT_X, S, SQRT_X], []],
                ([], [], [S, SQRT_X], [S]),
                {("+Z", "+X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict21,
                [[SQRT_X, S, SQRT_X, Z], [], [], []],
                ([S, SQRT_X], [S_DAG], [], []),
                {("+Z", "+X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict21,
                [[], [], [SQRT_X, S, SQRT_X, Z], []],
                ([], [], [S, SQRT_X], [S_DAG]),
                {("+Z", "+X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict21,
                [[], [], [SQRT_X, S, SQRT_X], [Z]],
                ([], [], [S, SQRT_X], [S_DAG]),
                {("+Z", "+X"): [[S, SQRT_X, S]]},
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_correct_when_unitary_block_must_be_exchanged_for_equivalent(
        self,
        two_qubit_gate,
        comp_dict,
        unitaries_before,
        unitaries_after,
        gate_exchange_dict,
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate, comp_dict, *unitaries_before, gate_exchange_dict
            )
            == unitaries_after
        )

    @pytest.mark.parametrize("up_to_paulis", [False, True])
    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before",
        [
            [CX, compilation_dict21, ([H], [], [], [])],
            [CX, compilation_dict21, ([SQRT_X], [], [], [])],
            [CX, compilation_dict21, ([SQRT_Y], [], [], [])],
            [CX, compilation_dict21, ([SQRT_Y_DAG], [], [], [])],
            [CX, compilation_dict21, ([], [], [SQRT_Y_DAG], [])],
            [CZ, compilation_dict21, ([H], [], [], [])],
            [CZ, compilation_dict21, ([SQRT_X], [], [], [])],
            [CZ, compilation_dict21, ([], [], [SQRT_X], [])],
            [CZ, compilation_dict21, ([SQRT_Y], [], [], [])],
            [CZ, compilation_dict21, ([], [], [SQRT_Y], [])],
            [ISWAP, compilation_dict21, ([H], [], [], [])],
            [ISWAP, compilation_dict21, ([SQRT_Y], [], [], [])],
            [ISWAP, compilation_dict21, ([], [], [SQRT_Y], [])],
            [ISWAP, compilation_dict21, ([SQRT_Y_DAG], [], [], [])],
            [ISWAP, compilation_dict21, ([], [], [SQRT_Y_DAG], [])],
            [ISWAP, compilation_dict21, ([SQRT_X], [], [], [])],
            [ISWAP, compilation_dict21, ([], [], [SQRT_X], [])],
            [ISWAP, compilation_dict21, ([SQRT_X_DAG], [], [], [])],
            [ISWAP, compilation_dict21, ([], [], [SQRT_X_DAG], [])],
            [SQRT_XX, compilation_dict21, ([H], [], [], [])],
            [SQRT_XX, compilation_dict21, ([S_DAG], [], [], [])],
            [SQRT_XX, compilation_dict21, ([S_DAG], [], [S_DAG], [])],
            [SQRT_XX, compilation_dict21, ([S], [], [], [])],
            [SQRT_XX, compilation_dict21, ([], [], [S], [])],
            [SQRT_XX, compilation_dict21, ([SQRT_Y], [], [], [])],
            [SQRT_XX, compilation_dict21, ([], [], [SQRT_Y], [])],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_does_not_compile_if_two_qubit_gate_and_ub_conjugate_not_identity_like(
        self, two_qubit_gate, comp_dict, unitaries_before, up_to_paulis
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate, comp_dict, *unitaries_before, up_to_paulis=up_to_paulis
            )
            == unitaries_before
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [CX, compilation_dict21, [[X], [X], [], []], ([], [], [], [X])],
            [CX, compilation_dict21, [[X], [S, X], [], []], ([], [S_DAG], [], [X])],
            [CX, compilation_dict21, [[X], [S, X], [], [X, S]], ([], [S_DAG], [], [S])],
            [CX, compilation_dict21, [[Z], [Z], [], []], ([], [], [], [])],
            [CX, compilation_dict21, [[Z], [X], [], []], ([], [Y], [], [])],
            [CX, compilation_dict21, [[], [], [X], [Z]], ([], [], [], [Y])],
            [CX, compilation_dict21, [[], [], [Z], [Z]], ([], [Z], [], [])],
            [CX, compilation_dict21, [[], [Z], [Z], [Z]], ([], [], [], [])],
            [CX, compilation_dict21, [[X], [X, Z], [], []], ([], [Z], [], [X])],
            [CX, compilation_dict21, [[Z], [Z, X], [], []], ([], [X], [], [])],
            [CZ, compilation_dict21, [[Z], [Z], [], []], ([], [], [], [])],
            [CZ, compilation_dict21, [[X], [X], [], []], ([], [], [], [Z])],
            [CZ, compilation_dict21, [[X], [S, X], [], []], ([], [S_DAG], [], [Z])],
            [CZ, compilation_dict21, [[], [Z], [Z], []], ([], [Z], [], [Z])],
            [CZ, compilation_dict21, [[Z], [X], [Z], [X]], ([], [Y], [], [Y])],
            [CZ, compilation_dict21, [[Y], [Y], [], []], ([], [], [], [Z])],
            [CZ, compilation_dict21, [[], [Z], [Y], [Y]], ([], [], [], [])],
            [CZ, compilation_dict21, [[Z], [X, Z], [], []], ([], [X], [], [])],
            [ISWAP, compilation_dict21, [[X], [Z], [], []], ([], [], [], [Y])],
            [
                ISWAP,
                compilation_dict21,
                [[X], [Z], [], [Z, S, X]],
                ([], [], [], [S_DAG]),
            ],
            [ISWAP, compilation_dict21, [[], [Z], [X], []], ([], [X], [], [Z])],
            [ISWAP, compilation_dict21, [[Y], [Z], [], [X]], ([], [], [], [])],
            [ISWAP, compilation_dict21, [[], [Z], [Y], [X]], ([], [Y], [], [Y])],
            [ISWAP, compilation_dict21, [[X], [X, Z], [], []], ([], [X], [], [Y])],
            [SQRT_XX, compilation_dict21, [[Z], [X], [], []], ([], [Z], [], [X])],
            [SQRT_XX, compilation_dict21, [[], [X], [Z], []], ([], [], [], [Y])],
            [SQRT_XX, compilation_dict21, [[], [], [Z], [Y]], ([], [X], [], [])],
            [SQRT_XX, compilation_dict21, [[], [Z], [Z], []], ([], [Y], [], [Y])],
            [SQRT_XX, compilation_dict21, [[Z], [], [], []], ([], [Y], [], [X])],
            [SQRT_XX, compilation_dict21, [[Z], [X], [], []], ([], [Z], [], [X])],
            [CX, compilation_dict21, [[X], [X], [X], [X]], ([], [], [], [X])],
            [CX, compilation_dict21, [[Z, X], [X], [Z, X], [X]], ([], [], [], [Y])],
            [
                CX,
                compilation_dict21,
                [[Z, X], [Z, X], [Z, X], [Z, X]],
                ([], [Z], [], [X]),
            ],
            [  # this case can be reduced further if the equivalent tableau dictionary is also provided, such that the code considers X,H = H,Z
                CX,
                compilation_dict21,
                [[X, H, X], [Z, H, X], [X, H, X], [Z, H, X]],
                ([X, H], [H, Z], [X, H], [H]),
            ],
            [
                CX,
                compilation_dict21,
                [[SQRT_X, X, Z, S], [S, X, Z], [S, Z], [Z, Y]],
                ([SQRT_X], [], [S], [Z]),
            ],
            [
                CX,
                compilation_dict21,
                [[Y, Z, X], [Y, Z, X], [Y, Z, X], [Y, Z, X]],
                ([], [], [], []),
            ],
            [
                CZ,
                compilation_dict21,
                [[Y, Z, X], [Y, Z, X], [Y, Z, X], [Y, Z, X]],
                ([], [], [], []),
            ],
            [
                SWAP,
                compilation_dict21,
                [[Y, Z, X], [Y, Z, X], [Y, Z, X], [Y, Z, X]],
                ([], [], [], []),
            ],
            [
                ISWAP,
                compilation_dict21,
                [[Y, Z, X], [Y, Z, X], [Y, Z, X], [Y, Z, X]],
                ([], [], [], []),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_pulls_terms_through_with_terms_after_two_qubit_gate(
        self,
        two_qubit_gate,
        comp_dict,
        unitaries_before,
        unitaries_after,
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate, comp_dict, *unitaries_before
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, missing_tableau",
        [
            [CX, compilation_dict21, [[Z], [SQRT_X], [], []], "('-X', '-Y')"],
            [CX, compilation_dict0, [[Z], [], [], []], "('-X', '+Z')"],
            [CX, compilation_dict1, [[X, Z], [], [], []], "('-X', '-Z')"],
            [CX, compilation_dict21, [[], [], [Z], [SQRT_X]], "('-X', '-Y')"],
            [CX, compilation_dict0, [[], [], [Z], []], "('-X', '+Z')"],
            [CX, compilation_dict1, [[], [], [X, Z], []], "('-X', '-Z')"],
            [CZ, compilation_dict21, [[Z], [SQRT_X], [], []], "('-X', '-Y')"],
            [CZ, compilation_dict0, [[Z], [], [], []], "('-X', '+Z')"],
            [CZ, compilation_dict1, [[X, Z], [], [], []], "('-X', '-Z')"],
            [CZ, compilation_dict21, [[], [], [Z], [SQRT_X]], "('-X', '-Y')"],
            [CZ, compilation_dict0, [[], [], [Z], []], "('-X', '+Z')"],
            [CZ, compilation_dict1, [[], [], [X, Z], []], "('-X', '-Z')"],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_throws_KeyError_if_pulling_term_through_creates_tableau_not_in_comp_dict(
        self, two_qubit_gate, comp_dict, unitaries_before, missing_tableau
    ):
        with pytest.raises(KeyError, match=re.escape(missing_tableau)):
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate, comp_dict, *unitaries_before
            )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [CX, compilation_dict11_nosign, [[Z], [], [], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[Z], [], [], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[Z], [], [], []], ([], [], [], [])],
            [ISWAP, compilation_dict11_nosign, [[Z], [], [], []], ([], [], [], [])],
            [CX, compilation_dict11_nosign, [[], [], [Z], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[], [], [Z], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[], [], [Z], []], ([], [], [], [])],
            [ISWAP, compilation_dict11_nosign, [[], [], [Z], []], ([], [], [], [])],
            [CX, compilation_dict11_nosign, [[], [], [X], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[], [], [X], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[], [], [X], []], ([], [], [], [])],
            [SQRT_XX, compilation_dict11_nosign, [[], [], [X], []], ([], [], [], [])],
            [CX, compilation_dict11_nosign, [[X], [], [], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[X], [], [], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[X], [], [], []], ([], [], [], [])],
            [SQRT_XX, compilation_dict11_nosign, [[X], [], [], []], ([], [], [], [])],
            [CX, compilation_dict11_nosign, [[S], [], [], []], ([], [S], [], [])],
            [CZ, compilation_dict11_nosign, [[S], [], [], []], ([], [S], [], [])],
            [SWAP, compilation_dict11_nosign, [[S], [], [], []], ([], [], [], [S])],
            [ISWAP, compilation_dict11_nosign, [[S], [], [], []], ([], [], [], [S])],
            [CZ, compilation_dict11_nosign, [[], [], [S], []], ([], [], [], [S])],
            [SWAP, compilation_dict11_nosign, [[], [], [S], []], ([], [S], [], [])],
            [ISWAP, compilation_dict11_nosign, [[], [], [S], []], ([], [S], [], [])],
            [CZ, compilation_dict11_nosign, [[S], [], [S], []], ([], [S], [], [S])],
            [SWAP, compilation_dict11_nosign, [[S], [], [S], []], ([], [S], [], [S])],
            [ISWAP, compilation_dict11_nosign, [[S], [], [S], []], ([], [S], [], [S])],
            [CX, compilation_dict11_nosign, [[H], [], [], []], ([H], [], [], [])],
            [CZ, compilation_dict11_nosign, [[H], [], [], []], ([H], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[H], [], [], []], ([], [], [], [H])],
            [SWAP, compilation_dict11_nosign, [[], [], [H], []], ([], [H], [], [])],
            [ISWAP, compilation_dict11_nosign, [[], [], [H], []], ([], [], [H], [])],
            [SQRT_XX, compilation_dict11_nosign, [[], [], [H], []], ([], [], [H], [])],
            [
                CX,
                compilation_dict11_nosign,
                [[], [], [SQRT_X_DAG], []],
                ([], [], [], [SQRT_X]),
            ],
            [CX, compilation_dict11_nosign, [[S_DAG], [], [], []], ([], [S], [], [])],
            [CZ, compilation_dict11_nosign, [[S_DAG], [], [], []], ([], [S], [], [])],
            [CZ, compilation_dict11_nosign, [[], [], [S_DAG], []], ([], [], [], [S])],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_with_up_to_paulis_True_correct_for_pull_through_single_gate_case(
        self,
        two_qubit_gate,
        comp_dict,
        unitaries_before,
        unitaries_after,
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                up_to_paulis=True,
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [CX, compilation_dict11_nosign, [[Z, Z], [], [], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[Z, Z], [], [], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[Z, Z], [], [], []], ([], [], [], [])],
            [CX, compilation_dict11_nosign, [[Z], [Z], [], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[Z], [Z], [], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[Z], [Z], [], []], ([], [], [], [])],
            [CX, compilation_dict11_nosign, [[], [], [X, X], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[], [], [X, X], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[], [], [X, X], []], ([], [], [], [])],
            [
                CX,
                compilation_dict11_nosign,
                [[S_DAG], [], [SQRT_X_DAG], []],
                ([], [S], [], [SQRT_X]),
            ],
            [CX, compilation_dict11_nosign, [[S_DAG], [], [], []], ([], [S], [], [])],
            [
                CX,
                compilation_dict11_nosign,
                [[], [], [SQRT_X_DAG], []],
                ([], [], [], [SQRT_X]),
            ],
            [CZ, compilation_dict11_nosign, [[S_DAG], [], [], []], ([], [S], [], [])],
            [CZ, compilation_dict11_nosign, [[], [], [S_DAG], []], ([], [], [], [S])],
            [
                CZ,
                compilation_dict11_nosign,
                [[S_DAG], [], [S_DAG], []],
                ([], [S], [], [S]),
            ],
            [
                ISWAP,
                compilation_dict11_nosign,
                [[S_DAG], [], [], []],
                ([], [], [], [S]),
            ],
            [
                ISWAP,
                compilation_dict11_nosign,
                [[], [], [S_DAG], []],
                ([], [S], [], []),
            ],
            [
                ISWAP,
                compilation_dict11_nosign,
                [[S_DAG], [], [S_DAG], []],
                ([], [S], [], [S]),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_with_up_to_paulis_True_correct_for_pull_through_multiple_gate_case(
        self,
        two_qubit_gate,
        comp_dict,
        unitaries_before,
        unitaries_after,
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                up_to_paulis=True,
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [CX, compilation_dict11_nosign, [[Z], [Z], [], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[Z], [Z], [], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[Z], [], [], [Z]], ([], [], [], [])],
            [ISWAP, compilation_dict11_nosign, [[Z], [], [], [Z]], ([], [], [], [])],
            [
                SQRT_XX,
                compilation_dict11_nosign,
                [[SQRT_X], [SQRT_X], [], []],
                ([], [], [], []),
            ],
            [CX, compilation_dict11_nosign, [[], [], [X], [X]], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[], [], [X], [X]], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[], [], [X], [X]], ([], [], [], [])],
            [ISWAP, compilation_dict11_nosign, [[], [], [X], [X]], ([], [], [], [])],
            [
                SQRT_XX,
                compilation_dict11_nosign,
                [[], [], [SQRT_X], [SQRT_X]],
                ([], [], [], []),
            ],
            [CX, compilation_dict11_nosign, [[S], [S_DAG], [], []], ([], [], [], [])],
            [CZ, compilation_dict11_nosign, [[S], [S_DAG], [], []], ([], [], [], [])],
            [SWAP, compilation_dict11_nosign, [[S], [], [], [S_DAG]], ([], [], [], [])],
            [
                ISWAP,
                compilation_dict11_nosign,
                [[S], [], [], [S_DAG]],
                ([], [], [], []),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_with_up_to_paulis_True_correct_for_case_with_gates_after_2q_gate(
        self,
        two_qubit_gate,
        comp_dict,
        unitaries_before,
        unitaries_after,
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                up_to_paulis=True,
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after, gate_exchange_dict",
        [
            [
                CX,
                compilation_dict11_nosign,
                [[SQRT_X, S, SQRT_X], [], [], []],
                ([S, SQRT_X], [S], [], []),
                {("Z", "X"): [[S, SQRT_X, S]]},
            ],
            [
                CX,
                compilation_dict11_nosign,
                [[SQRT_X, S, SQRT_X, Z], [], [], []],
                ([S, SQRT_X], [S], [], []),
                {("Z", "X"): [[S, SQRT_X, S]]},
            ],
            [
                CX,
                compilation_dict11_nosign,
                [[SQRT_X, S_DAG, SQRT_X], [], [], []],
                ([S_DAG, SQRT_X], [S], [], []),
                {("Z", "X"): [[S_DAG, SQRT_X, S_DAG]]},
            ],
            [
                CX,
                compilation_dict11_nosign,
                [[SQRT_X, S_DAG, SQRT_X], [], [], []],
                ([S, SQRT_X], [S], [], []),
                {("Z", "X"): [[S, SQRT_X, S_DAG]]},
            ],
            [
                CZ,
                compilation_dict11_nosign,
                [[SQRT_X, S, SQRT_X], [], [], []],
                ([S, SQRT_X], [S], [], []),
                {("Z", "X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict11_nosign,
                [[], [], [SQRT_X, S, SQRT_X], []],
                ([], [], [S, SQRT_X], [S]),
                {("Z", "X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict11_nosign,
                [[SQRT_X, S, SQRT_X, Z], [], [], []],
                ([S, SQRT_X], [S], [], []),
                {("Z", "X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict11_nosign,
                [[], [], [SQRT_X, S, SQRT_X, Z], []],
                ([], [], [S, SQRT_X], [S]),
                {("Z", "X"): [[S, SQRT_X, S]]},
            ],
            [
                CZ,
                compilation_dict11_nosign,
                [[], [], [SQRT_X, S, SQRT_X], [Z]],
                ([], [], [S, SQRT_X], [S]),
                {("Z", "X"): [[S, SQRT_X, S]]},
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_with_up_to_paulis_True_correct_when_unitary_block_must_be_exchanged_for_equivalent(
        self,
        two_qubit_gate,
        comp_dict,
        unitaries_before,
        unitaries_after,
        gate_exchange_dict,
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                gate_exchange_dict,
                up_to_paulis=True,
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before",
        [
            [
                CX,
                compilation_dict21,
                ([Y], [], [], []),
            ],
            [
                CX,
                compilation_dict21,
                ([], [], [Y], []),
            ],
            [
                CZ,
                compilation_dict21,
                ([X], [], [], []),
            ],
            [
                CZ,
                compilation_dict21,
                ([], [], [X], []),
            ],
            [
                ISWAP,
                compilation_dict21,
                ([], [], [X], []),
            ],
            [
                SQRT_XX,
                compilation_dict21,
                ([], [], [Z], []),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_does_not_compile_relevant_cases_if_allow_terms_to_mutate_is_False(
        self, two_qubit_gate, comp_dict, unitaries_before
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                allow_terms_to_mutate=False,
            )
            == unitaries_before
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before",
        [
            [
                CX,
                compilation_dict21,
                ([Y], [], [], []),
            ],
            [
                CX,
                compilation_dict21,
                ([], [], [Y], []),
            ],
            [
                CZ,
                compilation_dict21,
                ([X], [], [], []),
            ],
            [
                CZ,
                compilation_dict21,
                ([], [], [X], []),
            ],
            [
                ISWAP,
                compilation_dict21,
                ([], [], [X], []),
            ],
            [
                SQRT_XX,
                compilation_dict21,
                ([], [], [Z], []),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_does_not_compile_relevant_cases_if_allow_terms_to_multiply_is_False(
        self, two_qubit_gate, comp_dict, unitaries_before
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                allow_terms_to_multiply=False,
            )
            == unitaries_before
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before",
        [
            [
                CX,
                compilation_dict11_nosign,
                ([], [], [S], []),
            ],
            [
                CX,
                compilation_dict11_nosign,
                ([SQRT_X], [], [], []),
            ],
            [
                CZ,
                compilation_dict11_nosign,
                ([SQRT_X], [], [], []),
            ],
            [
                ISWAP,
                compilation_dict11_nosign,
                ([S], [], [], []),
            ],
            [
                ISWAP,
                compilation_dict11_nosign,
                ([], [], [S], []),
            ],
            [
                SQRT_XX,
                compilation_dict11_nosign,
                ([S], [], [], []),
            ],
            [
                SQRT_XX,
                compilation_dict11_nosign,
                ([], [], [S], []),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_does_not_compile_relevant_cases_if_allow_terms_to_multiply_is_False_and_up_to_paulis_True(
        self, two_qubit_gate, comp_dict, unitaries_before
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                allow_terms_to_multiply=False,
                up_to_paulis=True,
            )
            == unitaries_before
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, comp_dict, unitaries_before, unitaries_after",
        [
            [
                CX,
                compilation_dict21,
                ([X], [], [], []),
                ([], [X], [], [X]),
            ],
            [
                CX,
                compilation_dict21,
                ([], [], [Z], []),
                ([], [Z], [], [Z]),
            ],
            [
                CXSWAP,
                compilation_dict21,
                ([X], [], [], []),
                ([], [X], [], [X]),
            ],
            [
                CXSWAP,
                compilation_dict21,
                ([], [], [Z], []),
                ([], [Z], [], [Z]),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_allows_certain_cases_if_allow_terms_to_multiply_True_and_allow_terms_to_mutate_False(
        self, two_qubit_gate, comp_dict, unitaries_before, unitaries_after
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                comp_dict,
                *unitaries_before,
                allow_terms_to_multiply=True,
                allow_terms_to_mutate=False,
            )
            == unitaries_after
        )

    @pytest.mark.parametrize(
        "two_qubit_gate, unitaries_before, unitaries_after",
        [
            [
                SQRT_XX,
                [[H], [S, H, S], [H], [H, S, Z]],
                ([H], [S, H, S], [H], [H, S, Z]),
            ],
            [SQRT_XX, [[], [H, S, Z], [S, H, S], [S, H, S]], ([], [X, H, S], [], [X])],
            [SQRT_XX, [[S, H, S], [S, H, S], [], [H, S, Z]], ([], [X], [], [X, H, S])],
            [SQRT_XX, [[], [S, H, S], [S, H, S], [S, H, S]], ([], [S, H, S], [], [X])],
            [SQRT_XX, [[], [S, H, S], [S, H, S], [S, H, S]], ([], [S, H, S], [], [X])],
            [
                SQRT_XX,
                [[], [S, H, S], [Z, H, S], [H, S, Z]],
                ([], [S, H, S], [Z, H, S], [H, S, Z]),
            ],
            [
                SQRT_XX,
                [[H, S, Z], [S, H, S], [], [S, H, S]],
                ([H, S], [S, X, H, S], [], [H, S, H]),
            ],
            [
                SQRT_XX,
                [[S, H, S], [S, H, S], [H, S, Z], [H, S, Z]],
                ([], [], [H, S], [Z, H, S]),
            ],
            [
                SQRT_XX,
                [[X, H, S], [S, H, S], [X], [H, S, Z]],
                ([H, S], [S, X, H, S], [], [X, H, S]),
            ],
            [
                SQRT_XX,
                [[S, X, H, S], [S, H, S], [H, S, H], [H, S, Z]],
                ([], [Z, X], [], [X, H]),
            ],
            [
                SQRT_XX,
                [[X, H, S], [X, H, S], [H], [S, H, S]],
                ([H, S], [Z, H, S], [H], [H, S, H]),
            ],
            [
                SQRT_XX,
                [[H], [S, H, S], [S, H, S], [X, H, S]],
                ([H], [S, H, S], [], [X, H]),
            ],
            [
                SQRT_XX,
                [[X, H, S], [S, H, S], [S, H, S], [X, H, S]],
                ([H, S], [S, X, H, S], [], [H]),
            ],
            [
                SQRT_XX,
                [[H], [S, H, S], [S, X, H, S], [X, H, S]],
                ([H], [H, S, H], [], [Z, X, H]),
            ],
        ],
    )
    def test__get_compilation_with_two_qubit_gates_for_snippets_of_memory_circuit(
        self, two_qubit_gate, unitaries_before, unitaries_after
    ):
        assert (
            _get_compilation_with_two_qubit_gates(
                two_qubit_gate,
                compilation_dict_full,
                *unitaries_before,
                gate_exchange_dict=gate_exchange_dict_full,
            )
            == unitaries_after
        )


class TestIsIdentityLike:
    @pytest.mark.parametrize(
        "gates",
        [
            ["I"],
            ["I", "I"],
            ["I", "I", "I"],
            ["I", "I", "I", "I"],
        ],
    )
    def test__is_identity_like_returns_True_for_identity_stop(self, gates):
        tableau = reduce(add, [stim.Tableau.from_named_gate(gate) for gate in gates])
        assert _is_identity_like(tableau)

    @pytest.mark.parametrize(
        "gates",
        [
            [CX],
            [ISWAP],
            [SWAP],
            [SQRT_XX],
            [CX, SWAP],
            [CX, ISWAP],
        ],
    )
    def test__is_identity_like_returns_False_for_non_identity_gates(self, gates):
        tableau = reduce(
            add, [stim.Tableau.from_named_gate(gate.stim_string) for gate in gates]
        )
        assert not _is_identity_like(tableau)

    @pytest.mark.parametrize(
        "control_gates, target_gates, two_q_gate",
        [
            [[I], [S], CX],
            [[S], [H], CX],
            [[X], [Z], CX],
            [[I], [S], CZ],
            [[S], [H], CZ],
            [[X], [Z], CZ],
            [[I], [S], ISWAP],
            [[S], [H], ISWAP],
            [[X], [Z], ISWAP],
        ],
    )
    def test__is_identity_like_returns_False_for_single_qubit_gates_followed_by_two_qubit_gate(
        self, control_gates, target_gates, two_q_gate
    ):
        tableau = stim.Tableau.from_named_gate(two_q_gate.stim_string) * reduce(
            add,
            [
                stim.Tableau.from_named_gate(c.stim_string)
                + stim.Tableau.from_named_gate(t.stim_string)
                for c, t in zip(control_gates, target_gates)
            ],
        )
        assert not _is_identity_like(tableau)

    @pytest.mark.parametrize(
        "gate, conjugate",
        [
            [CX, [H, I]],
            [CX, [SQRT_X, I]],
            [CX, [SQRT_Y, I]],
            [CX, [SQRT_Y_DAG, I]],
            [CX, [I, SQRT_Y]],
            [CX, [I, SQRT_Y_DAG]],
            [ISWAP, [SQRT_Y, I]],
            [ISWAP, [SQRT_Y_DAG, I]],
            [ISWAP, [SQRT_X, I]],
            [ISWAP, [SQRT_X_DAG, I]],
            [ISWAP, [I, SQRT_Y]],
            [ISWAP, [I, SQRT_Y_DAG]],
            [ISWAP, [I, SQRT_X]],
            [ISWAP, [I, SQRT_X_DAG]],
            [SQRT_XX, [S_DAG, I]],
            [SQRT_XX, [S, I]],
            [SQRT_XX, [SQRT_Y, I]],
            [SQRT_XX, [I, S_DAG]],
            [SQRT_XX, [I, S]],
            [SQRT_XX, [I, SQRT_Y]],
        ],
    )
    def test__is_identity_like_returns_False_for_non_identity_gate_conjugates(
        self, gate, conjugate
    ):
        tableau = (
            stim.Tableau.from_named_gate(gate.stim_string)
            * (
                reduce(
                    add,
                    [
                        stim.Tableau.from_named_gate(gate.stim_string)
                        for gate in conjugate
                    ],
                )
            )
            * stim.Tableau.from_named_gate(gate.stim_string).inverse()
        )
        assert not _is_identity_like(tableau)

    @pytest.mark.parametrize(
        "gate, conjugate",
        [
            [CX, [I, I]],
            [CX, [X, I]],
            [CX, [I, Z]],
            [CX, [Z, Z]],
            [CX, [S, I]],
            [ISWAP, [I, I]],
            [ISWAP, [I, X]],
            [ISWAP, [Y, X]],
            [ISWAP, [S, I]],
            [SQRT_XX, [SQRT_X, I]],
            [SQRT_XX, [I, SQRT_X]],
            [SQRT_XX, [Z, I]],
            [SQRT_XX, [I, Z]],
        ],
    )
    def test__is_identity_like_returns_True_for_identity_like_non_trivial_gates(
        self, gate, conjugate
    ):
        tableau = (
            stim.Tableau.from_named_gate(gate.stim_string)
            * (
                reduce(
                    add,
                    [
                        stim.Tableau.from_named_gate(gate.stim_string)
                        for gate in conjugate
                    ],
                )
            )
            * stim.Tableau.from_named_gate(gate.stim_string).inverse()
        )
        assert _is_identity_like(tableau)


class TestCompilationData:
    class TestTrivialCase:
        def test_extract_structure_from_circuit_gives_empty_returns_for_trivial_circuit(
            self,
        ):
            assert _extract_structure_from_circuit(Circuit()) == CompilationData(
                {}, {}, {}, {}, {}
            )

        def test_create_circuit_from_compilation_data_returns_empty_Circuit_for_trivial_input(
            self,
        ):
            assert (
                _create_circuit_from_compilation_data(
                    CompilationData({}, {}, {}, {}, {}), {}
                )
                == Circuit()
            )

    @pytest.mark.parametrize(
        "reset_gate",
        [
            RX,
            RY,
            RZ,
        ],
    )
    @pytest.mark.parametrize(
        "circuit, reset_dict, unitary_blocks",
        [
            # single length unitary blocks
            [
                Circuit(GateLayer(RX(0))),
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [], 1: []},
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(RX(0))]),
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0)], 1: []},
            ],
            [
                Circuit([GateLayer(RX(0)), GateLayer(X(0))]),
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [], 1: [X(0)]},
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(RX(0)), GateLayer(Y(0))]),
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0)], 1: [Y(0)]},
            ],
            [
                Circuit([GateLayer(RX(0)), GateLayer(RX(0))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (1, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [], 1: [], 2: []},
            ],
            [
                Circuit([GateLayer(RX(0)), GateLayer(X(0)), GateLayer(RX(0))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (2, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [], 1: [X(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(RX(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0)], 1: [Y(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(RX(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                        GateLayer(Z(0)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0)], 1: [Y(0)], 2: [Z(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(RX(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                        GateLayer(Z(0)),
                        GateLayer(RX(0)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                    (5, Qubit(0)): {"preceding": 2, "succeeding": 3},
                },
                {0: [X(0)], 1: [Y(0)], 2: [Z(0)], 3: []},
            ],
            # multiple length unitary blocks
            [
                Circuit([GateLayer(X(0)), GateLayer(Y(0)), GateLayer(RX(0))]),
                {(2, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0), Y(0)], 1: []},
            ],
            [
                Circuit([GateLayer(RX(0)), GateLayer(X(0)), GateLayer(Y(0))]),
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [], 1: [X(0), Y(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                    ]
                ),
                {(2, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0), Y(0)], 1: [Z(0), S(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(RX(0)),
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [], 1: [X(0), Y(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                        GateLayer(RX(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (5, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0), Y(0)], 1: [Z(0), S(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                        GateLayer(RX(0)),
                        GateLayer(S_DAG(0)),
                        GateLayer(SQRT_X(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (5, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0), Y(0)], 1: [Z(0), S(0)], 2: [S_DAG(0), SQRT_X(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(RX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                        GateLayer(RX(0)),
                        GateLayer(S_DAG(0)),
                        GateLayer(SQRT_X(0)),
                        GateLayer(RX(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (5, Qubit(0)): {"preceding": 1, "succeeding": 2},
                    (8, Qubit(0)): {"preceding": 2, "succeeding": 3},
                },
                {0: [X(0), Y(0)], 1: [Z(0), S(0)], 2: [S_DAG(0), SQRT_X(0)], 3: []},
            ],
        ],
    )
    class TestResetGates:
        def test_extract_structure_from_circuit_gives_correct_output_for_resets(
            self, circuit: Circuit, reset_dict, unitary_blocks, reset_gate
        ):
            circuit.replace_gates({RX: lambda gate: reset_gate(gate.qubit)})
            results = _extract_structure_from_circuit(circuit)
            assert results.unitary_blocks == unitary_blocks
            assert results.reset_gates == {
                (*key, reset_gate.stim_string): value
                for key, value in reset_dict.items()
            }
            circuit.replace_gates({reset_gate: lambda gate: RX(gate.qubit)})

        def test_create_circuit_from_compilation_data_returns_correct_circuit_for_reset_gates(
            self, circuit: Circuit, reset_dict, unitary_blocks, reset_gate
        ):
            circuit.replace_gates({RX: lambda gate: reset_gate(gate.qubit)})
            reset_dict = {
                (*key, reset_gate.stim_string): value
                for key, value in reset_dict.items()
            }
            comp_data = CompilationData(unitary_blocks, reset_dict, {}, {}, {})
            assert (
                _create_circuit_from_compilation_data(
                    comp_data, {i: i for i in range(len(circuit.layers))}
                )
                == circuit
            )
            circuit.replace_gates({reset_gate: lambda gate: RX(gate.qubit)})

    @pytest.mark.parametrize(
        "meas_gate",
        [
            MX,
            MY,
            MZ,
            MRX,
            MRY,
            MRZ,
        ],
    )
    @pytest.mark.parametrize(
        "circuit, meas_dict, unitary_blocks",
        [
            # single length unitary blocks
            [
                Circuit(GateLayer(MX(0))),
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [], 1: []},
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(MX(0))]),
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0)], 1: []},
            ],
            [
                Circuit([GateLayer(MX(0)), GateLayer(X(0))]),
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [], 1: [X(0)]},
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(MX(0)), GateLayer(Y(0))]),
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0)], 1: [Y(0)]},
            ],
            [
                Circuit([GateLayer(MX(0)), GateLayer(MX(0))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (1, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [], 1: [], 2: []},
            ],
            [
                Circuit([GateLayer(MX(0)), GateLayer(X(0)), GateLayer(MX(0))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (2, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [], 1: [X(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(MX(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0)], 1: [Y(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(MX(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                        GateLayer(Z(0)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0)], 1: [Y(0)], 2: [Z(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(MX(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                        GateLayer(Z(0)),
                        GateLayer(MX(0)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                    (5, Qubit(0)): {"preceding": 2, "succeeding": 3},
                },
                {0: [X(0)], 1: [Y(0)], 2: [Z(0)], 3: []},
            ],
            # multiple length unitary blocks
            [
                Circuit([GateLayer(X(0)), GateLayer(Y(0)), GateLayer(MX(0))]),
                {(2, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0), Y(0)], 1: []},
            ],
            [
                Circuit([GateLayer(MX(0)), GateLayer(X(0)), GateLayer(Y(0))]),
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [], 1: [X(0), Y(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                    ]
                ),
                {(2, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {0: [X(0), Y(0)], 1: [Z(0), S(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(MX(0)),
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [], 1: [X(0), Y(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                        GateLayer(MX(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (5, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0), Y(0)], 1: [Z(0), S(0)], 2: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                        GateLayer(MX(0)),
                        GateLayer(SQRT_X(0)),
                        GateLayer(S_DAG(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (5, Qubit(0)): {"preceding": 1, "succeeding": 2},
                },
                {0: [X(0), Y(0)], 1: [Z(0), S(0)], 2: [SQRT_X(0), S_DAG(0)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(MX(0)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                        GateLayer(MX(0)),
                        GateLayer(S_DAG(0)),
                        GateLayer(SQRT_X(0)),
                        GateLayer(MX(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 1},
                    (5, Qubit(0)): {"preceding": 1, "succeeding": 2},
                    (8, Qubit(0)): {"preceding": 2, "succeeding": 3},
                },
                {0: [X(0), Y(0)], 1: [Z(0), S(0)], 2: [S_DAG(0), SQRT_X(0)], 3: []},
            ],
        ],
    )
    class TestMeasurementGates:
        def test_extract_structure_from_circuit_gives_correct_output_for_measurements(
            self,
            circuit: Circuit,
            meas_dict,
            unitary_blocks,
            meas_gate,
        ):
            circuit.replace_gates({MX: lambda gate: meas_gate(gate.qubit)})
            results = _extract_structure_from_circuit(circuit)
            assert results.unitary_blocks == unitary_blocks
            assert results.measurement_gates == {
                (*key, meas_gate.stim_string): value for key, value in meas_dict.items()
            }
            circuit.replace_gates({meas_gate: lambda gate: MX(gate.qubit)})

        def test_create_circuit_from_compilation_data_returns_correct_circuit_for_measurement_gates(
            self, unitary_blocks, meas_dict, circuit: Circuit, meas_gate
        ):
            circuit.replace_gates({MX: lambda gate: meas_gate(gate.qubit)})
            meas_dict = {
                (*key, meas_gate.stim_string): value for key, value in meas_dict.items()
            }
            comp_data = CompilationData(unitary_blocks, {}, meas_dict, {}, {})
            assert (
                _create_circuit_from_compilation_data(
                    comp_data, {i: i for i in range(len(circuit.layers))}
                )
                == circuit
            )
            circuit.replace_gates({meas_gate: lambda gate: MX(gate.qubit)})

    @pytest.mark.parametrize("two_qubit_gate", [CX, CY, CZ, SWAP, ISWAP, SQRT_XX])
    @pytest.mark.parametrize(
        "circuit, two_q_dict, unitary_blocks",
        [
            # single unitary gate around single 2q gate
            [
                Circuit(GateLayer(CX(0, 1))),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [], 3: []},
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(CX(0, 1))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0)], 1: [], 2: [], 3: []},
            ],
            [
                Circuit([GateLayer(X(1)), GateLayer(CX(0, 1))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [X(1)], 2: [], 3: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(0))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [X(0)], 3: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(1))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [], 3: [X(1)]},
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(CX(0, 1)), GateLayer(Y(0))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0)], 1: [], 2: [Y(0)], 3: []},
            ],
            [
                Circuit([GateLayer(X(1)), GateLayer(CX(0, 1)), GateLayer(Y(1))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [X(1)], 2: [], 3: [Y(1)]},
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(CX(0, 1)), GateLayer(Y(1))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0)], 1: [], 2: [], 3: [Y(1)]},
            ],
            [
                Circuit([GateLayer(X(1)), GateLayer(CX(0, 1)), GateLayer(Y(0))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [X(1)], 2: [Y(0)], 3: []},
            ],
            # single unitary gate around multiple 2q gate
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(0)), GateLayer(CX(0, 1))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [], 5: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(1)), GateLayer(CX(0, 1))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
                {0: [], 1: [], 2: [], 3: [X(1)], 4: [], 5: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(0)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [Y(0)], 5: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(1)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [], 5: [Y(1)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(1)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(0)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
                {0: [], 1: [], 2: [], 3: [X(1)], 4: [Y(0)], 5: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(1)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(1)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
                {0: [], 1: [], 2: [], 3: [X(1)], 4: [], 5: [Y(1)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(0)),
                        GateLayer(CX(0, 1)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                    (4, Qubit(0)): {"preceding": 4, "succeeding": 6},
                    (4, Qubit(1)): {"preceding": 5, "succeeding": 7},
                },
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [Y(0)], 5: [], 6: [], 7: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(1)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(0)),
                        GateLayer(CX(0, 1)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                    (4, Qubit(0)): {"preceding": 4, "succeeding": 6},
                    (4, Qubit(1)): {"preceding": 5, "succeeding": 7},
                },
                {0: [], 1: [], 2: [], 3: [X(1)], 4: [Y(0)], 5: [], 6: [], 7: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(1)),
                        GateLayer(CX(0, 1)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                    (4, Qubit(0)): {"preceding": 4, "succeeding": 6},
                    (4, Qubit(1)): {"preceding": 5, "succeeding": 7},
                },
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [], 5: [Y(1)], 6: [], 7: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(CX(0, 1)),
                        GateLayer(X(1)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(1)),
                        GateLayer(CX(0, 1)),
                    ]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                    (4, Qubit(0)): {"preceding": 4, "succeeding": 6},
                    (4, Qubit(1)): {"preceding": 5, "succeeding": 7},
                },
                {0: [], 1: [], 2: [], 3: [X(1)], 4: [], 5: [Y(1)], 6: [], 7: []},
            ],
            # multiple unitaries around 2q gate
            [
                Circuit([GateLayer(X(0)), GateLayer(Y(0)), GateLayer(CX(0, 1))]),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (2, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0), Y(0)], 1: [], 2: [], 3: []},
            ],
            [
                Circuit([GateLayer(X(1)), GateLayer(Y(1)), GateLayer(CX(0, 1))]),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (2, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [X(1), Y(1)], 2: [], 3: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(0)), GateLayer(Y(0))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [X(0), Y(0)], 3: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(1)), GateLayer(Y(1))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [], 3: [X(1), Y(1)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (2, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0), Y(0)], 1: [], 2: [Z(0), S(0)], 3: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(1)),
                        GateLayer(Y(1)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Z(1)),
                        GateLayer(S(1)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (2, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [X(1), Y(1)], 2: [], 3: [Z(1), S(1)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(Y(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Z(1)),
                        GateLayer(S(1)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (2, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0), Y(0)], 1: [], 2: [], 3: [Z(1), S(1)]},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(1)),
                        GateLayer(Y(1)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Z(0)),
                        GateLayer(S(0)),
                    ]
                ),
                {
                    (2, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (2, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [X(1), Y(1)], 2: [Z(0), S(0)], 3: []},
            ],
            # multiple unitaries around 2q gate, mixed qubits
            [
                Circuit([GateLayer([X(0), Y(1)]), GateLayer(CX(0, 1))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0)], 1: [Y(1)], 2: [], 3: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer([X(0), Y(1)])]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [X(0)], 3: [Y(1)]},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer([X(0), Y(1)])]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [X(0)], 3: [Y(1)]},
            ],
            [
                Circuit([GateLayer([X(1), Y(0)]), GateLayer(CX(0, 1))]),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [Y(0)], 1: [X(1)], 2: [], 3: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer([X(1), Y(0)])]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [], 1: [], 2: [Y(0)], 3: [X(1)]},
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([Z(0), S(1)]),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {0: [X(0)], 1: [Y(1)], 2: [Z(0)], 3: [S(1)]},
            ],
            # 2q gates interleaving on one qubit
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(CX(1, 2))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (1, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []},
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(1)), GateLayer(CX(1, 2))]),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (2, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (2, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {0: [], 1: [], 2: [X(1)], 3: [], 4: [], 5: [], 6: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(Y(1)),
                        GateLayer(CX(1, 2)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (3, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {0: [X(0)], 1: [], 2: [Y(1)], 3: [], 4: [], 5: [], 6: []},
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer(Z(1)),
                        GateLayer(CX(1, 2)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (3, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {0: [X(0)], 1: [Y(1)], 2: [Z(1)], 3: [], 4: [], 5: [], 6: []},
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([Z(0), S(1)]),
                        GateLayer(CX(1, 2)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (3, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {0: [X(0)], 1: [Y(1)], 2: [S(1)], 3: [], 4: [Z(0)], 5: [], 6: []},
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([Z(0), S(1)]),
                        GateLayer(CX(1, 2)),
                        GateLayer(SQRT_X(1)),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (3, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {
                    0: [X(0)],
                    1: [Y(1)],
                    2: [S(1)],
                    3: [],
                    4: [Z(0)],
                    5: [SQRT_X(1)],
                    6: [],
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([Z(0), S(1)]),
                        GateLayer(CX(1, 2)),
                        GateLayer([SQRT_X(1), S_DAG(2)]),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (3, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {
                    0: [X(0)],
                    1: [Y(1)],
                    2: [S(1)],
                    3: [],
                    4: [Z(0)],
                    5: [SQRT_X(1)],
                    6: [S_DAG(2)],
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([CX(0, 1), Z(2)]),
                        GateLayer([S(0), S_DAG(1), SQRT_X(2)]),
                        GateLayer([SQRT_X_DAG(0), CX(1, 2)]),
                        GateLayer([SQRT_Y(1), SQRT_Y_DAG(2)]),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                    (3, Qubit(2)): {"preceding": 3, "succeeding": 6},
                },
                {
                    0: [X(0)],
                    1: [Y(1)],
                    2: [S_DAG(1)],
                    3: [Z(2), SQRT_X(2)],
                    4: [S(0), SQRT_X_DAG(0)],
                    5: [SQRT_Y(1)],
                    6: [SQRT_Y_DAG(2)],
                },
            ],
            # 2q gates interleaving on 2 qubits
            [
                Circuit(
                    [GateLayer(CX(0, 1)), GateLayer(CX(1, 2)), GateLayer(CX(2, 3))]
                ),
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 6},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 2, "succeeding": 7},
                    (1, Qubit(2)): {"preceding": 3, "succeeding": 4},
                    (2, Qubit(2)): {"preceding": 4, "succeeding": 8},
                    (2, Qubit(3)): {"preceding": 5, "succeeding": 9},
                },
                {
                    0: [],
                    1: [],
                    2: [],
                    3: [],
                    4: [],
                    5: [],
                    6: [],
                    7: [],
                    8: [],
                    9: [],
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([CX(0, 1), Z(2)]),
                        GateLayer(
                            [
                                S_DAG(0),
                                SQRT_X(1),
                                SQRT_X_DAG(2),
                                S(3),
                            ]
                        ),
                        GateLayer([SQRT_Y_DAG(0), CX(1, 2), SQRT_Y(3)]),
                        GateLayer([S(0), X(1), Y(2), Z(3)]),
                        GateLayer([S_DAG(1), CX(2, 3)]),
                        GateLayer([SQRT_X(2), SQRT_X_DAG(3)]),
                    ]
                ),
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 6},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 2},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 7},
                    (3, Qubit(2)): {"preceding": 3, "succeeding": 4},
                    (5, Qubit(2)): {"preceding": 4, "succeeding": 8},
                    (5, Qubit(3)): {"preceding": 5, "succeeding": 9},
                },
                {
                    0: [X(0)],
                    1: [Y(1)],
                    2: [SQRT_X(1)],
                    3: [Z(2), SQRT_X_DAG(2)],
                    4: [Y(2)],
                    5: [S(3), SQRT_Y(3), Z(3)],
                    6: [S_DAG(0), SQRT_Y_DAG(0), S(0)],
                    7: [X(1), S_DAG(1)],
                    8: [SQRT_X(2)],
                    9: [SQRT_X_DAG(3)],
                },
            ],
        ],
    )
    class TestTwoQubitGates:
        def test_extract_structure_from_circuit_gives_correct_output_for_two_qubit_gates(
            self, circuit, two_q_dict, unitary_blocks, two_qubit_gate: Gate
        ):
            circuit.replace_gates({CX: lambda gate: two_qubit_gate(*gate.qubits)})
            results = _extract_structure_from_circuit(circuit)
            assert results.unitary_blocks == unitary_blocks
            assert results.two_qubit_gates == {
                (*key, two_qubit_gate.stim_string): value
                for key, value in two_q_dict.items()
            }
            circuit.replace_gates({two_qubit_gate: lambda gate: CX(*gate.qubits)})

        def test_create_circuit_from_compilation_data_returns_correct_circuit_for_two_qubit_gates(
            self, unitary_blocks, two_q_dict, circuit: Circuit, two_qubit_gate
        ):
            circuit.replace_gates({CX: lambda gate: two_qubit_gate(*gate.qubits)})
            two_q_dict = {
                (*key, two_qubit_gate.stim_string): value
                for key, value in two_q_dict.items()
            }
            comp_data = CompilationData(unitary_blocks, {}, {}, two_q_dict, {})
            assert (
                _create_circuit_from_compilation_data(
                    comp_data, {i: i for i in range(len(circuit.layers))}
                )
                == circuit
            )
            circuit.replace_gates({two_qubit_gate: lambda gate: CX(*gate.qubits)})

    @pytest.mark.parametrize(
        "circuit, non_gatelayer_dict",
        [
            [
                Circuit(Detector(MeasurementRecord(-1))),
                {0: Detector(MeasurementRecord(-1))},
            ],
            [
                Circuit([Detector(MeasurementRecord(-1))]),
                {0: Detector(MeasurementRecord(-1))},
            ],
            [
                Circuit(Observable(0, MeasurementRecord(-1))),
                {0: Observable(0, MeasurementRecord(-1))},
            ],
            [
                Circuit([Observable(0, MeasurementRecord(-1))]),
                {0: Observable(0, MeasurementRecord(-1))},
            ],
            [Circuit(ShiftCoordinates([1])), {0: ShiftCoordinates([1])}],
            [
                Circuit([ShiftCoordinates([1])]),
                {0: ShiftCoordinates([1])},
            ],
            [
                Circuit(NoiseLayer(Depolarise2(0, 1, 0.01))),
                {0: NoiseLayer(Depolarise2(0, 1, 0.01))},
            ],
            [
                Circuit([NoiseLayer(Depolarise2(0, 1, 0.01)), ShiftCoordinates([1])]),
                {0: NoiseLayer(Depolarise2(0, 1, 0.01)), 1: ShiftCoordinates([1])},
            ],
            [
                Circuit(
                    [
                        NoiseLayer(Depolarise2(0, 1, 0.01)),
                        ShiftCoordinates([1]),
                        NoiseLayer(Depolarise2(0, 1, 0.01)),
                    ]
                ),
                {
                    0: NoiseLayer(Depolarise2(0, 1, 0.01)),
                    1: ShiftCoordinates([1]),
                    2: NoiseLayer(Depolarise2(0, 1, 0.01)),
                },
            ],
        ],
    )
    class TestNonGateLayers:
        def test_extract_structure_from_circuit_gives_correct_output_for_none_GateLayers(
            self, circuit, non_gatelayer_dict
        ):
            assert (
                _extract_structure_from_circuit(circuit).non_gatelayer_layers
                == non_gatelayer_dict
            )

        def test_create_circuit_from_compilation_data_correct_for_non_gatelayer_layers(
            self, circuit, non_gatelayer_dict
        ):
            assert (
                _create_circuit_from_compilation_data(
                    CompilationData({}, {}, {}, {}, non_gatelayer_dict),
                    {i: i for i in range(len(circuit.layers))},
                )
                == circuit
            )

    @pytest.mark.parametrize(
        "reset_gate",
        [
            RX,
            RY,
            RZ,
        ],
    )
    @pytest.mark.parametrize(
        "meas_gate",
        [
            MX,
            MY,
            MZ,
            MRX,
            MRY,
            MRZ,
        ],
    )
    @pytest.mark.parametrize(
        "circuit, unitary_blocks, reset_dict, meas_dict",
        [
            [
                Circuit([GateLayer(RZ(0)), GateLayer(MZ(0))]),
                {0: [], 1: [], 2: []},
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(1, Qubit(0)): {"preceding": 1, "succeeding": 2}},
            ],
            [
                Circuit([GateLayer(RZ(0)), GateLayer(X(0)), GateLayer(MZ(0))]),
                {0: [], 1: [X(0)], 2: []},
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(2, Qubit(0)): {"preceding": 1, "succeeding": 2}},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(RZ(0)),
                        GateLayer(Y(0)),
                        GateLayer(MZ(0)),
                    ]
                ),
                {0: [X(0)], 1: [Y(0)], 2: []},
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(3, Qubit(0)): {"preceding": 1, "succeeding": 2}},
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(RZ(0)),
                        GateLayer(Y(0)),
                        GateLayer(MZ(0)),
                        GateLayer(Z(0)),
                    ]
                ),
                {0: [X(0)], 1: [Y(0)], 2: [Z(0)]},
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(3, Qubit(0)): {"preceding": 1, "succeeding": 2}},
            ],
            [
                Circuit([GateLayer([RZ(0), RZ(1)]), GateLayer(MZ(0))]),
                {0: [], 1: [], 2: [], 3: [], 4: []},
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 4},
                },
                {(1, Qubit(0)): {"preceding": 2, "succeeding": 3}},
            ],
            [
                Circuit([GateLayer([RZ(0), RZ(1)]), GateLayer(X(0)), GateLayer(MZ(0))]),
                {0: [], 1: [], 2: [X(0)], 3: [], 4: []},
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 4},
                },
                {(2, Qubit(0)): {"preceding": 2, "succeeding": 3}},
            ],
            [
                Circuit(
                    [
                        GateLayer([RZ(0), RZ(1)]),
                        GateLayer([X(0), Y(1)]),
                        GateLayer(MZ(0)),
                    ]
                ),
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [Y(1)]},
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 4},
                },
                {(2, Qubit(0)): {"preceding": 2, "succeeding": 3}},
            ],
            [
                Circuit(
                    [
                        GateLayer([RZ(0), RZ(1)]),
                        GateLayer([X(0), Y(1)]),
                        GateLayer([MZ(0), Z(1)]),
                    ]
                ),
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [Y(1), Z(1)]},
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 4},
                },
                {(2, Qubit(0)): {"preceding": 2, "succeeding": 3}},
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([RZ(0), RZ(1)]),
                        GateLayer([Z(0), S(1)]),
                        GateLayer(MZ(0)),
                    ]
                ),
                {0: [X(0)], 1: [Y(1)], 2: [Z(0)], 3: [], 4: [S(1)]},
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 4},
                },
                {(3, Qubit(0)): {"preceding": 2, "succeeding": 3}},
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([RZ(0), RZ(1)]),
                        GateLayer([Z(0), S(1)]),
                        GateLayer([MZ(0), S_DAG(1)]),
                    ]
                ),
                {0: [X(0)], 1: [Y(1)], 2: [Z(0)], 3: [], 4: [S(1), S_DAG(1)]},
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 4},
                },
                {(3, Qubit(0)): {"preceding": 2, "succeeding": 3}},
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([RZ(0), RZ(1)]),
                        GateLayer([Z(0), S(1)]),
                        GateLayer([MZ(0), S_DAG(1)]),
                        GateLayer(SQRT_X(0)),
                    ]
                ),
                {0: [X(0)], 1: [Y(1)], 2: [Z(0)], 3: [SQRT_X(0)], 4: [S(1), S_DAG(1)]},
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 4},
                },
                {(3, Qubit(0)): {"preceding": 2, "succeeding": 3}},
            ],
            [
                Circuit([GateLayer([RZ(0), RZ(1)]), GateLayer([MZ(0), MZ(1)])]),
                {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {
                    (1, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([RZ(0), RZ(1)]),
                        GateLayer(X(0)),
                        GateLayer([MZ(0), MZ(1)]),
                    ]
                ),
                {0: [], 1: [], 2: [X(0)], 3: [], 4: [], 5: []},
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {
                    (2, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (2, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([RZ(0), RZ(1)]),
                        GateLayer([Z(0), S(1)]),
                        GateLayer([MZ(0), MZ(1)]),
                        GateLayer([S_DAG(0), SQRT_X(1)]),
                    ]
                ),
                {
                    0: [X(0)],
                    1: [Y(1)],
                    2: [Z(0)],
                    3: [S(1)],
                    4: [S_DAG(0)],
                    5: [SQRT_X(1)],
                },
                {
                    (1, Qubit(0)): {"preceding": 0, "succeeding": 2},
                    (1, Qubit(1)): {"preceding": 1, "succeeding": 3},
                },
                {
                    (3, Qubit(0)): {"preceding": 2, "succeeding": 4},
                    (3, Qubit(1)): {"preceding": 3, "succeeding": 5},
                },
            ],
        ],
    )
    class TestResetAndMeas:
        def test_extract_structure_from_circuit_gives_correct_output_for_reset_and_meas_in_same_circ(
            self,
            circuit,
            unitary_blocks,
            reset_dict,
            meas_dict,
            reset_gate,
            meas_gate,
        ):
            circuit.replace_gates({RZ: lambda gate: reset_gate(gate.qubit)})
            circuit.replace_gates({MZ: lambda gate: meas_gate(gate.qubit)})
            results = _extract_structure_from_circuit(circuit)
            assert results.unitary_blocks == unitary_blocks
            assert results.reset_gates == {
                (*key, reset_gate.stim_string): value
                for key, value in reset_dict.items()
            }
            assert results.measurement_gates == {
                (*key, meas_gate.stim_string): value for key, value in meas_dict.items()
            }
            circuit.replace_gates({reset_gate: lambda gate: RZ(gate.qubit)})
            circuit.replace_gates({meas_gate: lambda gate: MZ(gate.qubit)})

        def test_create_circuit_from_compilation_data_gives_correct_output_for_reset_and_meas_in_same_circ(
            self,
            circuit,
            unitary_blocks,
            reset_dict,
            meas_dict,
            reset_gate,
            meas_gate,
        ):
            circuit.replace_gates({RZ: lambda gate: reset_gate(gate.qubit)})
            circuit.replace_gates({MZ: lambda gate: meas_gate(gate.qubit)})
            reset_dict = {
                (*key, reset_gate.stim_string): value
                for key, value in reset_dict.items()
            }
            meas_dict = {
                (*key, meas_gate.stim_string): value for key, value in meas_dict.items()
            }
            assert (
                _create_circuit_from_compilation_data(
                    CompilationData(unitary_blocks, reset_dict, meas_dict, {}, {}),
                    {i: i for i in range(len(circuit.layers))},
                )
                == circuit
            )
            circuit.replace_gates({reset_gate: lambda gate: RZ(gate.qubit)})
            circuit.replace_gates({meas_gate: lambda gate: MZ(gate.qubit)})

    @pytest.mark.parametrize(
        "reset_gate",
        [
            RX,
            RY,
            RZ,
        ],
    )
    @pytest.mark.parametrize(
        "meas_gate",
        [
            MX,
            MY,
            MZ,
            MRX,
            MRY,
            MRZ,
        ],
    )
    @pytest.mark.parametrize("two_qubit_gate", [CX, CY, CZ, SWAP, ISWAP, SQRT_XX])
    @pytest.mark.parametrize(
        "circuit, unitary_blocks, reset_dict, meas_dict, two_q_dict",
        [
            [
                Circuit([GateLayer(RZ(0)), GateLayer(CX(0, 1)), GateLayer(MZ(0))]),
                {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(2, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (1, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (1, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [GateLayer([RZ(0), X(1)]), GateLayer(CX(0, 1)), GateLayer(MZ(0))]
                ),
                {0: [], 1: [], 2: [X(1)], 3: [], 4: [], 5: []},
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(2, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (1, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (1, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([RZ(0), X(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([MZ(0), Y(1)]),
                    ]
                ),
                {0: [], 1: [], 2: [X(1)], 3: [], 4: [], 5: [Y(1)]},
                {(0, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(2, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (1, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (1, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer([RZ(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([MZ(0), Z(1)]),
                    ]
                ),
                {0: [X(0)], 1: [], 2: [Y(1)], 3: [], 4: [], 5: [Z(1)]},
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(3, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (2, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (2, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(RZ(0)),
                        GateLayer([Z(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([MZ(0), S(1)]),
                    ]
                ),
                {0: [X(0)], 1: [Z(0)], 2: [Y(1)], 3: [], 4: [], 5: [S(1)]},
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(4, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(RZ(0)),
                        GateLayer([Z(0), Y(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([S(0), S_DAG(1)]),
                        GateLayer([MZ(0)]),
                    ]
                ),
                {0: [X(0)], 1: [Z(0)], 2: [Y(1)], 3: [S(0)], 4: [], 5: [S_DAG(1)]},
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(5, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([RZ(0), Z(1)]),
                        GateLayer([S(0), S_DAG(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([SQRT_X(0), SQRT_X_DAG(1)]),
                        GateLayer([MZ(0), SQRT_Y(1)]),
                    ]
                ),
                {
                    0: [X(0)],
                    1: [S(0)],
                    2: [Y(1), Z(1), S_DAG(1)],
                    3: [SQRT_X(0)],
                    4: [],
                    5: [SQRT_X_DAG(1), SQRT_Y(1)],
                },
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(5, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1)]),
                        GateLayer([RZ(0), Z(1)]),
                        GateLayer([S(0), S_DAG(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([SQRT_X(0), SQRT_X_DAG(1)]),
                        GateLayer([MZ(0), SQRT_Y(1)]),
                        GateLayer([SQRT_Y_DAG(0), X(1)]),
                    ]
                ),
                {
                    0: [X(0)],
                    1: [S(0)],
                    2: [Y(1), Z(1), S_DAG(1)],
                    3: [SQRT_X(0)],
                    4: [SQRT_Y_DAG(0)],
                    5: [SQRT_X_DAG(1), SQRT_Y(1), X(1)],
                },
                {(1, Qubit(0)): {"preceding": 0, "succeeding": 1}},
                {(5, Qubit(0)): {"preceding": 3, "succeeding": 4}},
                {
                    (3, Qubit(0)): {"preceding": 1, "succeeding": 3},
                    (3, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            # interleaved 2q gates
            [
                Circuit(
                    [
                        GateLayer([RZ(0), RZ(1), RZ(2)]),
                        GateLayer(CX(0, 1)),
                        GateLayer(CX(1, 2)),
                        GateLayer([MZ(0), MZ(1), MZ(2)]),
                    ]
                ),
                {
                    0: [],
                    1: [],
                    2: [],
                    3: [],
                    4: [],
                    5: [],
                    6: [],
                    7: [],
                    8: [],
                    9: [],
                    10: [],
                    11: [],
                    12: [],
                },
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 3},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 4},
                    (0, Qubit(2)): {"preceding": 2, "succeeding": 6},
                },
                {
                    (3, Qubit(0)): {"preceding": 7, "succeeding": 10},
                    (3, Qubit(1)): {"preceding": 8, "succeeding": 11},
                    (3, Qubit(2)): {"preceding": 9, "succeeding": 12},
                },
                {
                    (1, Qubit(0)): {"preceding": 3, "succeeding": 7},
                    (1, Qubit(1)): {"preceding": 4, "succeeding": 5},
                    (2, Qubit(1)): {"preceding": 5, "succeeding": 8},
                    (2, Qubit(2)): {"preceding": 6, "succeeding": 9},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([RZ(0), RZ(1), RZ(2)]),
                        GateLayer([X(0), Y(1), Z(2)]),
                        GateLayer(CX(0, 1)),
                        GateLayer(CX(1, 2)),
                        GateLayer([MZ(0), MZ(1), MZ(2)]),
                    ]
                ),
                {
                    0: [],
                    1: [],
                    2: [],
                    3: [X(0)],
                    4: [Y(1)],
                    5: [],
                    6: [Z(2)],
                    7: [],
                    8: [],
                    9: [],
                    10: [],
                    11: [],
                    12: [],
                },
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 3},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 4},
                    (0, Qubit(2)): {"preceding": 2, "succeeding": 6},
                },
                {
                    (4, Qubit(0)): {"preceding": 7, "succeeding": 10},
                    (4, Qubit(1)): {"preceding": 8, "succeeding": 11},
                    (4, Qubit(2)): {"preceding": 9, "succeeding": 12},
                },
                {
                    (2, Qubit(0)): {"preceding": 3, "succeeding": 7},
                    (2, Qubit(1)): {"preceding": 4, "succeeding": 5},
                    (3, Qubit(1)): {"preceding": 5, "succeeding": 8},
                    (3, Qubit(2)): {"preceding": 6, "succeeding": 9},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([RZ(0), RZ(1), RZ(2)]),
                        GateLayer([X(0), Y(1), Z(2)]),
                        GateLayer([CX(0, 1), SQRT_X(2)]),
                        GateLayer([S_DAG(1)]),
                        GateLayer([S(0), CX(1, 2)]),
                        GateLayer([SQRT_X_DAG(0), SQRT_Y(1), SQRT_Y_DAG(2)]),
                        GateLayer([MZ(0), MZ(1), MZ(2)]),
                    ]
                ),
                {
                    0: [],
                    1: [],
                    2: [],
                    3: [X(0)],
                    4: [Y(1)],
                    5: [S_DAG(1)],
                    6: [Z(2), SQRT_X(2)],
                    7: [S(0), SQRT_X_DAG(0)],
                    8: [SQRT_Y(1)],
                    9: [SQRT_Y_DAG(2)],
                    10: [],
                    11: [],
                    12: [],
                },
                {
                    (0, Qubit(0)): {"preceding": 0, "succeeding": 3},
                    (0, Qubit(1)): {"preceding": 1, "succeeding": 4},
                    (0, Qubit(2)): {"preceding": 2, "succeeding": 6},
                },
                {
                    (6, Qubit(0)): {"preceding": 7, "succeeding": 10},
                    (6, Qubit(1)): {"preceding": 8, "succeeding": 11},
                    (6, Qubit(2)): {"preceding": 9, "succeeding": 12},
                },
                {
                    (2, Qubit(0)): {"preceding": 3, "succeeding": 7},
                    (2, Qubit(1)): {"preceding": 4, "succeeding": 5},
                    (4, Qubit(1)): {"preceding": 5, "succeeding": 8},
                    (4, Qubit(2)): {"preceding": 6, "succeeding": 9},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1), Z(2), S(3)]),
                        GateLayer([CX(0, 1), RZ(2), MZ(3)]),
                    ]
                ),
                {
                    0: [Z(2)],
                    1: [X(0)],
                    2: [Y(1)],
                    3: [S(3)],
                    4: [],
                    5: [],
                    6: [],
                    7: [],
                },
                {
                    (1, Qubit(2)): {"preceding": 0, "succeeding": 6},
                },
                {
                    (1, Qubit(3)): {"preceding": 3, "succeeding": 7},
                },
                {
                    (1, Qubit(0)): {"preceding": 1, "succeeding": 4},
                    (1, Qubit(1)): {"preceding": 2, "succeeding": 5},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1), Z(2), S(3)]),
                        GateLayer([CX(0, 2), RZ(1), MZ(3)]),
                    ]
                ),
                {
                    0: [Y(1)],
                    1: [X(0)],
                    2: [Z(2)],
                    3: [S(3)],
                    4: [],
                    5: [],
                    6: [],
                    7: [],
                },
                {
                    (1, Qubit(1)): {"preceding": 0, "succeeding": 5},
                },
                {
                    (1, Qubit(3)): {"preceding": 3, "succeeding": 7},
                },
                {
                    (1, Qubit(0)): {"preceding": 1, "succeeding": 4},
                    (1, Qubit(2)): {"preceding": 2, "succeeding": 6},
                },
            ],
            [
                Circuit(
                    [
                        GateLayer([X(0), Y(1), Z(2), S(3)]),
                        GateLayer([CX(2, 0), RZ(1), MZ(3)]),
                    ]
                ),
                {
                    0: [Y(1)],
                    1: [Z(2)],
                    2: [X(0)],
                    3: [S(3)],
                    4: [],
                    5: [],
                    6: [],
                    7: [],
                },
                {
                    (1, Qubit(1)): {"preceding": 0, "succeeding": 5},
                },
                {
                    (1, Qubit(3)): {"preceding": 3, "succeeding": 7},
                },
                {
                    (1, Qubit(2)): {"preceding": 1, "succeeding": 6},
                    (1, Qubit(0)): {"preceding": 2, "succeeding": 4},
                },
            ],
        ],
    )
    class TestResetMeasAndTwoQGate:
        def test_extract_structure_from_circuit_gives_correct_output_for_reset_and_meas_and_2q_gates_in_same_circ(
            self,
            circuit,
            unitary_blocks,
            reset_dict,
            meas_dict,
            two_q_dict,
            reset_gate,
            meas_gate,
            two_qubit_gate,
        ):
            circuit.replace_gates({RZ: lambda gate: reset_gate(gate.qubit)})
            circuit.replace_gates({MZ: lambda gate: meas_gate(gate.qubit)})
            circuit.replace_gates({CX: lambda gate: two_qubit_gate(*gate.qubits)})
            results = _extract_structure_from_circuit(circuit)
            assert results.unitary_blocks == unitary_blocks
            assert results.reset_gates == {
                (*key, reset_gate.stim_string): value
                for key, value in reset_dict.items()
            }
            assert results.measurement_gates == {
                (*key, meas_gate.stim_string): value for key, value in meas_dict.items()
            }
            assert results.two_qubit_gates == {
                (*key, two_qubit_gate.stim_string): value
                for key, value in two_q_dict.items()
            }
            circuit.replace_gates({reset_gate: lambda gate: RZ(gate.qubit)})
            circuit.replace_gates({meas_gate: lambda gate: MZ(gate.qubit)})
            circuit.replace_gates({two_qubit_gate: lambda gate: CX(*gate.qubits)})

        def test_create_circuit_from_compilation_data_gives_correct_output_for_reset_and_meas_and_2q_gates_in_same_circ(
            self,
            circuit,
            unitary_blocks,
            reset_dict,
            meas_dict,
            two_q_dict,
            reset_gate,
            meas_gate,
            two_qubit_gate,
        ):
            circuit.replace_gates({RZ: lambda gate: reset_gate(gate.qubit)})
            circuit.replace_gates({MZ: lambda gate: meas_gate(gate.qubit)})
            circuit.replace_gates({CX: lambda gate: two_qubit_gate(*gate.qubits)})
            reset_dict = {
                (*key, reset_gate.stim_string): value
                for key, value in reset_dict.items()
            }
            meas_dict = {
                (*key, meas_gate.stim_string): value for key, value in meas_dict.items()
            }
            two_q_dict = {
                (*key, two_qubit_gate.stim_string): value
                for key, value in two_q_dict.items()
            }
            assert (
                _create_circuit_from_compilation_data(
                    CompilationData(
                        unitary_blocks, reset_dict, meas_dict, two_q_dict, {}
                    ),
                    {i: i for i in range(len(circuit.layers))},
                )
                == circuit
            )
            circuit.replace_gates({reset_gate: lambda gate: RZ(gate.qubit)})
            circuit.replace_gates({meas_gate: lambda gate: MZ(gate.qubit)})
            circuit.replace_gates({two_qubit_gate: lambda gate: CX(*gate.qubits)})

    @pytest.mark.parametrize(
        "circuit, non_gatelayer_layers, reset_dict, unitary_blocks",
        [
            [
                Circuit(Circuit(GateLayer(RZ(0)), iterations=2)),
                {
                    0: CompilationData(
                        {0: [], 1: []},
                        {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                        {},
                        {},
                        {},
                        2,
                        1,
                    )
                },
                {},
                {0: [], 1: []},
            ],
            [
                Circuit([GateLayer(RZ(0)), Circuit(GateLayer(RZ(0)), iterations=2)]),
                {
                    1: CompilationData(
                        {0: [], 1: []},
                        {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                        {},
                        {},
                        {},
                        2,
                        1,
                    )
                },
                {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                {0: [], 1: []},
            ],
            [
                Circuit(
                    [
                        GateLayer(RZ(0)),
                        Circuit(GateLayer(RZ(0)), iterations=2),
                        GateLayer(RZ(0)),
                    ]
                ),
                {
                    1: CompilationData(
                        {0: [], 1: []},
                        {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                        {},
                        {},
                        {},
                        2,
                        1,
                    )
                },
                {
                    (0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1},
                    (2, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2},
                },
                {0: [], 1: [], 2: []},
            ],
            [
                Circuit(Circuit(Circuit(GateLayer(RZ(0)), iterations=2), iterations=2)),
                {
                    0: CompilationData(
                        {0: []},
                        {},
                        {},
                        {},
                        {
                            0: CompilationData(
                                {0: [], 1: []},
                                {
                                    (0, Qubit(0), "RZ"): {
                                        "preceding": 0,
                                        "succeeding": 1,
                                    }
                                },
                                {},
                                {},
                                {},
                                2,
                                1,
                            )
                        },
                        2,
                        1,
                    )
                },
                {},
                {},
            ],
        ],
    )
    class TestNestedCircuits:
        def test_extract_structure_from_circuit_gives_correct_output_for_nested_Circuits(
            self, circuit, non_gatelayer_layers, reset_dict, unitary_blocks
        ):
            results = _extract_structure_from_circuit(circuit)
            assert results.non_gatelayer_layers == non_gatelayer_layers

    def test_extract_structure_from_circuit_raises_NotImplementedError_if_MPP_present(
        self,
    ):
        with pytest.raises(NotImplementedError, match=r"MPP gates not yet supported"):
            _extract_structure_from_circuit(
                Circuit(GateLayer(MPP((PauliX(0), PauliX(1)))))
            )


class TestCompileCircuitWithTableau:
    @pytest.fixture
    def standard_gate_set_z(self):
        return NativeGateSet(
            one_qubit_gates={X, Z, S, H},
            two_qubit_gates={CX},
            reset_gates={RZ},
            measurement_gates={MZ},
        )

    @pytest.fixture
    def standard_gate_set_x(self):
        return NativeGateSet(
            one_qubit_gates={X, Z, S, H},
            two_qubit_gates={CX},
            reset_gates={RX},
            measurement_gates={MX},
        )

    def test_compile_circuit_tableau_valid_for_trivial_circuit(self):
        assert compile_circuit_to_native_gates(Circuit(), NativeGateSet()) == Circuit()

    @pytest.mark.parametrize("up_to_paulis", [False, True])
    @pytest.mark.parametrize(
        "circuit, expected_circuit",
        [
            [Circuit(GateLayer(X(0))), Circuit()],
            [
                Circuit([GateLayer(X(0)), GateLayer(X(0)), GateLayer(MZ(0))]),
                Circuit([GateLayer(MZ(0))]),
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(X(0)), GateLayer(RZ(0))]),
                Circuit([GateLayer(RZ(0))]),
            ],
            [
                Circuit([GateLayer(X(0)), GateLayer(X(0)), GateLayer(CX(0, 1))]),
                Circuit([GateLayer(CX(0, 1))]),
            ],
            [
                Circuit([GateLayer(MZ(0)), GateLayer(X(0)), GateLayer(X(0))]),
                Circuit([GateLayer(MZ(0))]),
            ],
            [
                Circuit([GateLayer(RZ(0)), GateLayer(X(0)), GateLayer(X(0))]),
                Circuit([GateLayer(RZ(0))]),
            ],
            [
                Circuit([GateLayer(CX(0, 1)), GateLayer(X(0)), GateLayer(X(0))]),
                Circuit([GateLayer(CX(0, 1))]),
            ],
            [
                Circuit([GateLayer(RZ(0)), GateLayer(Z(0)), GateLayer(MZ(0))]),
                Circuit([GateLayer(RZ(0)), GateLayer(MZ(0))]),
            ],
            [
                Circuit(
                    [
                        GateLayer(RZ(0)),
                        GateLayer(Z(0)),
                        GateLayer(MZ(0)),
                        GateLayer(Z(0)),
                    ]
                ),
                Circuit([GateLayer(RZ(0)), GateLayer(MZ(0))]),
            ],
            [
                Circuit([GateLayer(Z(0)), GateLayer(CX(0, 1)), GateLayer(RZ(0))]),
                Circuit([GateLayer(CX(0, 1)), GateLayer(RZ(0))]),
            ],
            [
                Circuit([GateLayer(X(1)), GateLayer(CX(0, 1)), GateLayer(RZ(1))]),
                Circuit([GateLayer(CX(0, 1)), GateLayer(RZ(1))]),
            ],
            [
                Circuit(
                    [GateLayer([Z(0), Z(1)]), GateLayer(CX(0, 1)), GateLayer(RZ(1))]
                ),
                Circuit([GateLayer(CX(0, 1)), GateLayer(RZ(1))]),
            ],
            [
                Circuit(
                    [
                        GateLayer([Z(0), Z(1)]),
                        GateLayer(CX(0, 1)),
                        GateLayer([RZ(0), RZ(1)]),
                    ]
                ),
                Circuit([GateLayer(CX(0, 1)), GateLayer([RZ(0), RZ(1)])]),
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(H(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(RZ(0)),
                    ]
                ),
                Circuit([GateLayer(H(0)), GateLayer(CX(0, 1)), GateLayer(RZ(0))]),
            ],
            [
                Circuit(
                    [
                        GateLayer(X(0)),
                        GateLayer(H(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(MZ(0)),
                    ]
                ),
                Circuit([GateLayer(H(0)), GateLayer(CX(0, 1)), GateLayer(MZ(0))]),
            ],
            [
                Circuit(
                    [
                        GateLayer(RZ(0)),
                        GateLayer(X(0)),
                        GateLayer(H(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(MZ(0)),
                        GateLayer(RZ(0)),
                    ]
                ),
                Circuit(
                    [
                        GateLayer(RZ(0)),
                        GateLayer(H(0)),
                        GateLayer(CX(0, 1)),
                        GateLayer(MZ(0)),
                        GateLayer(RZ(0)),
                    ]
                ),
            ],
        ],
    )
    def test_compile_circuit_with_tableau_removes_unnecessary_pauli_terms(
        self, circuit, expected_circuit, standard_gate_set_z, up_to_paulis
    ):
        assert (
            compile_circuit_to_native_gates(
                circuit, standard_gate_set_z, up_to_paulis=up_to_paulis
            )
            == expected_circuit
        )

    @pytest.mark.parametrize("up_to_paulis", [False, True])
    @pytest.mark.parametrize(
        "circuit, expected_circuit",
        [
            [Circuit([GateLayer(X(0))]), Circuit()],
            [Circuit([GateLayer([X(0), Z(1)])]), Circuit()],
            [
                Circuit([GateLayer([X(0), Z(1)]), GateLayer(RZ(0))]),
                Circuit(GateLayer(RZ(0))),
            ],
            [
                Circuit([GateLayer([X(0), Z(1)]), GateLayer(RZ(1))]),
                Circuit(GateLayer(RZ(1))),
            ],
            [
                Circuit([GateLayer([X(0), Z(4)]), GateLayer(RZ(0))]),
                Circuit(GateLayer(RZ(0))),
            ],
            [
                Circuit([GateLayer([X(10), Z(1)]), GateLayer(RZ(1))]),
                Circuit(GateLayer(RZ(1))),
            ],
            [
                Circuit([GateLayer([X(0), Z(1)]), GateLayer(MZ(1))]),
                Circuit(GateLayer(MZ(1))),
            ],
            [
                Circuit([GateLayer([X(10), Z(1)]), GateLayer(MZ(1))]),
                Circuit(GateLayer(MZ(1))),
            ],
        ],
    )
    def test_compile_circuit_with_tableau_drops_qubits_with_no_special_gates(
        self, circuit, expected_circuit, standard_gate_set_z, up_to_paulis
    ):
        assert (
            compile_circuit_to_native_gates(
                circuit, standard_gate_set_z, up_to_paulis=up_to_paulis
            )
            == expected_circuit
        )

    @pytest.mark.parametrize(
        "circuit",
        [
            Circuit(
                [
                    GateLayer(RZ(0)),
                    GateLayer(H(0)),
                    GateLayer(CX(0, 1)),
                    GateLayer(CX(0, 2)),
                    GateLayer(H(0)),
                    GateLayer(MZ(0)),
                ]
            ),
            Circuit(
                [
                    GateLayer(RZ(0)),
                    GateLayer(H(0)),
                    GateLayer(CX(0, 1)),
                    GateLayer(CX(0, 2)),
                    GateLayer(CX(0, 3)),
                    GateLayer(CX(0, 4)),
                    GateLayer(H(0)),
                    GateLayer(MZ(0)),
                ]
            ),
        ],
    )
    def test_compile_circuit_with_tableau_does_not_change_already_optimal_circuits(
        self, circuit, standard_gate_set_z
    ):
        assert compile_circuit_to_native_gates(circuit, standard_gate_set_z) == circuit

    @pytest.mark.parametrize(
        "circuit",
        [
            Circuit(
                [
                    GateLayer(RZ(0)),
                    GateLayer(H(0)),
                    GateLayer(CZ(1, 0)),
                    GateLayer(CZ(2, 0)),
                    GateLayer(H(0)),
                    GateLayer(MZ(0)),
                ]
            ),
            Circuit(
                [
                    GateLayer(RZ(0)),
                    GateLayer(H(0)),
                    GateLayer(CZ(1, 0)),
                    GateLayer(CZ(2, 0)),
                    GateLayer(CZ(3, 0)),
                    GateLayer(CZ(4, 0)),
                    GateLayer(H(0)),
                    GateLayer(MZ(0)),
                ]
            ),
        ],
    )
    class TestCompilationReducesCircuits:
        def test_compile_circuit_with_tableau_reduces_circuits_after_compilation_to_native_gates(
            self, circuit, standard_gate_set_z
        ):
            compiled_circ = compile_circuit_to_native_gates(
                circuit, standard_gate_set_z
            )
            assert len(compiled_circ.layers) < len(circuit.layers)

        def test_no_non_native_gates_left_after_compilation(
            self, circuit, standard_gate_set_z
        ):
            compiled_circ = compile_circuit_to_native_gates(
                circuit, standard_gate_set_z
            )
            for gate_layer in compiled_circ.gate_layers():
                for gate in gate_layer.gates:
                    assert type(gate) in standard_gate_set_z.native_gates

    @pytest.mark.parametrize(
        "code",
        [
            UnrotatedPlanarCode,
            RotatedPlanarCode,
        ],
    )
    @pytest.mark.parametrize("basis", [PauliBasis.X, PauliBasis.Z])
    @pytest.mark.parametrize("d", [3, 5, 7, 9])
    @pytest.mark.parametrize(
        "native_gate_set",
        [
            NativeGateSet(
                one_qubit_gates={I},
                two_qubit_gates={CX, CZ},
                reset_gates={RX, RZ},
                measurement_gates={MX, MZ},
            ),
            NativeGateSet(
                one_qubit_gates={H, S, X},
                two_qubit_gates={CX, CZ},
                reset_gates={RX, RZ},
                measurement_gates={MX, MZ},
            ),
        ],
    )
    class TestCompileMemoryCircuit:
        def test_compile_circuit_with_tableau_removes_identities_from_memory_circuit(
            self, basis, d, code, native_gate_set
        ):
            rpc = code(d, d)
            circ = css_code_memory_circuit(rpc, d, basis)
            assert len(
                compile_circuit_to_native_gates(
                    circ,
                    native_gate_set,
                ).layers
            ) < len(circ.layers)

        def test_no_non_native_gates_left_after_compilation(
            self, basis, d, code, native_gate_set
        ):
            rpc = code(d, d)
            circ = css_code_memory_circuit(rpc, d, basis)
            compiled_circ = compile_circuit_to_native_gates(
                circ,
                native_gate_set,
            )
            for gate_layer in compiled_circ.gate_layers():
                for gate in gate_layer.gates:
                    assert type(gate) in native_gate_set.native_gates

    def test_compile_circuit_with_tableau_throws_error_if_MPP_in_native_gates(self):
        with pytest.raises(NotImplementedError, match=r"MPP gates not yet supported"):
            compile_circuit_to_native_gates(
                Circuit(GateLayer(MPP([PauliX(0), PauliX(1)]))),
                NativeGateSet(
                    one_qubit_gates={I},
                    two_qubit_gates={CX},
                    reset_gates={RZ},
                    measurement_gates={MPP},
                ),
            )

    @pytest.mark.parametrize(
        "circ, native_gate_set, expected_circ",
        [
            [
                Circuit([GateLayer([CX(1, 0)]), GateLayer(MX(1)), GateLayer(RX(1))]),
                NativeGateSet(
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({SQRT_XX}),
                ),
                Circuit(
                    [
                        GateLayer(H(1)),
                        GateLayer(SQRT_XX(1, 0)),
                        GateLayer([S(0)]),
                        GateLayer([H(0)]),
                        GateLayer([S(1), S(0)]),
                        GateLayer(H(1)),
                        GateLayer(MZ(1)),
                        GateLayer(RZ(1)),
                        GateLayer(H(1)),
                    ]
                ),
            ],
            [
                Circuit([GateLayer([CX(1, 0)]), GateLayer(MX(1)), GateLayer(RX(1))]),
                NativeGateSet(
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({CZ}),
                ),
                Circuit(
                    [
                        GateLayer(H(0)),
                        GateLayer(CZ(1, 0)),
                        GateLayer([H(0), H(1)]),
                        GateLayer(MZ(1)),
                        GateLayer(RZ(1)),
                        GateLayer(H(1)),
                    ]
                ),
            ],
            [
                Circuit([GateLayer([CX(1, 0)]), GateLayer(MX(1)), GateLayer(RX(1))]),
                NativeGateSet(
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({CZ}),
                ),
                Circuit(
                    [
                        GateLayer(H(0)),
                        GateLayer(CZ(1, 0)),
                        GateLayer([H(0), H(1)]),
                        GateLayer(MZ(1)),
                        GateLayer(RZ(1)),
                        GateLayer(H(1)),
                    ]
                ),
            ],
            [
                Circuit([GateLayer([CX(1, 0)]), GateLayer(MX(1)), GateLayer(RX(1))]),
                NativeGateSet(
                    one_qubit_gates=set({H, S}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({YCX}),
                ),
                Circuit(
                    [
                        GateLayer(H(1)),
                        GateLayer(S(1)),
                        GateLayer(YCX(1, 0)),
                        GateLayer(MZ(1)),
                        GateLayer(RZ(1)),
                        GateLayer(H(1)),
                    ]
                ),
            ],
            [
                Circuit([GateLayer([RX(0), I(1)]), GateLayer([CZ(1, 0)])]),
                NativeGateSet(
                    one_qubit_gates=set({H, S_DAG}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({SQRT_XX}),
                ),
                Circuit(
                    [
                        GateLayer([RX(0)]),
                        GateLayer([H(0), H(1)]),
                        GateLayer(SQRT_XX(1, 0)),
                        GateLayer([H(0), H(1)]),
                        GateLayer([S_DAG(0), S_DAG(1)]),
                    ]
                ),
            ],
            [
                Circuit([GateLayer([RX(0), I(1)]), GateLayer([CZ(1, 0)])]),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({CX}),
                ),
                Circuit(
                    [
                        GateLayer(RX(0)),
                        GateLayer(H(0)),
                        GateLayer(CX(1, 0)),
                        GateLayer(H(0)),
                    ]
                ),
            ],
            [
                Circuit([GateLayer([RX(1), I(0)]), GateLayer([CZ(0, 1)])]),
                NativeGateSet(
                    one_qubit_gates=set({H, S_DAG}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({SQRT_XX}),
                ),
                Circuit(
                    [
                        GateLayer([RX(1)]),
                        GateLayer([H(1), H(0)]),
                        GateLayer(SQRT_XX(0, 1)),
                        GateLayer([H(1), H(0)]),
                        GateLayer([S_DAG(1), S_DAG(0)]),
                    ]
                ),
            ],
            [
                Circuit([GateLayer([RX(0)]), GateLayer([CZ(1, 0)])]),
                NativeGateSet(
                    one_qubit_gates=set({H, S_DAG}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MZ}),
                    two_qubit_gates=set({SQRT_XX}),
                ),
                Circuit(
                    [
                        GateLayer([RX(0)]),
                        GateLayer([H(0), H(1)]),
                        GateLayer(SQRT_XX(1, 0)),
                        GateLayer([H(0), H(1)]),
                        GateLayer([S_DAG(0), S_DAG(1)]),
                    ]
                ),
            ],
            [
                Circuit(
                    [
                        GateLayer(S(0)),
                        GateLayer(H(0)),
                        GateLayer(S(0)),
                        GateLayer(MZ(0)),
                    ]
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, S, Z}),
                    measurement_gates=set({MZ}),
                ),
                Circuit([GateLayer(S(0)), GateLayer(H(0)), GateLayer(MZ(0))]),
            ],
            [
                Circuit(
                    [
                        Circuit(
                            [
                                GateLayer(
                                    [
                                        RX(Qubit(Coord2D(1, 0))),
                                        RX(Qubit(Coord2D(3, 0))),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        I(Qubit(Coord2D(4, 0))),
                                        I(Qubit(Coord2D(2, 0))),
                                        I(Qubit(Coord2D(0, 0))),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        CX(
                                            control=Qubit(Coord2D(1, 0)),
                                            target=Qubit(Coord2D(0, 0)),
                                        ),
                                        CX(
                                            control=Qubit(Coord2D(3, 0)),
                                            target=Qubit(Coord2D(2, 0)),
                                        ),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        CX(
                                            control=Qubit(Coord2D(1, 0)),
                                            target=Qubit(Coord2D(2, 0)),
                                        ),
                                        CX(
                                            control=Qubit(Coord2D(3, 0)),
                                            target=Qubit(Coord2D(4, 0)),
                                        ),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        MX(Qubit(Coord2D(1, 0)), probability=0.0),
                                        MX(Qubit(Coord2D(3, 0)), probability=0.0),
                                    ]
                                ),
                                Detector(
                                    [MeasurementRecord(-4), MeasurementRecord(-2)],
                                    coordinate=Coord2D(1, 0, 0),
                                ),
                                Detector(
                                    [MeasurementRecord(-3), MeasurementRecord(-1)],
                                    coordinate=Coord2D(3, 0, 0),
                                ),
                                ShiftCoordinates(Coord2D(0, 0, 1)),
                            ],
                            iterations=3,
                        ),
                    ],
                    iterations=1,
                ),
                NativeGateSet(
                    one_qubit_gates={H},
                    two_qubit_gates={CZ},
                    reset_gates={RZ},
                    measurement_gates={MZ},
                ),
                Circuit(
                    [
                        Circuit(
                            [
                                GateLayer(
                                    [
                                        RZ(Qubit(Coord2D(1, 0))),
                                        RZ(Qubit(Coord2D(3, 0))),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        H(Qubit(Coord2D(1, 0))),
                                        H(Qubit(Coord2D(3, 0))),
                                        H(Qubit(Coord2D(0, 0))),
                                        H(Qubit(Coord2D(2, 0))),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        CZ(
                                            control=Qubit(Coord2D(1, 0)),
                                            target=Qubit(Coord2D(0, 0)),
                                        ),
                                        CZ(
                                            control=Qubit(Coord2D(3, 0)),
                                            target=Qubit(Coord2D(2, 0)),
                                        ),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        H(Qubit(Coord2D(0, 0))),
                                        H(Qubit(Coord2D(4, 0))),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        CZ(
                                            control=Qubit(Coord2D(1, 0)),
                                            target=Qubit(Coord2D(2, 0)),
                                        ),
                                        CZ(
                                            control=Qubit(Coord2D(3, 0)),
                                            target=Qubit(Coord2D(4, 0)),
                                        ),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        H(Qubit(Coord2D(2, 0))),
                                        H(Qubit(Coord2D(4, 0))),
                                        H(Qubit(Coord2D(1, 0))),
                                        H(Qubit(Coord2D(3, 0))),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        MZ(Qubit(Coord2D(1, 0))),
                                        MZ(Qubit(Coord2D(3, 0))),
                                    ]
                                ),
                                GateLayer(
                                    [
                                        H(Qubit(Coord2D(1, 0))),
                                        H(Qubit(Coord2D(3, 0))),
                                    ]
                                ),
                                Detector(
                                    [MeasurementRecord(-4), MeasurementRecord(-2)],
                                    coordinate=Coord2D(1, 0, 0),
                                ),
                                Detector(
                                    [MeasurementRecord(-3), MeasurementRecord(-1)],
                                    coordinate=Coord2D(3, 0, 0),
                                ),
                                ShiftCoordinates(Coord2D(0, 0, 1)),
                            ],
                            iterations=3,
                        )
                    ]
                ),
            ],
        ],
    )
    class TestMemoryCircuitSnippets:
        def test_compile_circuit_with_tableau_compiles_snippets_of_memory_circuits_correctly(
            self, circ, native_gate_set, expected_circ
        ):
            compiled_circ = compile_circuit_to_native_gates(circ, native_gate_set)
            assert compiled_circ == expected_circ

        def test_no_non_native_gates_left_after_compilation(
            self, circ, native_gate_set, expected_circ
        ):
            compiled_circ = compile_circuit_to_native_gates(circ, native_gate_set)
            for gate_layer in compiled_circ.gate_layers():
                for gate in gate_layer.gates:
                    assert type(gate) in native_gate_set.native_gates

    @pytest.mark.parametrize(
        "circ, native_gate_set, expected_circ",
        [
            [
                Circuit([GateLayer(RX(1)), Circuit(GateLayer(RX(0)))]),
                NativeGateSet(
                    reset_gates=set({RZ}),
                    one_qubit_gates=set({H}),
                ),
                Circuit(
                    [
                        GateLayer(RZ(1)),
                        GateLayer([H(1), RZ(0)]),
                        GateLayer(H(0)),
                    ]
                ),
            ],
            [
                Circuit([GateLayer(RX(1)), Circuit(GateLayer(RX(0)), iterations=2)]),
                NativeGateSet(
                    reset_gates=set({RZ}),
                    one_qubit_gates=set({H}),
                ),
                Circuit(
                    [
                        GateLayer(RZ(1)),
                        GateLayer(H(1)),
                        Circuit(
                            [
                                GateLayer(RZ(0)),
                                GateLayer(H(0)),
                            ],
                            iterations=2,
                        ),
                    ]
                ),
            ],
            [
                Circuit(
                    [
                        GateLayer(RX(1)),
                        Circuit(
                            [GateLayer(RX(0)), Circuit(GateLayer(RX(0)), iterations=3)],
                            iterations=2,
                        ),
                    ]
                ),
                NativeGateSet(
                    reset_gates=set({RZ}),
                    one_qubit_gates=set({H}),
                ),
                Circuit(
                    [
                        GateLayer(RZ(1)),
                        GateLayer(H(1)),
                        Circuit(
                            [
                                GateLayer(RZ(0)),
                                GateLayer(H(0)),
                                Circuit(
                                    [
                                        GateLayer(RZ(0)),
                                        GateLayer(H(0)),
                                    ],
                                    iterations=3,
                                ),
                            ],
                            iterations=2,
                        ),
                    ]
                ),
            ],
        ],
    )
    class TestCompileRepeatBlocks:
        def test_repeat_blocks_compiled_to_native_gates_correctly(
            self, circ, native_gate_set, expected_circ
        ):
            compiled_circ = compile_circuit_to_native_gates(circ, native_gate_set)
            assert compiled_circ == expected_circ

        def test_no_non_native_gates_left_after_compilation(
            self, circ, native_gate_set, expected_circ
        ):
            compiled_circ = compile_circuit_to_native_gates(circ, native_gate_set)
            for gate_layer in compiled_circ.gate_layers():
                for gate in gate_layer.gates:
                    assert type(gate) in native_gate_set.native_gates

    class TestNonSymmetric2QGates:
        @pytest.mark.parametrize(
            "circ, native_gate_set, expected_circ",
            [
                [
                    Circuit(
                        [GateLayer([H(0)]), GateLayer([CZ(0, 1)]), GateLayer([H(0)])]
                    ),
                    NativeGateSet(two_qubit_gates=set({CX})),
                    Circuit([GateLayer([CX(1, 0)])]),
                ],
                [
                    Circuit(
                        [
                            GateLayer([H(0)]),
                            GateLayer([CZSWAP(0, 1)]),
                            GateLayer([H(0)]),
                        ]
                    ),
                    NativeGateSet(two_qubit_gates=set({CXSWAP})),
                    Circuit([GateLayer(CXSWAP(1, 0)), GateLayer([H(1), H(0)])]),
                ],
                [
                    Circuit([GateLayer([CZ(1, 0)]), GateLayer([CZ(2, 0)])]),
                    NativeGateSet(two_qubit_gates=set({CX})),
                    Circuit(
                        [
                            GateLayer(H(0)),
                            GateLayer([CX(1, 0)]),
                            GateLayer([CX(2, 0)]),
                            GateLayer(H(0)),
                        ]
                    ),
                ],
                [
                    Circuit(
                        [
                            GateLayer(H(0)),
                            GateLayer([CZ(0, 1)]),
                            GateLayer([CZ(0, 2)]),
                            GateLayer(H(0)),
                        ]
                    ),
                    NativeGateSet(two_qubit_gates=set({CX})),
                    Circuit([GateLayer([CX(1, 0)]), GateLayer([CX(2, 0)])]),
                ],
                [  # this test gives suboptimal results, because the first entangling gate gives identical
                    # gate counts for both orientations, However, the compilation is better if it is reversed,
                    # due to the following gate. So the compilation can only guarantee optimal compilation if
                    # there is some more complex operation happening that relays information between gates.
                    Circuit([GateLayer([CZ(0, 1)]), GateLayer([CZ(0, 2)])]),
                    NativeGateSet(two_qubit_gates=set({CX})),
                    Circuit(
                        [
                            GateLayer(H(1)),
                            GateLayer([CX(0, 1)]),
                            GateLayer([H(1), H(2)]),
                            GateLayer([CX(0, 2)]),
                            GateLayer(H(2)),
                        ]
                    ),
                ],
                [
                    Circuit(
                        [
                            GateLayer([S_DAG(0), H(1)]),
                            GateLayer([H(0), S(1)]),
                            GateLayer([CY(0, 1)]),
                        ]
                    ),
                    NativeGateSet(
                        two_qubit_gates=set({CY}), one_qubit_gates=set({S_DAG, H, S})
                    ),
                    Circuit(
                        [
                            GateLayer([CY(1, 0)]),
                            GateLayer([S_DAG(0), H(1)]),
                            GateLayer([H(0), S(1)]),
                        ]
                    ),
                ],
            ],
        )
        def test_compiles_to_non_symmetric_2q_gates_optimally(
            self, circ, native_gate_set, expected_circ
        ):
            assert (
                compile_circuit_to_native_gates(circ, native_gate_set) == expected_circ
            )

        @pytest.mark.parametrize(
            "circ, native_gate_set, expected_circs",
            [
                [
                    Circuit(
                        [
                            GateLayer(SQRT_Y(0)),
                            GateLayer(X(0)),
                            GateLayer([CZ(0, 1)]),
                            GateLayer(SQRT_Y(0)),
                            GateLayer(X(0)),
                        ]
                    ),
                    NativeGateSet(
                        two_qubit_gates=set({CX}), one_qubit_gates=set({X, SQRT_Y})
                    ),
                    [Circuit([GateLayer([CX(1, 0)])])],
                ],
                [
                    Circuit(
                        [
                            GateLayer(Z(0)),
                            GateLayer(SQRT_Y(0)),
                            GateLayer([CZ(0, 1)]),
                            GateLayer(Z(0)),
                            GateLayer(SQRT_Y(0)),
                        ]
                    ),
                    NativeGateSet(
                        two_qubit_gates=set({CX}), one_qubit_gates=set({Z, SQRT_Y})
                    ),
                    [Circuit([GateLayer([CX(1, 0)])])],
                ],
                [
                    Circuit(
                        [
                            GateLayer(S(0)),
                            GateLayer(S(0)),
                            GateLayer([S(0), H(1)]),
                            GateLayer([H(0), S(1)]),
                            GateLayer([CY(0, 1)]),
                        ]
                    ),
                    NativeGateSet(
                        two_qubit_gates=set({CY}), one_qubit_gates=set({H, S})
                    ),
                    [
                        Circuit(
                            [
                                GateLayer([CY(1, 0)]),
                                GateLayer([S(0), H(1)]),
                                GateLayer([S(0), S(1)]),
                                GateLayer(S(0)),
                                GateLayer(H(0)),
                            ]
                        ),
                        Circuit(
                            [
                                GateLayer([CY(1, 0)]),
                                GateLayer([H(0), H(1)]),
                                GateLayer([S(0), S(1)]),
                                GateLayer(H(0)),
                                GateLayer(S(0)),
                            ]
                        ),
                    ],
                ],
            ],
        )
        def test_compiles_to_non_symmetric_2q_gates_even_if_non_native_gate_in_order_swapping_unitaries(
            self, circ, native_gate_set, expected_circs
        ):
            assert (
                compile_circuit_to_native_gates(circ, native_gate_set) in expected_circs
            )


class TestCompileOrExchangeUnitaryBlock:
    @pytest.mark.parametrize("up_to_paulis", [True, False])
    def test_returns_empty_for_empty_ub(self, up_to_paulis):
        assert _compile_or_exchange_unitary_block([], {}, up_to_paulis) == []

    @pytest.mark.parametrize(
        "ub, comp_dict, expected_ub",
        [
            [[X(0), X(0)], compilation_dict21, []],
            [[S(0), SQRT_X(0), S(0)], compilation_dict21, [H(0)]],
        ],
    )
    def test_compiles_ub_to_shorter_option_when_available(
        self, ub, comp_dict, expected_ub
    ):
        assert _compile_or_exchange_unitary_block(ub, comp_dict, False) == expected_ub

    @pytest.mark.parametrize(
        "ub, comp_dict, expected_ub",
        [
            [[X(0), X(0)], compilation_dict11_nosign, []],
            [[S(0), SQRT_X(0), S(0)], compilation_dict11_nosign, [H(0)]],
        ],
    )
    def test_compiles_ub_to_shorter_option_when_available_with_up_to_paulis_True(
        self, ub, comp_dict, expected_ub
    ):
        assert _compile_or_exchange_unitary_block(ub, comp_dict, True) == expected_ub


@pytest.fixture(scope="function")
def random_qubit():
    return Qubit(random.randint(0, 1000))


@pytest.fixture(scope="function")
def random_gate_layer():
    return random.randint(0, 1000)


class TestCompileResetsToNativeGatesPlusUnitaries:
    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize("gate", [RX, RY, RZ])
    def test_returns_empty_unitary_blocks_if_compiling_to_self(
        self, gate, up_to_paulis, random_qubit, random_gate_layer
    ):
        assert _compile_reset_to_native_gates_plus_unitaries(
            (random_gate_layer, random_qubit, gate.stim_string),
            [],
            [],
            NativeGateSet(reset_gates=set({gate})),
            compilation_dict23,
            up_to_paulis,
        )[0:2] == ([], [])

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize("gate", [RX, RY, RZ])
    def test_returns_target_gate_if_compiling_to_self(
        self, gate, up_to_paulis, random_qubit, random_gate_layer
    ):
        assert (
            _compile_reset_to_native_gates_plus_unitaries(
                (random_gate_layer, random_qubit, gate.stim_string),
                [],
                [],
                NativeGateSet(reset_gates=set({gate})),
                compilation_dict23,
                up_to_paulis,
            )[2]
            == gate.stim_string
        )

    @pytest.mark.parametrize("gate", [RX, RY, RZ])
    @pytest.mark.parametrize("target_gate", [RX, RY, RZ])
    class TestDictionaryDefinitions:
        def test_compiling_returns_unitary_blocks_from_dictionary(
            self, gate, target_gate, random_qubit, random_gate_layer
        ):
            expected_ubs = (
                [
                    g(random_qubit)
                    for g in RESET_COMPILATION_LOOKUP_DICT[gate][target_gate][0]
                ],
                [
                    g(random_qubit)
                    for g in RESET_COMPILATION_LOOKUP_DICT[gate][target_gate][1]
                ],
            )
            assert (
                _compile_reset_to_native_gates_plus_unitaries(
                    (random_gate_layer, random_qubit, gate.stim_string),
                    [],
                    [],
                    NativeGateSet(
                        one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                        reset_gates=set({target_gate}),
                    ),
                    compilation_dict23,
                    False,
                )[0:2]
                == expected_ubs
            )

        def test_dict_values_to_circuit_is_valid(self, gate, target_gate, random_qubit):
            r_to_m_gate = {
                RX: MX,
                RY: MY,
                RZ: MZ,
            }
            compiled_ubs = _compile_reset_to_native_gates_plus_unitaries(
                (random_gate_layer, random_qubit, gate.stim_string),
                [],
                [],
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                    reset_gates=set({target_gate}),
                ),
                compilation_dict23,
                False,
            )[0:2]
            # compile reset gate then measure out in the original basis
            circ = Circuit(
                [
                    *(GateLayer(gate) for gate in compiled_ubs[0]),
                    GateLayer(target_gate(random_qubit)),
                    *(GateLayer(gate) for gate in compiled_ubs[1]),
                    GateLayer(r_to_m_gate[gate](random_qubit)),
                ]
            )
            assert np.sum(circ.as_stim_circuit().compile_sampler().sample(100)) == 0

    @pytest.mark.parametrize(
        "current_gate, preceding_ub, succeeding_ub, target_gate, expected_unitary_blocks",
        [
            [
                RX,
                [X],
                [],
                RZ,
                ([X], [H]),
            ],
            [
                RX,
                [],
                [X],
                RZ,
                ([], [H, X]),
            ],
            [
                RX,
                [],
                [X],
                RZ,
                ([], [H, X]),
            ],
            [
                RX,
                [Y],
                [],
                RZ,
                ([Y], [H]),
            ],
            [
                RX,
                [],
                [S],
                RY,
                ([], []),
            ],
            [
                RY,
                [X],
                [],
                RZ,
                ([X], [H, S]),
            ],
            [
                RY,
                [],
                [S_DAG, H],
                RZ,
                ([], []),
            ],
            [
                RY,
                [S],
                [],
                RZ,
                ([S], [H, S]),
            ],
            [
                RZ,
                [],
                [H, S],
                RY,
                ([], []),
            ],
            [
                RZ,
                [X],
                [],
                RY,
                ([X], [S_DAG, H]),
            ],
            [
                RZ,
                [],
                [H, S],
                RY,
                ([], []),
            ],
        ],
    )
    def test_returns_correctly_compiled_unitary_blocks_with_unitaries_already_in_blocks(
        self,
        current_gate,
        preceding_ub,
        succeeding_ub,
        target_gate,
        expected_unitary_blocks,
        random_qubit,
        random_gate_layer,
    ):
        gate_info = (random_gate_layer, random_qubit, current_gate.stim_string)
        preceding_ub = [g(random_qubit) for g in preceding_ub]
        succeeding_ub = [g(random_qubit) for g in succeeding_ub]
        expected_unitary_blocks = (
            [g(random_qubit) for g in expected_unitary_blocks[0]],
            [g(random_qubit) for g in expected_unitary_blocks[1]],
        )
        assert (
            _compile_reset_to_native_gates_plus_unitaries(
                gate_info,
                preceding_ub,
                succeeding_ub,
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                    reset_gates=set({target_gate}),
                ),
                compilation_dict23,
                False,
            )[0:2]
            == expected_unitary_blocks
        )

    @pytest.mark.parametrize(
        "current_gate, preceding_ub, succeeding_ub, target_gate, expected_unitary_blocks",
        [
            [
                RX,
                [X],
                [],
                RZ,
                ([], [H]),
            ],
            [
                RX,
                [],
                [X],
                RZ,
                ([], [H]),
            ],
            [
                RX,
                [],
                [X],
                RZ,
                ([], [H]),
            ],
            [
                RX,
                [Y],
                [],
                RZ,
                ([], [H]),
            ],
            [
                RX,
                [],
                [S],
                RY,
                ([], []),
            ],
            [
                RY,
                [X],
                [],
                RZ,
                ([], [H, S]),
            ],
            [
                RY,
                [],
                [S_DAG, H],
                RZ,
                ([], []),
            ],
            [
                RY,
                [S],
                [],
                RZ,
                ([S], [H, S]),
            ],
            [
                RZ,
                [],
                [H, S],
                RY,
                ([], []),
            ],
            [
                RZ,
                [X],
                [],
                RY,
                ([], [S, H]),
            ],
            [
                RZ,
                [],
                [H, S],
                RY,
                ([], []),
            ],
        ],
    )
    def test_returns_correctly_compiled_unitary_blocks_with_unitaries_already_in_blocks_with_up_to_paulis_True(
        self,
        current_gate,
        preceding_ub,
        succeeding_ub,
        target_gate,
        expected_unitary_blocks,
        random_qubit,
        random_gate_layer,
    ):
        gate_info = (random_gate_layer, random_qubit, current_gate.stim_string)
        preceding_ub = [g(random_qubit) for g in preceding_ub]
        succeeding_ub = [g(random_qubit) for g in succeeding_ub]
        expected_unitary_blocks = (
            [g(random_qubit) for g in expected_unitary_blocks[0]],
            [g(random_qubit) for g in expected_unitary_blocks[1]],
        )
        assert (
            _compile_reset_to_native_gates_plus_unitaries(
                gate_info,
                preceding_ub,
                succeeding_ub,
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H, X, Y, Z, SQRT_X, SQRT_X_DAG}),
                    reset_gates=set({target_gate}),
                ),
                compilation_dict12_nosign,
                True,
            )[0:2]
            == expected_unitary_blocks
        )

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize("gate", [RX, RY, RZ])
    def test_raises_ValueError_when_compilation_not_possible(
        self, gate, up_to_paulis, random_qubit, random_gate_layer
    ):
        target_gates = set({RX, RY, RZ})
        target_gates.discard(gate)
        with pytest.raises(
            ValueError,
            match=r"Unable to compile to provided native reset and measurement gates, please try changing the native gate set.",
        ):
            _compile_reset_to_native_gates_plus_unitaries(
                (random_gate_layer, random_qubit, gate.stim_string),
                [],
                [],
                NativeGateSet(one_qubit_gates=set({}), reset_gates=target_gates),
                compilation_dict23,
                up_to_paulis,
            )

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize("gate", [RX, RY, RZ])
    def test_compiles_to_current_gate_when_current_gate_in_native_gates(
        self, gate, up_to_paulis, random_qubit, random_gate_layer
    ):
        assert _compile_reset_to_native_gates_plus_unitaries(
            (random_gate_layer, random_qubit, gate.stim_string),
            [],
            [],
            NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RX, RY, RZ})),
            compilation_dict23,
            up_to_paulis,
        ) == ([], [], gate.stim_string)


class TestCompileMeasurementsToNativeGatesPlusUnitaries:
    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize("gate", [MX, MY, MZ, MRX, MRY, MRZ])
    def test_returns_empty_unitary_blocks_if_compiling_to_self(
        self, gate, up_to_paulis, random_qubit, random_gate_layer
    ):
        assert _compile_measurement_to_native_gates_plus_unitaries(
            (random_gate_layer, random_qubit, gate.stim_string),
            [],
            [],
            NativeGateSet(measurement_gates=set({gate})),
            compilation_dict23,
            up_to_paulis,
        )[0:2] == ([], [])

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize("gate", [MX, MY, MZ, MRX, MRY, MRZ])
    def test_returns_target_gate_if_compiling_to_self(
        self, gate, up_to_paulis, random_qubit, random_gate_layer
    ):
        assert (
            _compile_measurement_to_native_gates_plus_unitaries(
                (random_gate_layer, random_qubit, gate.stim_string),
                [],
                [],
                NativeGateSet(measurement_gates=set({gate})),
                compilation_dict23,
                up_to_paulis,
            )[2]
            == gate.stim_string
        )

    @pytest.mark.parametrize("gate", [MX, MY, MZ])
    @pytest.mark.parametrize("target_gate", [MX, MY, MZ])
    class TestDictionaryValues:
        def test_compiling_returns_unitary_blocks_from_dictionary(
            self, gate, target_gate, random_qubit, random_gate_layer
        ):
            expected_ubs = (
                [
                    g(random_qubit)
                    for g in MEAS_COMPILATION_LOOKUP_DICT[gate][target_gate][0]
                ],
                [
                    g(random_qubit)
                    for g in MEAS_COMPILATION_LOOKUP_DICT[gate][target_gate][1]
                ],
            )
            assert (
                _compile_measurement_to_native_gates_plus_unitaries(
                    (random_gate_layer, random_qubit, gate.stim_string),
                    [],
                    [],
                    NativeGateSet(
                        one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                        measurement_gates=set({target_gate}),
                    ),
                    compilation_dict23,
                    False,
                )[0:2]
                == expected_ubs
            )

        def test_dict_values_to_circuit_is_valid(self, gate, target_gate, random_qubit):
            m_to_r_gate = {
                MX: RX,
                MY: RY,
                MZ: RZ,
            }
            compiled_ubs = _compile_measurement_to_native_gates_plus_unitaries(
                (random_gate_layer, random_qubit, gate.stim_string),
                [],
                [],
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                    measurement_gates=set({target_gate}),
                ),
                compilation_dict23,
                False,
            )[0:2]
            # reset in original basis, then compiled measurement, and then measure in original basis.
            circ = Circuit(
                [
                    GateLayer(m_to_r_gate[gate](random_qubit)),
                    *(GateLayer(gate) for gate in compiled_ubs[0]),
                    GateLayer(target_gate(random_qubit)),
                    *(GateLayer(gate) for gate in compiled_ubs[1]),
                    GateLayer(gate(random_qubit)),
                ]
            )
            assert np.sum(circ.as_stim_circuit().compile_sampler().sample(100)) == 0

    @pytest.mark.parametrize("gate", [MRX, MRY, MRZ])
    @pytest.mark.parametrize("target_gate", [MRX, MRY, MRZ])
    class TestDictionaryValuesMR:
        def test_compiling_returns_unitary_blocks_from_dictionary_mr(
            self, gate, target_gate, random_qubit, random_gate_layer
        ):
            expected_ubs = (
                [
                    g(random_qubit)
                    for g in MEAS_COMPILATION_LOOKUP_DICT[gate][target_gate][0]
                ],
                [
                    g(random_qubit)
                    for g in MEAS_COMPILATION_LOOKUP_DICT[gate][target_gate][1]
                ],
            )
            assert (
                _compile_measurement_to_native_gates_plus_unitaries(
                    (random_gate_layer, random_qubit, gate.stim_string),
                    [],
                    [],
                    NativeGateSet(
                        one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                        measurement_gates=set({target_gate}),
                    ),
                    compilation_dict23,
                    False,
                )[0:2]
                == expected_ubs
            )

        def test_dict_values_to_circuit_is_valid(self, gate, target_gate, random_qubit):
            mr_to_r_gate = {
                MRX: RX,
                MRY: RY,
                MRZ: RZ,
            }
            compiled_ubs = _compile_measurement_to_native_gates_plus_unitaries(
                (random_gate_layer, random_qubit, gate.stim_string),
                [],
                [],
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                    measurement_gates=set({target_gate}),
                ),
                compilation_dict23,
                False,
            )[0:2]
            # reset in original basis, then measure-reset as compiled, and finally measure-reset in original basis.
            circ = Circuit(
                [
                    GateLayer(mr_to_r_gate[gate](random_qubit)),
                    *(GateLayer(gate) for gate in compiled_ubs[0]),
                    GateLayer(target_gate(random_qubit)),
                    *(GateLayer(gate) for gate in compiled_ubs[1]),
                    GateLayer(gate(random_qubit)),
                ]
            )
            assert np.sum(circ.as_stim_circuit().compile_sampler().sample(100)) == 0

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize("gate", [MX, MY, MZ, MRX, MRY, MRZ])
    def test_compiles_to_current_gate_when_current_gate_in_native_gates(
        self, gate, up_to_paulis, random_qubit, random_gate_layer
    ):
        assert _compile_measurement_to_native_gates_plus_unitaries(
            (random_gate_layer, random_qubit, gate.stim_string),
            [],
            [],
            NativeGateSet(
                one_qubit_gates=set({}),
                measurement_gates=set({MX, MY, MZ, MRX, MRY, MRZ}),
            ),
            compilation_dict23,
            up_to_paulis,
        ) == ([], [], gate.stim_string)

    @pytest.mark.parametrize(
        "current_gate, preceding_ub, succeeding_ub, target_gate, expected_unitary_blocks",
        [
            [
                MX,
                [X],
                [],
                MZ,
                ([H, Z], [H]),
            ],
            [
                MX,
                [X],
                [Y],
                MZ,
                ([H, Z], [H, Y]),
            ],
            [
                MX,
                [S],
                [],
                MY,
                ([Z], [S_DAG]),
            ],
            [
                MX,
                [],
                [H],
                MZ,
                ([H], []),
            ],
            [
                MY,
                [S],
                [],
                MZ,
                ([H], [H, S]),
            ],
            [
                MY,
                [S_DAG],
                [],
                MZ,
                ([H, X], [H, S]),
            ],
            [
                MZ,
                [S_DAG],
                [],
                MY,
                ([S_DAG, H, S], [S_DAG, H]),
            ],
            [
                MRX,
                [X],
                [X],
                MRZ,
                ([H, Z], [H, X]),
            ],
            [
                MRX,
                [S],
                [],
                MRY,
                ([Z], [S_DAG]),
            ],
            [
                MRX,
                [S],
                [S],
                MRY,
                ([Z], []),
            ],
            [
                MRY,
                [S_DAG],
                [],
                MRX,
                ([Z], [S]),
            ],
            [
                MRZ,
                [],
                [X],
                MRZ,
                ([], [X]),
            ],
            [
                MRZ,
                [H],
                [SQRT_X, X],
                MRY,
                ([S], [H, X]),
            ],
            [
                MRZ,
                [S_DAG, Z],
                [S_DAG, Z],
                MRX,
                ([S, H], [H, S]),
            ],
        ],
    )
    def test_returns_correctly_compiled_unitary_blocks(
        self,
        current_gate,
        preceding_ub,
        succeeding_ub,
        target_gate,
        expected_unitary_blocks,
        random_qubit,
        random_gate_layer,
    ):
        gate_info = (random_gate_layer, random_qubit, current_gate.stim_string)
        preceding_ub = [g(random_qubit) for g in preceding_ub]
        succeeding_ub = [g(random_qubit) for g in succeeding_ub]
        expected_unitary_blocks = (
            [g(random_qubit) for g in expected_unitary_blocks[0]],
            [g(random_qubit) for g in expected_unitary_blocks[1]],
        )
        assert (
            _compile_measurement_to_native_gates_plus_unitaries(
                gate_info,
                preceding_ub,
                succeeding_ub,
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H, X, Y, Z}),
                    measurement_gates=set({target_gate}),
                ),
                compilation_dict23,
                False,
            )[0:2]
            == expected_unitary_blocks
        )

    @pytest.mark.parametrize(
        "current_gate, preceding_ub, succeeding_ub, target_gate, expected_unitary_blocks",
        [
            [
                MX,
                [X],
                [],
                MZ,
                ([H], [H]),
            ],
            [
                MX,
                [X],
                [Y],
                MZ,
                ([H], [H]),
            ],
            [
                MX,
                [S],
                [],
                MY,
                ([], [S]),
            ],
            [
                MX,
                [],
                [H],
                MZ,
                ([H], []),
            ],
            [
                MY,
                [S_DAG],
                [],
                MZ,
                ([H], [H, S]),
            ],
            [
                MY,
                [S],
                [],
                MZ,
                ([H], [H, S]),
            ],
            [
                MZ,
                [S],
                [],
                MY,
                ([SQRT_X], [S, H]),
            ],
            [
                MRX,
                [X],
                [X],
                MRZ,
                ([H], [H]),
            ],
            [
                MRX,
                [S],
                [],
                MRY,
                ([], [S]),
            ],
            [
                MRX,
                [S],
                [S],
                MRY,
                ([], []),
            ],
            [
                MRY,
                [S_DAG],
                [],
                MRX,
                ([], [S]),
            ],
            [
                MRZ,
                [],
                [X],
                MRZ,
                ([], []),
            ],
            [
                MRZ,
                [H],
                [SQRT_X, X],
                MRY,
                ([S], [H]),
            ],
            [
                MRZ,
                [S_DAG, Z],
                [S_DAG, Z],
                MRX,
                ([S, H], [H, S]),
            ],
        ],
    )
    def test_returns_correctly_compiled_unitary_blocks_with_up_to_paulis_True(
        self,
        current_gate,
        preceding_ub,
        succeeding_ub,
        target_gate,
        expected_unitary_blocks,
        random_qubit,
        random_gate_layer,
    ):
        gate_info = (random_gate_layer, random_qubit, current_gate.stim_string)
        preceding_ub = [g(random_qubit) for g in preceding_ub]
        succeeding_ub = [g(random_qubit) for g in succeeding_ub]
        expected_unitary_blocks = (
            [g(random_qubit) for g in expected_unitary_blocks[0]],
            [g(random_qubit) for g in expected_unitary_blocks[1]],
        )
        assert (
            _compile_measurement_to_native_gates_plus_unitaries(
                gate_info,
                preceding_ub,
                succeeding_ub,
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H, X, Y, Z, SQRT_X, SQRT_X_DAG}),
                    measurement_gates=set({target_gate}),
                ),
                compilation_dict12_nosign,
                True,
            )[0:2]
            == expected_unitary_blocks
        )


class TestCompileResetAndMeasToNativeGates:
    def test_returns_empty_dicts_if_reset_and_meas_dicts_empty(self):
        assert _compile_reset_and_meas_to_native_gates(
            CompilationData({}, {}, {}, {}, {}), NativeGateSet(), {}, False, {}
        )[0] == CompilationData({}, {}, {}, {}, {})

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize(
        "comp_data, native_gate_set, comp_dict",
        [
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ}), one_qubit_gates=set({})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RY}), one_qubit_gates=set({})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RX}), one_qubit_gates=set({})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RX}), one_qubit_gates=set({X, Z})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RY}), one_qubit_gates=set({H})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MY}), one_qubit_gates=set({})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MX}), one_qubit_gates=set({})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}),
                    reset_gates=set({RX}),
                    one_qubit_gates=set({}),
                ),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MY}),
                    reset_gates=set({RY}),
                    one_qubit_gates=set({}),
                ),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MX}),
                    reset_gates=set({RX}),
                    one_qubit_gates=set({}),
                ),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRX}), one_qubit_gates=set({})),
                compilation_dict0,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ}), one_qubit_gates=set({S})),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RY}), one_qubit_gates=set({H})),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RX}), one_qubit_gates=set({Z, X})),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RX}), one_qubit_gates=set({X, Z})),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RY}), one_qubit_gates=set({H})),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}), one_qubit_gates=set({S, SQRT_X})
                ),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MY}), one_qubit_gates=set({H, X})),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MX}),
                    one_qubit_gates=set({SQRT_X, SQRT_X_DAG}),
                ),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}),
                    reset_gates=set({RX}),
                    one_qubit_gates=set({X, Z, S}),
                ),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MY}),
                    reset_gates=set({RY}),
                    one_qubit_gates=set({Y, H}),
                ),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MX}),
                    reset_gates=set({RX}),
                    one_qubit_gates=set({S_DAG, SQRT_X}),
                ),
                compilation_dict21,
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRX}), one_qubit_gates=set({S})),
                compilation_dict21,
            ],
        ],
    )
    def test_throws_ValueError_when_compilation_not_possible_because_of_non_native_unitaries(
        self, comp_data, native_gate_set, comp_dict, up_to_paulis
    ):
        with pytest.raises(
            ValueError,
            match=r"Unable to compile to provided native reset and measurement gates, please try changing the native gate set.",
        ):
            _compile_reset_and_meas_to_native_gates(
                comp_data,
                native_gate_set,
                comp_dict,
                up_to_paulis,
                {0: 0},
            )

    @pytest.mark.parametrize(
        "comp_data, native_gate_set, comp_dict, expected_comp_data",
        [
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RZ})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RX})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [H(0)]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H, S_DAG}), reset_gates=set({RY})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [S_DAG(0), H(0)]},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RZ})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [H(0)]},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RX})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({S_DAG}), reset_gates=set({RY})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [S_DAG(0)]},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H, S}), reset_gates=set({RZ})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [H(0), S(0)]},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({S}), reset_gates=set({RX})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [S(0)]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RY})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MX}), one_qubit_gates=set({})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({H})),
                compilation_dict21,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}), one_qubit_gates=set({H, X, Z})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [H(0), Z(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: [Y(0)]},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}), one_qubit_gates=set({H, X, Y, Z})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [H(0), Z(0)], 1: [H(0), Y(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: [Y(0)]},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}), one_qubit_gates=set({H, Z, Y})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [H(0), Z(0)], 1: [H(0), Y(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [S(0)], 1: [S_DAG(0)]},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [S_DAG(0)], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [S_DAG(0)]},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(1), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({H})),
                compilation_dict21,
                CompilationData(
                    {0: [H(1)], 1: [H(1)]},
                    {},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [H(1)]},
                    {},
                    {(0, Qubit(1), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({H})),
                compilation_dict21,
                CompilationData(
                    {0: [H(1)], 1: []},
                    {},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(3, Qubit(2), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [S(2)], 1: [S_DAG(2)]},
                    {},
                    {(3, Qubit(2), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MY}), one_qubit_gates=set({})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRX}), one_qubit_gates=set({})),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRZ}), one_qubit_gates=set({H})),
                compilation_dict21,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: [X(0)]},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MRZ}), one_qubit_gates=set({H, X, Z})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [H(0), Z(0)], 1: [H(0), X(0)]},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MRY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [S(0)], 1: [S_DAG(0)]},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [S_DAG(0)], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MRY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [S_DAG(0)]},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [S_DAG(0)], 1: [S(0)]},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MRY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRZ}), one_qubit_gates=set({H})),
                compilation_dict21,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MRY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict21,
                CompilationData(
                    {0: [S(0)], 1: [S_DAG(0)]},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [H(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [  # 31
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [H(0)], 1: [H(0)], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [H(0)], 1: [H(0)], 2: [H(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MX}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [H(0)], 1: [H(0)], 2: [H(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [X(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, X}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [H(0), X(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [X(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, Z, X}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [H(0), X(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [H(0), X(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({X}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [X(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [H(1), X(1)]},
                    {(1, Qubit(1), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({X}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [X(1)]},
                    {(1, Qubit(1), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [H(1), X(1)]},
                    {(1, Qubit(1), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, X}),
                    reset_gates=set({RX, RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [H(1), X(1)]},
                    {(1, Qubit(1), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                    {
                        (1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(1), "RX"): {"preceding": 2, "succeeding": 3},
                        (1, Qubit(2), "RX"): {"preceding": 4, "succeeding": 5},
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, X}),
                    reset_gates=set({RX, RZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                    {
                        (1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(1), "RX"): {"preceding": 2, "succeeding": 3},
                        (1, Qubit(2), "RX"): {"preceding": 4, "succeeding": 5},
                    },
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                    {
                        (1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(1), "RZ"): {"preceding": 2, "succeeding": 3},
                        (1, Qubit(2), "RX"): {"preceding": 4, "succeeding": 5},
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, X}),
                    reset_gates=set({RX, RZ}),
                ),
                compilation_dict21,
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                    {
                        (1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(1), "RZ"): {"preceding": 2, "succeeding": 3},
                        (1, Qubit(2), "RX"): {"preceding": 4, "succeeding": 5},
                    },
                    {},
                    {},
                    {},
                ),
            ],
        ],
    )
    def test_compiles_resets_and_meas_to_native_gates_if_possible(
        self,
        comp_data,
        native_gate_set,
        comp_dict,
        expected_comp_data,
    ):
        assert (
            _compile_reset_and_meas_to_native_gates(
                comp_data,
                native_gate_set,
                comp_dict,
                False,
                {0: 0, 1: 1, 2: 2},
            )[0]
            == expected_comp_data
        )

    @pytest.mark.parametrize(
        "comp_data, native_gate_set, comp_dict, expected_comp_data",
        [
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RZ})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RX})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [H(0)]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H, S_DAG}), reset_gates=set({RY})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [S_DAG(0), H(0)]},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RZ})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [H(0)]},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RX})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({S}), reset_gates=set({RY})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [S(0)]},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H, S}), reset_gates=set({RZ})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [H(0), S(0)]},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({S}), reset_gates=set({RX})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [S(0)]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RY})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MX}), one_qubit_gates=set({})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({H})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({H, X})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: [Y(0)]},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}), one_qubit_gates=set({H, X, Y})
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: [Y(0)]},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}), one_qubit_gates=set({H, Z, Y})
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MY}), one_qubit_gates=set({S})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [S(0)], 1: [S(0)]},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [S(0)], 1: []},
                    {},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MY}), one_qubit_gates=set({S_DAG, S})
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [S(0)]},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(1), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({H})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(1)], 1: [H(1)]},
                    {},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [H(1)]},
                    {},
                    {(0, Qubit(1), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), one_qubit_gates=set({H})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(1)], 1: []},
                    {},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(3, Qubit(2), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MY}), one_qubit_gates=set({S})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [S(2)], 1: [S(2)]},
                    {},
                    {(3, Qubit(2), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MY}), one_qubit_gates=set({})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRX}), one_qubit_gates=set({})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRZ}), one_qubit_gates=set({H})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [X(0)], 1: [X(0)]},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MRZ}), one_qubit_gates=set({H, X})
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRY}), one_qubit_gates=set({S})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [S(0)], 1: [S(0)]},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [S(0)], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRY}), one_qubit_gates=set({S})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [S(0)]},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [S(0)], 1: [S(0)]},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MRY}), one_qubit_gates=set({S_DAG})
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRZ}), one_qubit_gates=set({H})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)]},
                    {},
                    {(0, Qubit(0), "MRZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MRX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MRY}), one_qubit_gates=set({S})),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [S(0)], 1: [S(0)]},
                    {},
                    {(0, Qubit(0), "MRY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [], 2: [H(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)], 2: [H(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MX}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [H(0)], 1: [H(0)], 2: [H(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [X(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, X}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [], 2: [H(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [X(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, Z}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [], 2: [H(0)]},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [H(0), X(0)]},
                    {(1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({X}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [H(1), X(1)]},
                    {(1, Qubit(1), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({X}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict11_nosign,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(1, Qubit(1), "RZ"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [H(1), X(1)]},
                    {(1, Qubit(1), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H, X}),
                    reset_gates=set({RX, RZ}),
                    measurement_gates=set({MZ}),
                ),
                compilation_dict12_nosign,
                CompilationData(
                    # since measurement gate is part of native,
                    # its unitary blocks wont be touched until later.
                    {0: [], 1: [], 2: [H(1), X(1)]},
                    {(1, Qubit(1), "RX"): {"preceding": 1, "succeeding": 2}},
                    {(0, Qubit(1), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
            ],
        ],
    )
    def test_compiles_resets_and_meas_to_native_gates_if_possible_with_up_to_paulis_True(
        self,
        comp_data,
        native_gate_set,
        comp_dict,
        expected_comp_data,
    ):
        assert (
            _compile_reset_and_meas_to_native_gates(
                comp_data,
                native_gate_set,
                comp_dict,
                True,
                {0: 0, 1: 1, 2: 2},
            )[0]
            == expected_comp_data
        )

    @pytest.mark.parametrize(
        "comp_data, native_gate_set, comp_dict, expected_comp_data",
        [
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {
                        (0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({}), reset_gates=set({RZ})),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {
                        (0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {
                        (0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RX})),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [H(0)], 2: []},
                    {
                        (0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {
                        (0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RX})),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [], 2: [H(0)]},
                    {
                        (0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {
                        (0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RZ"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RX})),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [H(0)], 2: [H(0)]},
                    {
                        (0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RX"): {"preceding": 1, "succeeding": 2},
                    },
                    {},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RX})),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [H(0)], 2: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({H}), reset_gates=set({RX})),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MX}),
                ),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [H(0)], 2: [H(0)]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(0), "MX"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(2, Qubit(0), "MZ"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MX}),
                ),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [H(0)], 2: [H(0)]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(2, Qubit(0), "MX"): {"preceding": 1, "succeeding": 2}},
                    {},
                    {},
                ),
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(2, Qubit(1), "MZ"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({H}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MX}),
                ),
                compilation_dict23,
                CompilationData(
                    {0: [], 1: [], 2: [H(1)], 3: [H(1)]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(2, Qubit(1), "MX"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
            ],
        ],
    )
    def test_compiles_resets_and_meas_to_native_gates_with_more_complicated_structure(
        self,
        comp_data,
        native_gate_set,
        comp_dict,
        expected_comp_data,
    ):
        assert (
            _compile_reset_and_meas_to_native_gates(
                comp_data,
                native_gate_set,
                comp_dict,
                False,
                {0: 0, 1: 1, 2: 2},
            )[0]
            == expected_comp_data
        )

    @pytest.mark.parametrize("up_to_paulis", [False, True])
    @pytest.mark.parametrize(
        "comp_data, native_gate_set, layer_ind_lookup, expected_layer_ind_lookup",
        [
            [
                (
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RX})),
                {0: 0},
                {0: 0},
            ],
            [
                (
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ})),
                {0: 0},
                {0: 0},
            ],
            [
                (
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ})),
                {0: 0, 1: 1},
                {0: 0, 1: 2},
            ],
            [
                (
                    {0: [], 1: []},
                    {(1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ})),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 1, 2: 3},
            ],
            [
                (
                    {0: [], 1: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ})),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 2, 2: 3},
            ],
            [
                (
                    {0: [], 1: []},
                    {(0, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ})),
                {0: 0, 1: 1},
                {0: 0, 1: 3},
            ],
            [
                (
                    {0: [], 1: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(one_qubit_gates=set({S_DAG, H}), reset_gates=set({RY})),
                {0: 0, 1: 1},
                {0: 0, 1: 3},
            ],
            [
                (
                    {0: [], 1: [H]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ})),
                {0: 0, 1: 1},
                {0: 0, 1: 1},
            ],
            [
                (
                    {0: [], 1: [H]},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                    {},
                ),
                NativeGateSet(reset_gates=set({RZ})),
                {0: 0, 2: 2},
                {0: 0, 2: 2},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ})),
                {0: 0},
                {0: 0},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(3, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ})),
                {3: 3},
                {3: 3},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MX})),
                {0: 0},
                {0: 1},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MX})),
                {0: 0, 1: 1},
                {0: 1, 1: 3},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S_DAG, S, H}), measurement_gates=set({MY})
                ),
                {0: 0},
                {0: 2},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H}), measurement_gates=set({MZ})
                ),
                {0: 0},
                {0: 2},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H}), measurement_gates=set({MY})
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 2, 1: 5, 2: 6},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(0, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H}), measurement_gates=set({MZ})
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 2, 1: 5, 2: 6},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(2, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S_DAG, S, H}), measurement_gates=set({MY})
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 1, 2: 4},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(2, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H}), measurement_gates=set({MZ})
                ),
                {0: 0, 1: 1, 2: 2, 3: 3},
                {0: 0, 1: 1, 2: 4, 3: 7},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(2, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S_DAG, S, H}), measurement_gates=set({MY})
                ),
                {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                {0: 0, 1: 1, 2: 4, 3: 7, 4: 8},
            ],
            [
                (
                    {0: [], 1: []},
                    {},
                    {(2, Qubit(0), "MY"): {"preceding": 0, "succeeding": 1}},
                    {},
                    {},
                ),
                NativeGateSet(
                    one_qubit_gates=set({S, S_DAG, H}), measurement_gates=set({MZ})
                ),
                {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                {0: 0, 1: 1, 2: 4, 3: 7, 4: 8},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(1), "MZ"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ})),
                {0: 0},
                {0: 0},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(1), "MZ"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), reset_gates=set({RZ})),
                {0: 0},
                {0: 0},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(0, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(1), "MZ"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), reset_gates=set({RZ})),
                {0: 0, 1: 1},
                {0: 0, 1: 2},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(0, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(0, Qubit(1), "MX"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), reset_gates=set({RZ})),
                {0: 0, 1: 1},
                {0: 1, 1: 3},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(1), "MX"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), reset_gates=set({RZ})),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 2, 2: 4},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(1), "MX"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(measurement_gates=set({MZ}), reset_gates=set({RZ})),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 2, 2: 5},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(1, Qubit(0), "RZ"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(1), "MZ"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MY}),
                    reset_gates=set({RY}),
                    one_qubit_gates=set({S, S_DAG, H}),
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 3, 2: 8},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: []},
                    {(1, Qubit(0), "RY"): {"preceding": 0, "succeeding": 1}},
                    {(1, Qubit(1), "MY"): {"preceding": 2, "succeeding": 3}},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}),
                    reset_gates=set({RZ}),
                    one_qubit_gates=set({S, S_DAG, H}),
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 3, 2: 8},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                    {
                        (1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RX"): {"preceding": 2, "succeeding": 3},  # noqa: F601
                        (1, Qubit(0), "RX"): {"preceding": 4, "succeeding": 5},  # noqa: F601
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}),
                    reset_gates=set({RX, RZ}),
                    one_qubit_gates=set({S, S_DAG, H}),
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 1, 2: 2},
            ],
            [
                (
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                    {
                        (1, Qubit(0), "RX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(0), "RZ"): {"preceding": 2, "succeeding": 3},
                        (1, Qubit(0), "RX"): {"preceding": 4, "succeeding": 5},  # noqa: F601
                    },
                    {},
                    {},
                    {},
                ),
                NativeGateSet(
                    measurement_gates=set({MZ}),
                    reset_gates=set({RX, RZ}),
                    one_qubit_gates=set({S, S_DAG, H}),
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 1, 2: 2},
            ],
        ],
    )
    def test_shifts_following_layers_to_accommodate_new_unitaries(
        self,
        comp_data,
        native_gate_set,
        layer_ind_lookup,
        expected_layer_ind_lookup,
        up_to_paulis,
    ):
        # when parametrizing over `up_to_paulis`, the dictionaries in `comp_data`
        # and `layer_ind_lookup` are not re-initialised.
        # `_compile_reset_and_meas_to_native_gates` mutates them, and this would
        # cause problems.
        comp_data = deepcopy(comp_data)
        layer_ind_lookup = deepcopy(layer_ind_lookup)
        comp_data = CompilationData(*comp_data)
        assert (
            _compile_reset_and_meas_to_native_gates(
                comp_data,
                native_gate_set,
                compilation_dict23 if not up_to_paulis else compilation_dict11_nosign,
                up_to_paulis,
                layer_ind_lookup,
            )[1]
            == expected_layer_ind_lookup
        )


class TestTwoQubitGateCompilationDicts:
    @pytest.mark.parametrize("gate", list(GATE_TO_CZ_DICT.keys()))
    def test_two_qubit_gate_to_cz_dict_compilations_give_equivalent_tableaus(
        self, gate
    ):
        ub1, ub2, ub3, ub4 = GATE_TO_CZ_DICT[gate]
        ub1_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub1[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub2_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub2[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub3_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub3[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub4_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub4[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        cz_with_unitaries_tableau = (
            (ub2_tableau + ub4_tableau)
            * stim.Tableau.from_named_gate("CZ")
            * (ub1_tableau + ub3_tableau)
        )
        assert (
            stim.Tableau.from_named_gate(gate.stim_string)
            == cz_with_unitaries_tableau
        )

    @pytest.mark.parametrize("gate", list(CZ_TO_GATE_DICT.keys()))
    def test_cz_to_two_qubit_gate_dict_compilations_give_equivalent_tableaus(
        self, gate
    ):
        ub1, ub2, ub3, ub4 = CZ_TO_GATE_DICT[gate]
        ub1_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub1[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub2_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub2[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub3_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub3[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub4_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub4[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        cz_with_unitaries_tableau = (
            (ub2_tableau + ub4_tableau)
            * stim.Tableau.from_named_gate(gate.stim_string)
            * (ub1_tableau + ub3_tableau)
        )
        assert stim.Tableau.from_named_gate("CZ") == cz_with_unitaries_tableau

    @pytest.mark.parametrize("gate", list(GATE_TO_CZSWAP_DICT.keys()))
    def test_cpswap_to_czswap_dict_compilations_give_equivalent_tableaus(self, gate):
        ub1, ub2, ub3, ub4 = GATE_TO_CZSWAP_DICT[gate]
        ub1_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub1[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub2_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub2[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub3_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub3[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub4_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub4[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        czswap_with_unitaries_tableau = (
            (ub2_tableau + ub4_tableau)
            * stim.Tableau.from_named_gate("CZSWAP")
            * (ub1_tableau + ub3_tableau)
        )
        assert (
            stim.Tableau.from_named_gate(gate.stim_string)
            == czswap_with_unitaries_tableau
        )

    @pytest.mark.parametrize("gate", list(CZSWAP_TO_GATE_DICT.keys()))
    def test_czswap_to_cpswap_dict_compilations_give_equivalent_tableaus(self, gate):
        ub1, ub2, ub3, ub4 = CZSWAP_TO_GATE_DICT[gate]
        ub1_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub1[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub2_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub2[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub3_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub3[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        ub4_tableau = reduce(
            mul,
            (stim.Tableau.from_named_gate(g.stim_string) for g in ub4[::-1]),
            stim.Tableau.from_named_gate("I"),
        )
        czswap_with_unitaries_tableau = (
            (ub2_tableau + ub4_tableau)
            * stim.Tableau.from_named_gate(gate.stim_string)
            * (ub1_tableau + ub3_tableau)
        )
        assert stim.Tableau.from_named_gate("CZSWAP") == czswap_with_unitaries_tableau


class TestCompileTwoQubitGateToTarget:
    @pytest.mark.parametrize(
        "gate",
        [
            CX,
            CY,
            CZ,
            SQRT_XX,
            SQRT_XX_DAG,
            SQRT_YY,
            SQRT_YY_DAG,
            SQRT_ZZ,
            SQRT_ZZ_DAG,
            XCX,
            XCY,
            XCZ,
            YCX,
            YCY,
            YCZ,
        ],
    )
    def test_compiling_to_cz_gives_unitaries_as_in_dictionary(self, gate):
        assert (
            tuple(
                [type(g) for g in ub]
                for ub in _compile_two_qubit_gate_to_target(
                    gate,
                    CZ,
                    (
                        (0, Qubit(0), gate.stim_string),
                        {"preceding": 0, "succeeding": 1},
                    ),
                    (
                        (0, Qubit(1), gate.stim_string),
                        {"preceding": 0, "succeeding": 1},
                    ),
                    GATE_TO_CZ_DICT,
                    CZ_TO_GATE_DICT,
                    {0: [], 1: [], 2: [], 3: []},
                )
            )
            == GATE_TO_CZ_DICT[gate]
        )

    @pytest.mark.parametrize(
        "gate",
        [CZSWAP, CXSWAP, ISWAP, ISWAP_DAG],
    )
    def test_compiling_to_czswap_gives_unitaries_as_in_dictionary(self, gate):
        assert (
            tuple(
                [type(g) for g in ub]
                for ub in _compile_two_qubit_gate_to_target(
                    gate,
                    CZSWAP,
                    (
                        (0, Qubit(0), gate.stim_string),
                        {"preceding": 0, "succeeding": 1},
                    ),
                    (
                        (0, Qubit(1), gate.stim_string),
                        {"preceding": 0, "succeeding": 1},
                    ),
                    GATE_TO_CZSWAP_DICT,
                    CZSWAP_TO_GATE_DICT,
                    {0: [], 1: [], 2: [], 3: []},
                )
            )
            == GATE_TO_CZSWAP_DICT[gate]
        )

    @pytest.mark.parametrize(
        "gate",
        [
            CX,
            CY,
            CZ,
            SQRT_XX,
            SQRT_XX_DAG,
            SQRT_YY,
            SQRT_YY_DAG,
            SQRT_ZZ,
            SQRT_ZZ_DAG,
            XCX,
            XCY,
            XCZ,
            YCX,
            YCY,
            YCZ,
        ],
    )
    def test_compiling_from_cz_gives_unitaries_as_in_dictionary(self, gate):
        assert (
            tuple(
                [type(g) for g in ub]
                for ub in _compile_two_qubit_gate_to_target(
                    CZ,
                    gate,
                    ((0, Qubit(0), "CX"), {"preceding": 0, "succeeding": 1}),
                    ((0, Qubit(1), "CX"), {"preceding": 0, "succeeding": 1}),
                    GATE_TO_CZ_DICT,
                    CZ_TO_GATE_DICT,
                    {0: [], 1: [], 2: [], 3: []},
                )
            )
            == CZ_TO_GATE_DICT[gate]
        )

    @pytest.mark.parametrize(
        "gate",
        [
            CZSWAP,
            CXSWAP,
            ISWAP,
            ISWAP_DAG,
        ],
    )
    def test_compiling_from_czswap_gives_unitaries_as_in_dictionary(self, gate):
        assert (
            tuple(
                [type(g) for g in ub]
                for ub in _compile_two_qubit_gate_to_target(
                    CZSWAP,
                    gate,
                    ((0, Qubit(0), "CX"), {"preceding": 0, "succeeding": 1}),
                    ((0, Qubit(1), "CX"), {"preceding": 0, "succeeding": 1}),
                    GATE_TO_CZSWAP_DICT,
                    CZSWAP_TO_GATE_DICT,
                    {0: [], 1: [], 2: [], 3: []},
                )
            )
            == CZSWAP_TO_GATE_DICT[gate]
        )

    @pytest.mark.parametrize(
        "gate, target_gate, expected_unitaries",
        [
            [CY, CX, ([], [], [S_DAG(1), H(1), H(1)], [H(1), H(1), S(1)])],
            [
                CX,
                CY,
                (
                    [],
                    [],
                    [H(1), H(1), S(1)],
                    [S_DAG(1), H(1), H(1)],
                ),
            ],
            [
                CX,
                SQRT_XX,
                ([H(0)], [H(0), S_DAG(0)], [H(1), H(1)], [H(1), S_DAG(1), H(1)]),
            ],
            [SQRT_XX, CX, ([H(0)], [S(0), H(0)], [H(1), H(1)], [H(1), S(1), H(1)])],
            [
                CX,
                SQRT_YY_DAG,
                (
                    [H(0), S(0)],
                    [S_DAG(0), H(0), S_DAG(0)],
                    [H(1), H(1), S_DAG(1)],
                    [S(1), H(1), S_DAG(1), H(1)],
                ),
            ],
            [
                SQRT_YY_DAG,
                CX,
                (
                    [S_DAG(0), H(0)],
                    [S(0), H(0), S(0)],
                    [S(1), H(1), H(1)],
                    [H(1), S(1), H(1), S_DAG(1)],
                ),
            ],
        ],
    )
    def test_compiling_between_gates_stacks_unitaries_in_correct_way(
        self, gate, target_gate, expected_unitaries
    ):
        assert (
            _compile_two_qubit_gate_to_target(
                gate,
                target_gate,
                ((0, Qubit(0), gate.stim_string), {"preceding": 0, "succeeding": 1}),
                ((0, Qubit(1), gate.stim_string), {"preceding": 0, "succeeding": 1}),
                GATE_TO_CZ_DICT,
                CZ_TO_GATE_DICT,
                {0: [], 1: [], 2: [], 3: []},
            )
            == expected_unitaries
        )

    @pytest.mark.parametrize(
        "gate, target_gate, expected_unitaries",
        [
            [ISWAP, CZSWAP, ([], [S(0)], [], [S(1)])],
            [
                ISWAP,
                ISWAP_DAG,
                (
                    [S(0)],
                    [S(0)],
                    [S(1)],
                    [S(1)],
                ),
            ],
            [
                ISWAP_DAG,
                CXSWAP,
                ([S_DAG(0)], [H(0)], [S_DAG(1), H(1)], []),
            ],
            [
                ISWAP_DAG,
                CZSWAP,
                ([S_DAG(0)], [], [S_DAG(1)], []),
            ],
        ],
    )
    def test_compiling_between_gates_stacks_unitaries_in_correct_way_iswaplike(
        self, gate, target_gate, expected_unitaries
    ):
        assert (
            _compile_two_qubit_gate_to_target(
                gate,
                target_gate,
                ((0, Qubit(0), gate.stim_string), {"preceding": 0, "succeeding": 1}),
                ((0, Qubit(1), gate.stim_string), {"preceding": 0, "succeeding": 1}),
                GATE_TO_CZSWAP_DICT,
                CZSWAP_TO_GATE_DICT,
                {0: [], 1: [], 2: [], 3: []},
            )
            == expected_unitaries
        )

    @pytest.mark.parametrize(
        "gate, target_gate, gate_to_interm_rep_dict",
        [
            [CX, CZ, GATE_TO_CZSWAP_DICT],
            [CZ, CX, GATE_TO_CZSWAP_DICT],
            [CXSWAP, CZSWAP, GATE_TO_CZ_DICT],
            [CZSWAP, CXSWAP, GATE_TO_CZ_DICT],
        ],
    )
    def test_specifying_wrong_dictionary_gives_ValueError(
        self, gate, target_gate, gate_to_interm_rep_dict
    ):
        with pytest.raises(
            ValueError,
            match=r"Current gate not present in gate_to_intermediate_rep_dict dictionary",
        ):
            _compile_two_qubit_gate_to_target(
                gate,
                target_gate,
                ((0, Qubit(0), gate.stim_string), {"preceding": 0, "succeeding": 1}),
                ((0, Qubit(1), gate.stim_string), {"preceding": 2, "succeeding": 3}),
                gate_to_interm_rep_dict,
                CZ_TO_GATE_DICT,
                {0: [], 1: [], 2: [], 3: []},
            )


class TestCompileTwoQubitGatesToNativeGates:
    @pytest.mark.parametrize("up_to_paulis", [True, False])
    def test_empty_two_qubit_dict_gives_empty(self, up_to_paulis):
        assert _compile_two_qubit_gates_to_native_gates(
            CompilationData({}, {}, {}, {}, {}),
            NativeGateSet(),
            {},
            up_to_paulis,
            {},
        )[0] == CompilationData({}, {}, {}, {}, {})

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    def test_empty_native_two_qubit_gates_throws_ValueError(self, up_to_paulis):
        with pytest.raises(
            ValueError,
            match=r"Unable to compile to provided native two-qubit gates, please try changing the native gate set.",
        ):
            _compile_two_qubit_gates_to_native_gates(
                CompilationData(
                    {},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(two_qubit_gates=set({})),
                {},
                up_to_paulis,
                {0: 0},
            )

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize(
        "gate",
        [
            CY,
            CZ,
            SQRT_XX,
            SQRT_XX_DAG,
            SQRT_YY,
            SQRT_YY_DAG,
            SQRT_ZZ,
            SQRT_ZZ_DAG,
            XCX,
            XCY,
            YCX,
            YCY,
            YCZ,
        ],
    )
    def test_throws_ValueError_when_compilation_not_possible_for_non_iswaplike_2q_gates_due_to_lack_of_single_qubit_gates(
        self, gate, up_to_paulis
    ):
        comp_data = CompilationData(
            {0: [], 1: [], 2: [], 3: []},
            {},
            {},
            {
                (0, Qubit(0), gate.stim_string): {"preceding": 0, "succeeding": 1},
                (0, Qubit(1), gate.stim_string): {"preceding": 2, "succeeding": 3},
            },
            {},
        )
        native_gate_set = NativeGateSet(
            one_qubit_gates=set({}),
            two_qubit_gates=set({CX}),
        )
        with pytest.raises(
            ValueError,
            match=r"Unable to compile to provided native two-qubit gates, please try changing the native gate set.",
        ):
            _compile_two_qubit_gates_to_native_gates(
                comp_data,
                native_gate_set,
                compilation_dict11_nosign if up_to_paulis else compilation_dict21,
                up_to_paulis,
                {0: 0},
            )

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize(
        "gate",
        [
            CX,
            CY,
            CZ,
            SQRT_XX,
            SQRT_XX_DAG,
            SQRT_YY,
            SQRT_YY_DAG,
            SQRT_ZZ,
            SQRT_ZZ_DAG,
            XCX,
            XCY,
            XCZ,
            YCX,
            YCY,
            YCZ,
        ],
    )
    @pytest.mark.parametrize(
        "target_gate",
        [
            CX,
            CY,
            CZ,
            SQRT_XX,
            SQRT_XX_DAG,
            SQRT_YY,
            SQRT_YY_DAG,
            SQRT_ZZ,
            SQRT_ZZ_DAG,
            XCX,
            XCY,
            XCZ,
            YCX,
            YCY,
            YCZ,
        ],
    )
    def test_compilation_successful_when_all_unitaries_available(
        self, gate, target_gate, up_to_paulis
    ):
        comp_data = CompilationData(
            {0: [], 1: [], 2: [], 3: []},
            {},
            {},
            {
                (0, Qubit(0), gate.stim_string): {"preceding": 0, "succeeding": 1},
                (0, Qubit(1), gate.stim_string): {"preceding": 2, "succeeding": 3},
            },
            {},
        )
        expected_two_qubit_dict = {
            (0, Qubit(0), target_gate.stim_string): {"preceding": 0, "succeeding": 1},
            (0, Qubit(1), target_gate.stim_string): {"preceding": 2, "succeeding": 3},
        }
        native_gate_set = NativeGateSet(
            one_qubit_gates=set({S, S_DAG, H, SQRT_X, SQRT_X_DAG, X, Z, Y}),
            two_qubit_gates=set({target_gate}),
        )
        assert (
            _compile_two_qubit_gates_to_native_gates(
                comp_data,
                native_gate_set,
                compilation_dict12_nosign if up_to_paulis else compilation_dict23,
                up_to_paulis,
                {0: 0},
            )[0].two_qubit_gates
            == expected_two_qubit_dict
        )

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize(
        "gate",
        [
            ISWAP,
            ISWAP_DAG,
            CXSWAP,
            CZSWAP,
        ],
    )
    @pytest.mark.parametrize(
        "target_gate",
        [
            ISWAP,
            ISWAP_DAG,
            CXSWAP,
            CZSWAP,
        ],
    )
    def test_compilation_successful_for_iswaplike_when_all_unitaries_available(
        self, gate, target_gate, up_to_paulis
    ):
        comp_data = CompilationData(
            {0: [], 1: [], 2: [], 3: []},
            {},
            {},
            {
                (0, Qubit(0), gate.stim_string): {"preceding": 0, "succeeding": 1},
                (0, Qubit(1), gate.stim_string): {"preceding": 2, "succeeding": 3},
            },
            {},
        )
        expected_two_qubit_dict = {
            (0, Qubit(0), target_gate.stim_string): {"preceding": 0, "succeeding": 1},
            (0, Qubit(1), target_gate.stim_string): {"preceding": 2, "succeeding": 3},
        }
        native_gate_set = NativeGateSet(
            one_qubit_gates=set({S, S_DAG, H, SQRT_X, SQRT_X_DAG, X, Z, Y}),
            two_qubit_gates=set({target_gate}),
        )
        assert (
            _compile_two_qubit_gates_to_native_gates(
                comp_data,
                native_gate_set,
                compilation_dict12_nosign if up_to_paulis else compilation_dict23,
                up_to_paulis,
                {0: 0},
            )[0].two_qubit_gates
            == expected_two_qubit_dict
        )

    @pytest.mark.parametrize(
        "comp_data, target_gate, expected_unitary_blocks",
        [
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                CX,
                {0: [], 1: [], 2: [], 3: []},
            ],
            [
                CompilationData(
                    {0: [H(0)], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                CX,
                {0: [H(0)], 1: [], 2: [], 3: []},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                CY,
                # s_dag * s_dag * s_dag == s
                {0: [], 1: [], 2: [S(1)], 3: [S_DAG(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [S_DAG(1)], 3: [S(1)]},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                CY,
                {0: [], 1: [], 2: [], 3: []},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                CZ,
                {0: [], 1: [], 2: [H(1)], 3: [H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                        (1, Qubit(0), "CX"): {"preceding": 1, "succeeding": 4},
                        (1, Qubit(1), "CX"): {"preceding": 3, "succeeding": 5},
                    },
                    {},
                ),
                CZ,
                {0: [], 1: [], 2: [H(1)], 3: [], 4: [], 5: [H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                        (1, Qubit(1), "CX"): {"preceding": 3, "succeeding": 4},
                        (1, Qubit(2), "CX"): {"preceding": 5, "succeeding": 6},
                    },
                    {},
                ),
                CZ,
                {0: [], 1: [], 2: [H(1)], 3: [H(1)], 4: [], 5: [H(2)], 6: [H(2)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                SQRT_XX,
                {0: [H(0)], 1: [H(0), S_DAG(0)], 2: [], 3: [H(1), S_DAG(1), H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                SQRT_XX_DAG,
                # sqrt_x = h * s_dag * s_dag * s_dag * h
                {0: [H(0)], 1: [H(0), S(0)], 2: [], 3: [SQRT_X(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                SQRT_YY,
                {0: [H(0), S(0)], 1: [SQRT_X(0)], 2: [S(1)], 3: [H(1), S(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                SQRT_YY_DAG,
                {
                    0: [H(0), S(0)],
                    1: [SQRT_X(0)],
                    2: [S_DAG(1)],
                    3: [S(1), H(1), S_DAG(1), H(1)],
                },
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                SQRT_ZZ,
                {0: [], 1: [S_DAG(0)], 2: [H(1)], 3: [S_DAG(1), H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                SQRT_ZZ_DAG,
                {0: [], 1: [S(0)], 2: [H(1)], 3: [S(1), H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CZ"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CZ"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                SQRT_XX,
                {
                    0: [H(0)],
                    1: [H(0), S_DAG(0)],
                    2: [H(1)],
                    3: [H(1), S_DAG(1)],
                },
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [H(1)], 4: [], 5: [H(1)]},
                    {(2, Qubit(1), "RZ"): {"preceding": 4, "succeeding": 5}},
                    {(1, Qubit(1), "MZ"): {"preceding": 3, "succeeding": 4}},
                    {
                        (0, Qubit(1), "CX"): {"preceding": 1, "succeeding": 3},
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 2},
                    },
                    {},
                ),
                SQRT_XX,
                {
                    0: [],
                    1: [H(1)],
                    2: [H(0), S_DAG(0), H(0)],
                    3: [H(1), S_DAG(1), H(1)],
                    4: [],
                    5: [H(1)],
                },
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                YCX,
                {0: [H(0), S(0)], 1: [S_DAG(0), H(0)], 2: [], 3: []},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "YCX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "YCX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                CX,
                {0: [S_DAG(0), H(0)], 1: [H(0), S(0)], 2: [], 3: []},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                YCZ,
                {0: [H(0), S(0)], 1: [S_DAG(0), H(0)], 2: [H(1)], 3: [H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "YCZ"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "YCZ"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                CX,
                {0: [S_DAG(0), H(0)], 1: [H(0), S(0)], 2: [H(1)], 3: [H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "YCX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "YCX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                YCZ,
                {0: [], 1: [], 2: [H(1)], 3: [H(1)]},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "YCZ"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "YCZ"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                YCX,
                {0: [], 1: [], 2: [H(1)], 3: [H(1)]},
            ],
        ],
    )
    def test_unitary_blocks_correct_when_compilation_successful(
        self, comp_data, target_gate, expected_unitary_blocks
    ):
        native_gate_set = NativeGateSet(
            one_qubit_gates=set({S, S_DAG, H, SQRT_X, SQRT_X_DAG, X, Z, Y}),
            two_qubit_gates=set({target_gate}),
        )
        assert (
            _compile_two_qubit_gates_to_native_gates(
                comp_data, native_gate_set, compilation_dict23, False, {0: 0}
            )[0].unitary_blocks
            == expected_unitary_blocks
        )

    @pytest.mark.parametrize("up_to_paulis", [True, False])
    @pytest.mark.parametrize(
        "gate",
        [
            CX,
            CY,
            CZ,
            SQRT_XX,
            SQRT_XX_DAG,
            SQRT_YY,
            SQRT_YY_DAG,
            SQRT_ZZ,
            SQRT_ZZ_DAG,
            XCX,
            XCY,
            XCZ,
            YCX,
            YCY,
            YCZ,
        ],
    )
    @pytest.mark.parametrize("target_gate", [ISWAP, ISWAP_DAG, CXSWAP, CZSWAP])
    def test_compiling_between_groups_throws_ValueError(
        self, gate, target_gate, up_to_paulis
    ):
        comp_data = CompilationData(
            {0: [], 1: [], 2: [], 3: []},
            {},
            {},
            {
                (0, Qubit(0), gate.stim_string): {"preceding": 0, "succeeding": 1},
                (0, Qubit(1), gate.stim_string): {"preceding": 0, "succeeding": 1},
            },
            {},
        )
        native_gate_set = NativeGateSet(two_qubit_gates=set({target_gate}))
        with pytest.raises(
            ValueError, match=r"Cannot compile between groups - .* to .* not supported"
        ):
            _compile_two_qubit_gates_to_native_gates(
                comp_data,
                native_gate_set,
                compilation_dict23 if not up_to_paulis else compilation_dict12_nosign,
                up_to_paulis,
                {0: 0},
            )

    @pytest.mark.parametrize(
        "comp_data, native_gate_set, layer_ind_lookup, expected_layer_ind_lookup",
        [
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(two_qubit_gates=set({CX})),
                {0: 0},
                {0: 0},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CXSWAP"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CXSWAP"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(two_qubit_gates=set({CXSWAP})),
                {0: 0},
                {0: 0},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(two_qubit_gates=set({CZ})),
                {0: 0, 1: 1, 2: 2},
                {0: 1, 1: 3, 2: 4},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CXSWAP"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CXSWAP"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(two_qubit_gates=set({CZSWAP})),
                {0: 0, 1: 1, 2: 2},
                {0: 1, 1: 3, 2: 4},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CZ"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CZ"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(two_qubit_gates=set({CX})),
                {0: 0, 1: 1, 2: 2},
                {0: 1, 1: 3, 2: 4},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CZSWAP"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CZSWAP"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(two_qubit_gates=set({CXSWAP})),
                {0: 0, 1: 1, 2: 2},
                {0: 1, 1: 3, 2: 4},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "SQRT_XX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "SQRT_XX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({CX}), one_qubit_gates=set({H, S, SQRT_X})
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 1, 1: 4, 2: 5},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CXSWAP"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CXSWAP"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({ISWAP}), one_qubit_gates=set({H, S_DAG})
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 1, 1: 4, 2: 5},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []},
                    {(2, Qubit(0), "RX"): {"preceding": 5, "succeeding": 6}},
                    {(1, Qubit(0), "MX"): {"preceding": 0, "succeeding": 5}},
                    {
                        (0, Qubit(0), "CXSWAP"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CXSWAP"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({ISWAP}),
                    one_qubit_gates=set({H, S_DAG}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MX}),
                ),
                {0: 0, 1: 1, 2: 2, 3: 3},
                {0: 1, 1: 4, 2: 5, 3: 6},
            ],
            [
                CompilationData(
                    {0: [H(0)], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [H(0)]},
                    {(3, Qubit(0), "RZ"): {"preceding": 5, "succeeding": 6}},
                    {(2, Qubit(0), "MZ"): {"preceding": 0, "succeeding": 5}},
                    {
                        (0, Qubit(0), "CXSWAP"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CXSWAP"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({ISWAP}),
                    one_qubit_gates=set({H, S_DAG}),
                    reset_gates=set({RZ}),
                    measurement_gates=set({MZ}),
                ),
                {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                {0: 1, 1: 4, 2: 5, 3: 6, 4: 7},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (0, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (0, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({SQRT_XX}), one_qubit_gates=set({H, S_DAG})
                ),
                {0: 0, 1: 1, 2: 2},
                {0: 1, 1: 5, 2: 6},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (1, Qubit(0), "CX"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(1), "CX"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({SQRT_XX}), one_qubit_gates=set({H, S_DAG})
                ),
                {0: 0, 1: 1, 2: 2, 3: 3},
                {0: 0, 1: 2, 2: 6, 3: 7},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: []},
                    {},
                    {},
                    {
                        (1, Qubit(0), "CZ"): {"preceding": 0, "succeeding": 1},
                        (1, Qubit(1), "CZ"): {"preceding": 2, "succeeding": 3},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({SQRT_XX}), one_qubit_gates=set({H, S_DAG, S})
                ),
                {0: 0, 1: 1, 2: 2, 3: 3},
                {0: 0, 1: 2, 2: 5, 3: 6},
            ],
            [
                CompilationData(
                    {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []},
                    {(2, Qubit(0), "RX"): {"preceding": 5, "succeeding": 6}},
                    {(1, Qubit(0), "MX"): {"preceding": 0, "succeeding": 5}},
                    {
                        (0, Qubit(2), "CZ"): {"preceding": 1, "succeeding": 3},
                        (0, Qubit(1), "CZ"): {"preceding": 2, "succeeding": 4},
                        (3, Qubit(0), "CZ"): {"preceding": 6, "succeeding": 7},
                        (3, Qubit(1), "CZ"): {"preceding": 3, "succeeding": 8},
                    },
                    {},
                ),
                NativeGateSet(
                    two_qubit_gates=set({SQRT_XX}),
                    one_qubit_gates=set({H, S_DAG, S}),
                    reset_gates=set({RX}),
                    measurement_gates=set({MX}),
                ),
                {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
                {0: 1, 1: 4, 2: 5, 3: 7, 4: 10, 5: 11},
            ],
        ],
    )
    def test_shifts_following_layers_to_accommodate_new_unitaries(
        self,
        comp_data,
        native_gate_set,
        layer_ind_lookup,
        expected_layer_ind_lookup,
    ):
        assert (
            _compile_two_qubit_gates_to_native_gates(
                comp_data,
                native_gate_set,
                compilation_dict23,
                False,
                layer_ind_lookup,
            )[1]
            == expected_layer_ind_lookup
        )

class TestGetTableauFromSequenceOf1qGates:
    def test_error_with_2q_gate(self):
        gates = ['X', 'I', 'CX']
        message = "'gates' must be composed of only single qubit gates"
        with pytest.raises(ValueError, match=message):
            _get_tableau_from_sequence_of_1q_gates(gates)


class TestGetSingleQubitsTableauKeyFromTwoQubitTableau:
    def test_with_3_qubit_tableau(self):
        tableau = stim.Tableau.from_circuit(stim.Circuit("H 0\nH 1\nH 2"))
        message = "The given tableau does not describe a two qubit gate"
        with pytest.raises(ValueError, match=message):
            _get_single_qubits_tableau_key_from_two_qubit_tableau(tableau, 0)


class TestGetRelevantDictToUpdate:
    def test_with_incorrect_gate(self):
        message = "Gate is not MR, M, R or 2Q"
        with pytest.raises(ValueError, match=message):
            _get_relevant_dict_to_update(X, {}, {}, {})
