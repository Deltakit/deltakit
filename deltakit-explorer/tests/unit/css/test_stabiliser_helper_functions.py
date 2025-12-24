# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D
from deltakit_explorer.codes._css._stabiliser_helper_functions import \
    pauli_gates_to_stim_pauli_string
from stim import PauliString

import stim
from packaging.version import Version
from importlib.metadata import version


CURRENT_STIM_VERSION = Version(version("stim"))
STIM_VERSION_V1_13_0 = Version("1.13.0")


class TestPauliGatesToStimPauliString:
    @pytest.mark.parametrize("iterable", [[], ()])
    def test_empty_iterable_returns_empty_PauliString(self, iterable):
        assert pauli_gates_to_stim_pauli_string(iterable, {}) == PauliString("")

    @pytest.mark.parametrize(
        "pauli_gates, data_qubit_to_index_lookup",
        [
            [[PauliX(Qubit(Coord2D(1, 1)))], {}],
            [[PauliZ(Qubit(Coord2D(1, 1)))], {}],
            [
                [PauliX(Qubit(Coord2D(1, 1))), PauliX(Qubit(Coord2D(2, 2)))],
                {Qubit(Coord2D(1, 1)): 0},
            ],
            [
                [PauliX(Qubit(Coord2D(1, 1))), PauliX(Qubit(Coord2D(2, 2)))],
                {Qubit(Coord2D(2, 2)): 0},
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1))), PauliZ(Qubit(Coord2D(2, 2)))],
                {Qubit(Coord2D(1, 1)): 0},
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1))), PauliZ(Qubit(Coord2D(2, 2)))],
                {Qubit(Coord2D(2, 2)): 0},
            ],
        ],
    )
    def test_raises_ValueError_if_paulis_not_in_dictionary(
        self, pauli_gates, data_qubit_to_index_lookup
    ):
        with pytest.raises(
            ValueError,
            match=r"data_qubit_to_index_lookup does not contain entries for .* in pauli_gates",
        ):
            pauli_gates_to_stim_pauli_string(pauli_gates, data_qubit_to_index_lookup)

    @pytest.mark.skipif(
        CURRENT_STIM_VERSION < STIM_VERSION_V1_13_0,
        reason=(
            "Parsing Pauli strings with indices has been introduced in Stim v1.13.0."
            "See https://github.com/quantumlib/Stim/releases/tag/v1.13.0."
            f"Current Stim version is {CURRENT_STIM_VERSION}."
        )
    )
    @pytest.mark.parametrize(
        "pauli_gates, data_qubit_to_index_lookup, expected_pauli_string",
        [
            [
                [PauliX(Qubit(Coord2D(1, 1)))],
                {Qubit(Coord2D(1, 1)): 0},
                PauliString("X"),
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1)))],
                {Qubit(Coord2D(1, 1)): 0},
                PauliString("Z"),
            ],
            [
                [PauliX(Qubit(Coord2D(1, 1)))],
                {Qubit(Coord2D(1, 1)): 0, Qubit(Coord2D(2, 2)): 1},
                PauliString("X_"),
            ],
            [
                [PauliX(Qubit(Coord2D(2, 2)))],
                {Qubit(Coord2D(1, 1)): 0, Qubit(Coord2D(2, 2)): 1},
                PauliString("_X"),
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1)))],
                {Qubit(Coord2D(1, 1)): 0, Qubit(Coord2D(2, 2)): 1},
                PauliString("Z_"),
            ],
            [
                [PauliZ(Qubit(Coord2D(2, 2)))],
                {Qubit(Coord2D(1, 1)): 0, Qubit(Coord2D(2, 2)): 1},
                PauliString("_Z"),
            ],
            [
                [PauliX(Qubit(Coord2D(1, 1))), PauliZ(Qubit(Coord2D(2, 2)))],
                {Qubit(Coord2D(1, 1)): 0, Qubit(Coord2D(2, 2)): 1},
                PauliString("XZ"),
            ],
            [
                [PauliX(Qubit(Coord2D(1, 1))), PauliZ(Qubit(Coord2D(3, 3)))],
                {
                    Qubit(Coord2D(1, 1)): 0,
                    Qubit(Coord2D(2, 2)): 1,
                    Qubit(Coord2D(3, 3)): 2,
                },
                PauliString("X_Z"),
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1))), PauliX(Qubit(Coord2D(3, 3)))],
                {
                    Qubit(Coord2D(1, 1)): 0,
                    Qubit(Coord2D(2, 2)): 1,
                    Qubit(Coord2D(3, 3)): 2,
                },
                PauliString("Z_X"),
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1))), PauliX(Qubit(Coord2D(3, 3)))],
                {
                    Qubit(Coord2D(1, 1)): 1,
                    Qubit(Coord2D(2, 2)): 2,
                    Qubit(Coord2D(3, 3)): 0,
                },
                PauliString("XZ_"),
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1))), PauliX(Qubit(Coord2D(3, 3)))],
                {
                    Qubit(Coord2D(1, 1)): 2,
                    Qubit(Coord2D(2, 2)): 0,
                    Qubit(Coord2D(3, 3)): 1,
                },
                PauliString("_XZ"),
            ],
            [
                [PauliZ(Qubit(Coord2D(1, 1))), PauliX(Qubit(Coord2D(3, 3)))],
                {
                    Qubit(Coord2D(1, 1)): 1,
                    Qubit(Coord2D(2, 2)): 0,
                    Qubit(Coord2D(3, 3)): 2,
                },
                PauliString("_ZX"),
            ],
            [
                [PauliX(Qubit(Coord2D(1, 1))), PauliZ(Qubit(Coord2D(3, 3)))],
                {
                    Qubit(Coord2D(1, 1)): 0,
                    Qubit(Coord2D(2, 2)): 7,
                    Qubit(Coord2D(3, 3)): 2,
                },
                PauliString("X_Z"),
            ],
            [
                [PauliZ(Qubit(Coord2D(3, 3))), PauliX(Qubit(Coord2D(1, 1)))],
                {
                    Qubit(Coord2D(1, 1)): 0,
                    Qubit(Coord2D(2, 2)): 7,
                    Qubit(Coord2D(3, 3)): 2,
                },
                PauliString("X_Z"),
            ],
        ],
    )
    def test_gives_valid_pauli_string(
        self, pauli_gates, data_qubit_to_index_lookup, expected_pauli_string
    ) -> None:
        assert (
            pauli_gates_to_stim_pauli_string(pauli_gates, data_qubit_to_index_lookup)
            == expected_pauli_string
        )
