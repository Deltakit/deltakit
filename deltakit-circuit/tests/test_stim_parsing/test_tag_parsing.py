import itertools

import pytest
import stim

from deltakit_circuit._circuit import Circuit
from deltakit_circuit._qubit_identifiers import Coordinate, Qubit
from deltakit_circuit.gates._measurement_gates import (
    HERALD_LEAKAGE_EVENT,
    MEASUREMENT_GATES,
    MPP,
)
from deltakit_circuit.gates._one_qubit_gates import ONE_QUBIT_GATES
from deltakit_circuit.gates._reset_gates import RESET_GATES
from deltakit_circuit.gates._two_qubit_gates import TWO_QUBIT_GATES
from deltakit_circuit.noise_channels._correlated_noise import ALL_CORRELATED_NOISE


@pytest.fixture
def qubit_mapping() -> dict[int, Qubit]:
    return {i: Qubit(Coordinate(i, i)) for i in range(3)}


@pytest.mark.parametrize(
    "instr_template,tag",
    itertools.product(
        [
            *(f"{sqg.stim_string}[{{tag}}] 0" for sqg in ONE_QUBIT_GATES),
            *(f"{tqg.stim_string}[{{tag}}] 0 1" for tqg in TWO_QUBIT_GATES),
            *(
                f"{mg.stim_string}[{{tag}}] 0"
                for mg in (MEASUREMENT_GATES - {MPP, HERALD_LEAKAGE_EVENT})
            ),
            *(f"{rg.stim_string}[{{tag}}] 0" for rg in RESET_GATES),
            *(f"{cng.stim_string}[{{tag}}](0.1) X1 Z2" for cng in ALL_CORRELATED_NOISE),
        ],
        ["", "sjkdhf", "λ", "leaky<0>"],
    ),
)
def test_parse_tagged_instruction(
    instr_template: str, tag: str, qubit_mapping: dict[int, Qubit]
) -> None:
    instr_str = instr_template.format(tag=tag)
    stim_circuit = stim.Circuit(instr_str)
    circuit = Circuit.from_stim_circuit(stim_circuit)
    assert circuit.as_stim_circuit() == stim_circuit
