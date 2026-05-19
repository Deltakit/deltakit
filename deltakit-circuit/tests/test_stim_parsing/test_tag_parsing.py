import itertools

import deltakit_stim
import pytest

from deltakit_circuit._circuit import Circuit
from deltakit_circuit.gates._measurement_gates import (
    HERALD_LEAKAGE_EVENT,
    MEASUREMENT_GATES,
    MPP,
)
from deltakit_circuit.gates._one_qubit_gates import ONE_QUBIT_GATES
from deltakit_circuit.gates._reset_gates import RESET_GATES
from deltakit_circuit.gates._two_qubit_gates import TWO_QUBIT_GATES
from deltakit_circuit.noise_channels._correlated_noise import ALL_CORRELATED_NOISE


@pytest.mark.parametrize(
    ("instr_template", "tag"),
    itertools.product(
        [
            *(f"{sqg.deltakit_stim_string}[{{tag}}] 0" for sqg in ONE_QUBIT_GATES),
            *(f"{tqg.deltakit_stim_string}[{{tag}}] 0 1" for tqg in TWO_QUBIT_GATES),
            *(
                f"{mg.deltakit_stim_string}[{{tag}}] 0"
                for mg in (MEASUREMENT_GATES - {MPP, HERALD_LEAKAGE_EVENT})
            ),
            *(f"{rg.deltakit_stim_string}[{{tag}}] 0" for rg in RESET_GATES),
            *(
                f"{cng.deltakit_stim_string}[{{tag}}](0.1) X1 Z2"
                for cng in ALL_CORRELATED_NOISE
            ),
        ],
        ["", "sjkdhf", "λ", "leaky<0>"],
    ),
)
def test_parse_tagged_instruction(instr_template: str, tag: str) -> None:
    instr_str = instr_template.format(tag=tag)
    deltakit_stim_circuit = deltakit_stim.Circuit(instr_str)
    circuit = Circuit.from_deltakit_stim_circuit(deltakit_stim_circuit)
    assert circuit.as_deltakit_stim_circuit() == deltakit_stim_circuit
