# (c) Copyright Riverlane 2020-2025.

import pytest
import stim
from deltakit_core.decoding_graphs import FixedWidthBitstring

from deltakit_decode.utils._graph_circuit_helpers import (
    split_measurement_bitstring,
    stim_circuit_to_graph_dem,
)


def test_stim_circuit_to_graph_dem_does_not_decompose_the_rep_code():
    stim_rep_code = stim.Circuit.generated("repetition_code:memory",
                                             distance=5, rounds=5,
                                             after_clifford_depolarization=0.1)

    assert str(stim_circuit_to_graph_dem(stim_rep_code)).find('^') == -1


@pytest.mark.parametrize("code_task", [
    "surface_code:rotated_memory_x",
    "surface_code:unrotated_memory_z",
])
def test_stim_circuit_to_graph_dem_does_decompose_non_rep_codes(code_task):
    stim_rep_code = stim.Circuit.generated(code_task,
                                             distance=5, rounds=5,
                                             after_clifford_depolarization=0.1)

    assert str(stim_circuit_to_graph_dem(stim_rep_code)).find('^') != -1


class TestSplitMeasurementBitstring:

    @pytest.mark.parametrize(("stim_circuit", "measurement_bitstring", "expected_split_bitstring"), [
        (
            stim.Circuit.generated("surface_code:rotated_memory_x",
                                   distance=3,
                                   rounds=1,
                                   after_clifford_depolarization=0.1),
            FixedWidthBitstring(17, 0b01011010001011011),
            [FixedWidthBitstring(8, 0b01011011),
             FixedWidthBitstring(9, 0b010110100)],
        ), (
            stim.Circuit.generated("surface_code:rotated_memory_x",
                                   distance=3,
                                   rounds=3,
                                   after_clifford_depolarization=0.1),
            FixedWidthBitstring(33, 0b110011101100101001010001110000000),
            [FixedWidthBitstring(8, 0b10000000),
             FixedWidthBitstring(8, 0b10100011),
             FixedWidthBitstring(8, 0b10010100),
             FixedWidthBitstring(9, 0b110011101)],
        )
    ])
    def test_measurement_bitstring_can_be_split_by_each_layer_of_measurement_gates(self, stim_circuit, measurement_bitstring, expected_split_bitstring):
        split_bitstring = split_measurement_bitstring(
            measurement_bitstring, stim_circuit)
        for split_bitstring_i, expected_split_bitstring_i in zip(split_bitstring, expected_split_bitstring):
            assert split_bitstring_i == expected_split_bitstring_i

    @pytest.mark.parametrize(("stim_circuit", "measurement_bitstring", "expected_number_of_bitstrings"), [
        (
            stim.Circuit.generated("surface_code:rotated_memory_x",
                                   distance=3,
                                   rounds=1,
                                   after_clifford_depolarization=0.1),
            FixedWidthBitstring(17, 0b01011010001011011),
            2,
        ), (
            stim.Circuit.generated("surface_code:rotated_memory_x",
                                   distance=3,
                                   rounds=3,
                                   after_clifford_depolarization=0.1),
            FixedWidthBitstring(33, 0b110011101100101001010001110000000),
            4,
        )
    ])
    def test_length_of_split_bitstring_matches_number_of_layers_with_measurement_gates(
            self, stim_circuit, measurement_bitstring, expected_number_of_bitstrings):
        split_bitstring = split_measurement_bitstring(
            measurement_bitstring, stim_circuit)
        assert len(split_bitstring) == expected_number_of_bitstrings
