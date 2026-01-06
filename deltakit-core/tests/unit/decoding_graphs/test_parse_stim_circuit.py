# This file contains information which is proprietary to Riverlane Limited
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2021-2025. All rights reserved.

import pytest

try:
    import lestim as stim
except ImportError:
    import stim

from deltakit_core.decoding_graphs._decoding_graph_tools import parse_stim_circuit


def stim_circuit_rep_5x4() -> stim.Circuit:
    return stim.Circuit.generated(
        "repetition_code:memory",
        rounds=4,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


def stim_circuit_rplanar_3x3x3() -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


def stim_circuit_planar_5x5x2() -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        rounds=2,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


class TestParseStimCircuit:
    @pytest.fixture(
        params=[
            stim_circuit_rep_5x4(),
            stim_circuit_rplanar_3x3x3(),
            stim_circuit_planar_5x5x2(),
        ],
        scope="class",
    )
    def stim_circuit(self, request) -> stim.Circuit:
        return request.param

    def test_trimmed_stim_circuit_has_same_number_of_detectors_as_its_corresponding_trimmed_graph(
        self, stim_circuit: stim.Circuit
    ) -> None:
        trimmed_graph, _, trimmed_stim_circuit = parse_stim_circuit(stim_circuit)
        assert trimmed_stim_circuit.num_detectors == len(trimmed_graph.nodes) - len(
            trimmed_graph.boundaries
        )

    def test_trimmed_stim_circuit_has_same_number_of_observables_as_its_corresponding_trimmed_graph(
        self, stim_circuit: stim.Circuit
    ) -> None:
        _, trimmed_logicals, trimmed_stim_circuit = parse_stim_circuit(stim_circuit)
        assert trimmed_stim_circuit.num_observables == len(trimmed_logicals)
