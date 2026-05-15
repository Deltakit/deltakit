# This file contains information which is proprietary to Riverlane Limited
# ("Riverlane") and is Riverlane Confidential Information.
# (c) Copyright Riverlane 2021-2025. All rights reserved.

import deltakit_stim
import pytest

from deltakit_core.decoding_graphs._decoding_graph_tools import parse_stim_circuit


def deltakit_stim_circuit_rep_5x4() -> deltakit_stim.Circuit:
    return deltakit_stim.Circuit.generated(
        "repetition_code:memory",
        rounds=4,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


def deltakit_stim_circuit_rplanar_3x3x3() -> deltakit_stim.Circuit:
    return deltakit_stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


def deltakit_stim_circuit_planar_5x5x2() -> deltakit_stim.Circuit:
    return deltakit_stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        rounds=2,
        distance=5,
        before_round_data_depolarization=0.1,
        before_measure_flip_probability=0.1,
    )


class TestParseDeltakit_StimCircuit:
    @pytest.fixture(
        params=[
            deltakit_stim_circuit_rep_5x4(),
            deltakit_stim_circuit_rplanar_3x3x3(),
            deltakit_stim_circuit_planar_5x5x2(),
        ],
        scope="class",
    )
    def deltakit_stim_circuit(self, request) -> deltakit_stim.Circuit:
        return request.param

    def test_trimmed_deltakit_stim_circuit_has_same_number_of_detectors_as_its_corresponding_trimmed_graph(
        self, deltakit_stim_circuit: deltakit_stim.Circuit
    ) -> None:
        trimmed_graph, _, trimmed_deltakit_stim_circuit = parse_stim_circuit(
            deltakit_stim_circuit
        )
        assert trimmed_deltakit_stim_circuit.num_detectors == len(
            trimmed_graph.nodes
        ) - len(trimmed_graph.boundaries)

    def test_trimmed_deltakit_stim_circuit_has_same_number_of_observables_as_its_corresponding_trimmed_graph(
        self, deltakit_stim_circuit: deltakit_stim.Circuit
    ) -> None:
        _, trimmed_logicals, trimmed_deltakit_stim_circuit = parse_stim_circuit(
            deltakit_stim_circuit
        )
        assert trimmed_deltakit_stim_circuit.num_observables == len(trimmed_logicals)
