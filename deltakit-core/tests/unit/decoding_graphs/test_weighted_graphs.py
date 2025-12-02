# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import pytest
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DetectorRecord,
    NXDecodingGraph,
    change_graph_error_probabilities,
    vector_weights,
)


def approx_contains(container: Iterable[npt.ArrayLike], element: npt.ArrayLike) -> bool:
    return any(np.allclose(item, element) for item in container)


class TestChangingGraphErrorProbabilities:
    @pytest.mark.parametrize(
        "graph, p_errors",
        [
            (
                NXDecodingGraph.from_edge_list(
                    [],
                    detector_records={
                        0: DetectorRecord((0, 0), 0),
                        1: DetectorRecord((0, 0), 1),
                    },
                ),
                [0.01],
            ),
            (NXDecodingGraph.from_edge_list([(0, 1)]), []),
            (NXDecodingGraph.from_edge_list([(0, 1)]), [0.01, 0.02]),
        ],
    )
    def test_error_is_raised_when_changing_p_err_with_unequal_num_of_p_errors_to_edges(
        self, graph: NXDecodingGraph, p_errors: Sequence[float]
    ):
        with pytest.raises(
            ValueError,
            match="There should be an equal number of new "
            "error probabilities to edges in the graph.",
        ):
            change_graph_error_probabilities(graph, p_errors)

    def test_p_err_on_single_edge_of_graph_is_changed(self):
        graph = NXDecodingGraph.from_edge_list([(0, 1)])
        new_p_error = 0.01
        changed_graph = change_graph_error_probabilities(graph, [new_p_error])
        assert all(
            edge_record.p_err == new_p_error
            for edge_record in changed_graph.edge_records.values()
        )

    def test_changing_graph_p_err_keeps_the_same_detector_records(self):
        detector_records = {
            0: DetectorRecord((0, 0), 0),
            1: DetectorRecord((1, 0), 0),
            2: DetectorRecord((0, 1), 0),
        }
        graph = NXDecodingGraph.from_edge_list([], detector_records=detector_records)
        assert (
            change_graph_error_probabilities(graph, []).detector_records
            == detector_records
        )

    def test_changing_graph_p_err_keeps_the_same_boundaries(self):
        boundaries = {1, 2, 3}
        graph = NXDecodingGraph.from_edge_list(
            [],
            detector_records={
                1: DetectorRecord((0, 0), 0),
                2: DetectorRecord((1, 0), 0),
                3: DetectorRecord((0, 1), 0),
            },
            boundaries=boundaries,
        )
        assert change_graph_error_probabilities(graph, ()).boundaries == boundaries


class TestVectorEdges:
    @pytest.mark.parametrize(
        "graph",
        [
            NXDecodingGraph.from_edge_list([(0, 1)]),
            NXDecodingGraph.from_edge_list(
                [(0, 1)], detector_records={0: DetectorRecord((0, 0), 0)}
            ),
            NXDecodingGraph.from_edge_list(
                [(0, 1)], detector_records={1: DetectorRecord((0, 0), 0)}
            ),
        ],
    )
    def test_error_is_raised_if_edges_dont_have_coordinates(
        self, graph: NXDecodingGraph
    ):
        with pytest.raises(ValueError, match=r".*does not have proper coordinates."):
            vector_weights(graph)

    def test_vector_components_are_python_types_not_numpy_types(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1)],
            detector_records={
                0: DetectorRecord((0, 0), 0),
                1: DetectorRecord((0, 1), 0),
            },
        )
        for vector in vector_weights(graph):
            for component in vector:
                assert isinstance(component, int)

    def test_vectors_on_graph_with_no_edges_is_empty_dictionary(self):
        graph = NXDecodingGraph.from_edge_list(
            [],
            detector_records={
                0: DetectorRecord((0, 0), 0),
                1: DetectorRecord((1, 1), 1),
            },
        )
        assert vector_weights(graph) == {}

    def test_vectors_to_the_boundary_are_mapped_to_the_zero_vector(self):
        graph = NXDecodingGraph.from_edge_list(
            [DecodingEdge(0, 1)],
            detector_records={
                0: DetectorRecord((0, 0), 0),
                1: DetectorRecord((1, 1), 1),
            },
            boundaries=[1],
        )
        assert vector_weights(graph)[(0, 0, 0)] == [math.inf]

    def test_vectors_with_different_magnitudes_are_different_vectors(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1), (0, 2)],
            detector_records={
                0: DetectorRecord((0, 0), 0),
                1: DetectorRecord((1, 0), 0),
                2: DetectorRecord((2, 0), 0),
            },
        )
        assert len(vector_weights(graph)) == 2

    def test_edge_differing_in_direction_is_grouped_in_vector_weights(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1), (2, 3)],
            detector_records={
                0: DetectorRecord((0, 0), 0),
                1: DetectorRecord((0, 0), 1),
                2: DetectorRecord((0, 0), 2),
                3: DetectorRecord((0, 0), 3),
            },
        )
        assert len(vector_weights(graph)) == 1
