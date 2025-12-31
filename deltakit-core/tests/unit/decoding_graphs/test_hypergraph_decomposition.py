# (c) Copyright Riverlane 2020-2025.

import pytest
from deltakit_core.decoding_graphs import DecodingHyperEdge
from deltakit_core.decoding_graphs._hypergraph_decomposition import decompositions


class TestDecompositions:
    @pytest.mark.parametrize(
        "edge, edges",
        [
            (DecodingHyperEdge((0, 1)), []),
            (
                DecodingHyperEdge((0, 1, 2)),
                [DecodingHyperEdge((0,)), DecodingHyperEdge((2,))],
            ),
        ],
    )
    def test_decomposing_item_without_complete_decompositions_gives_empty_list(
        self,
        edge: DecodingHyperEdge,
        edges: list[DecodingHyperEdge],
    ):
        assert list(decompositions(edge, edges)) == []

    def test_there_is_only_one_decomposition_into_singletons(self):
        assert (
            len(
                list(
                    decompositions(
                        DecodingHyperEdge(range(3)),
                        [DecodingHyperEdge({i}) for i in range(3)],
                    )
                )
            )
            == 1
        )

    def test_id_of_decomposed_edges_matches_that_of_input_legal_edges(self):
        decomp_items = [DecodingHyperEdge({0, 1}), DecodingHyperEdge({2})]
        decomposition = next(decompositions(DecodingHyperEdge({0, 1, 2}), decomp_items))
        assert {id(item) for item in decomp_items} == {
            id(decomp) for decomp in decomposition
        }

    @pytest.mark.parametrize(
        "edges",
        [
            [DecodingHyperEdge({0, 1}), DecodingHyperEdge({2})],
            [DecodingHyperEdge({0, 2}), DecodingHyperEdge({1})],
            [DecodingHyperEdge({1, 2}), DecodingHyperEdge({0})],
            [DecodingHyperEdge({0}), DecodingHyperEdge({1, 2})],
            [DecodingHyperEdge({1}), DecodingHyperEdge({0, 2})],
            [DecodingHyperEdge({2}), DecodingHyperEdge({0, 1})],
            [DecodingHyperEdge({0}), DecodingHyperEdge({1}), DecodingHyperEdge({2})],
        ],
    )
    def test_individual_decompositions_of_degree_three_edge_is_expected_decomposition(
        self, edges: list[DecodingHyperEdge]
    ):
        assert next(decompositions(DecodingHyperEdge(range(3)), edges)) == set(edges)

    def test_all_decompositions_of_degree_three_edge_matches_expected_decomposition(
        self,
    ):
        edges = [
            DecodingHyperEdge({0}),
            DecodingHyperEdge({1}),
            DecodingHyperEdge({2}),
            DecodingHyperEdge({0, 1}),
            DecodingHyperEdge({1, 2}),
            DecodingHyperEdge({0, 2}),
        ]
        assert list(decompositions(DecodingHyperEdge(range(3)), edges)) == [
            {DecodingHyperEdge({i}) for i in range(3)},
            {DecodingHyperEdge({0}), DecodingHyperEdge({1, 2})},
            {DecodingHyperEdge({1}), DecodingHyperEdge({0, 2})},
            {DecodingHyperEdge({2}), DecodingHyperEdge({0, 1})},
        ]

    def test_decompositions_returned_in_different_order_if_input_is_in_different_order(
        self,
    ):
        edge = DecodingHyperEdge(range(3))
        edges = [
            DecodingHyperEdge({0}),
            DecodingHyperEdge({1}),
            DecodingHyperEdge({2}),
            DecodingHyperEdge({0, 1}),
            DecodingHyperEdge({1, 2}),
            DecodingHyperEdge({0, 2}),
        ]
        assert list(decompositions(edge, reversed(edges))) == [
            {DecodingHyperEdge({1}), DecodingHyperEdge({0, 2})},
            {DecodingHyperEdge({0}), DecodingHyperEdge({1, 2})},
            {DecodingHyperEdge({2}), DecodingHyperEdge({0, 1})},
            {DecodingHyperEdge({i}) for i in range(3)},
        ]

    @pytest.mark.parametrize(
        "edge, edges, expected_decomposition",
        [
            (
                DecodingHyperEdge({0, 1}),
                [DecodingHyperEdge({0}), DecodingHyperEdge({1})],
                {DecodingHyperEdge({0}), DecodingHyperEdge({1})},
            ),
            (
                DecodingHyperEdge({0, 1, 2}),
                [DecodingHyperEdge({i}) for i in range(4)],
                {DecodingHyperEdge({i}) for i in range(3)},
            ),
            (
                DecodingHyperEdge((0, 1, 2)),
                [
                    DecodingHyperEdge({0}),
                    DecodingHyperEdge({1, 2}),
                    DecodingHyperEdge({1}),
                    DecodingHyperEdge({2}),
                ],
                {DecodingHyperEdge({0}), DecodingHyperEdge({1, 2})},
            ),
            (
                DecodingHyperEdge({0, 1, 2, 3}),
                [
                    DecodingHyperEdge({0, 1}),
                    DecodingHyperEdge({1, 3}),
                    DecodingHyperEdge({0, 2}),
                    DecodingHyperEdge({2}),
                ],
                {DecodingHyperEdge({0, 2}), DecodingHyperEdge({1, 3})},
            ),
            (
                DecodingHyperEdge({0, 1, 2, 3, 4}),
                [
                    DecodingHyperEdge({0, 1}),
                    DecodingHyperEdge({2, 3}),
                    DecodingHyperEdge({4}),
                ],
                {
                    DecodingHyperEdge({0, 1}),
                    DecodingHyperEdge({2, 3}),
                    DecodingHyperEdge({4}),
                },
            ),
        ],
    )
    def test_first_decomposition_of_hyperedge_matches_expected_decomposition(
        self,
        edge: DecodingHyperEdge,
        edges: list[DecodingHyperEdge],
        expected_decomposition: frozenset[DecodingHyperEdge],
    ):
        assert next(decompositions(edge, edges)) == expected_decomposition
