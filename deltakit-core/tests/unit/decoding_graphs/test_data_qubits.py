# (c) Copyright Riverlane 2020-2025.
"""Tests for data qubits datastrucutres."""

import math
from typing import AbstractSet, Iterable, List, Tuple

import pytest
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DecodingHyperEdge,
    DetectorRecord,
    EdgeRecord,
    OrderedDecodingEdges,
)
from deltakit_core.decoding_graphs._data_qubits import EdgeT, errors_to_syndrome
from deltakit_core.decoding_graphs._syndromes import Bit, OrderedSyndrome


class TestDecodingHyperEdge:
    @pytest.mark.parametrize(
        "hyperedge", [DecodingHyperEdge(range(3)), DecodingHyperEdge(range(5))]
    )
    def test_error_is_raised_when_casting_hyperedge_of_degree_greater_than_two(
        self, hyperedge: DecodingHyperEdge
    ):
        with pytest.raises(
            ValueError, match=r"Cannot cast edge: .* of degree " r"\d to decoding edge."
        ):
            hyperedge.to_decoding_edge(-1)

    def test_error_is_raised_when_casting_degree_one_hyperedge_without_specifying_boundary(
        self,
    ):
        with pytest.raises(
            ValueError,
            match="Boundary vertex is required for " r"edge: \(0,\) of degree one.",
        ):
            DecodingHyperEdge({0}).to_decoding_edge()

    def test_casting_hyperedge_of_degree_two_returns_expected_decoding_edge(self):
        assert DecodingHyperEdge((0, 1)).to_decoding_edge(-1) == DecodingEdge(0, 1)

    def test_casting_hyperedge_of_degree_one_returns_expected_decoding_edge(self):
        assert DecodingHyperEdge((0,)).to_decoding_edge(-1) == DecodingEdge(0, -1)


class TestDecodingEdge:
    """Tests for basic decoding edge representation."""

    @pytest.fixture(
        params=[
            DecodingEdge(6, 7),
            DecodingEdge(7, 0),
            DecodingEdge(-1, 5),
            DecodingEdge(0, 1),
            DecodingEdge(4, 9),
            DecodingEdge(1, 9),
        ]
    )
    def decoding_edge(self, request) -> DecodingEdge:
        return request.param

    def test_decoding_edge_equals_reversed_edge(self, decoding_edge: DecodingEdge):
        reversed_edge = DecodingEdge(decoding_edge.second, decoding_edge.first)
        assert decoding_edge == reversed_edge

    def test_decoding_edge_hash_equals_reversed_edge(self, decoding_edge: DecodingEdge):
        reversed_edge = DecodingEdge(decoding_edge.second, decoding_edge.first)
        assert hash(decoding_edge) == hash(reversed_edge)

    def test_decoding_edge_unpacks_to_first_and_second(
        self, decoding_edge: DecodingEdge
    ):
        first, second = decoding_edge
        assert (first, second) == (decoding_edge.first, decoding_edge.second)

    def test_decoding_edge_length(self, decoding_edge: DecodingEdge):
        assert len(decoding_edge) == 2

    def test_decoding_edge_membership(self, decoding_edge: DecodingEdge):
        assert (
            decoding_edge.first in decoding_edge
            and decoding_edge.second in decoding_edge
        )

    def test_self_loop_decoding_edge_raises_value_error(self, random_generator):
        syndrome_a = random_generator.integers(100)
        with pytest.raises(ValueError):
            DecodingEdge(syndrome_a, syndrome_a)

    def test_example_decoding_edge_string(self):
        assert str(DecodingEdge(5, 2)) == "(2, 5)"

    def test_timelike_edge_is_timelike(self, random_generator):
        index = random_generator.integers(1000)
        time_1, time_2 = random_generator.choice(1000, 2, replace=False)
        detector_records = [
            DetectorRecord(index, time_1),
            DetectorRecord(index, time_2),
        ]
        assert DecodingEdge(0, 1).is_timelike(detector_records)

    def test_spacelike_edge_is_spacelike(self, random_generator):
        index_1, index_2 = random_generator.choice(1000, 2, replace=False)
        time = random_generator.integers(1000)
        detector_records = [
            DetectorRecord(index_1, time),
            DetectorRecord(index_2, time),
        ]
        assert DecodingEdge(0, 1).is_spacelike(detector_records)

    def test_hooklike_edge_is_hooklike(self, random_generator):
        index_1, index_2 = random_generator.choice(1000, 2, replace=False)
        time_1, time_2 = random_generator.choice(1000, 2, replace=False)
        detector_records = [
            DetectorRecord(index_1, time_1),
            DetectorRecord(index_2, time_2),
        ]
        assert DecodingEdge(0, 1).is_hooklike(detector_records)


class TestEdgeRecord:
    """Tests for EdgeRecord object."""

    @pytest.mark.parametrize(
        "weight, expected_p_err",
        [
            (0, 0.5),
            (1, 0.26894),
            (2, 0.11920),
            (-1, 0.73106),
            (math.inf, 0),
            (-math.inf, 1),
        ],
    )
    def test_constructor_from_loglikelihood(self, weight, expected_p_err):
        edgerecord = EdgeRecord.from_loglikelihood(weight)
        assert math.isclose(edgerecord.p_err, expected_p_err, abs_tol=1e-5)
        assert math.isclose(edgerecord.weight, weight, abs_tol=1e-9)

    def test_constructor_sets_key(self):
        edgerecord = EdgeRecord(stock_edge_key="val")
        assert edgerecord["stock_edge_key"] == "val"

        edgerecord = EdgeRecord(0.1, stock_edge_key="val")
        assert edgerecord["p_err"] == 0.1
        assert edgerecord["stock_edge_key"] == "val"

    @pytest.mark.parametrize(
        "edge_dict, expected_edge_record",
        [
            ({}, EdgeRecord()),
            ({"p_err": 0.5}, EdgeRecord(p_err=0.5)),
            ({"stock_edge_key": "val"}, EdgeRecord(stock_edge_key="val")),
            (
                {"p_err": 0.4, "stock_edge_key": "val"},
                EdgeRecord(p_err=0.4, stock_edge_key="val"),
            ),
        ],
    )
    def test_constructor_from_dict(self, edge_dict, expected_edge_record):
        edgerecord = EdgeRecord.from_dict(edge_dict)
        assert edgerecord == expected_edge_record

    @pytest.mark.parametrize(
        "invalid_perr",
        [
            -0.1,
            2,
            1.4,
            math.inf,
            -math.inf,
        ],
    )
    def test_weight_for_invalid_raises_value_error(self, invalid_perr):
        with pytest.raises(
            ValueError,
            match=rf"Edge weight undefined for error probability {invalid_perr}",
        ):
            print(
                f"invalid per {invalid_perr}, weight {EdgeRecord(p_err=invalid_perr).weight}"
            )

    @pytest.mark.parametrize(
        "perr, expected_weight",
        [
            (0.5, 0),
            (0.26894, 1),
            (0.119202, 2),
            (0.73106, -1),
            (0, math.inf),
            (1, -math.inf),
        ],
    )
    def test_weight_changes_according_to_perr(self, perr, expected_weight):
        edgerecord = EdgeRecord()
        edgerecord.p_err = perr

        assert math.isclose(edgerecord.weight, expected_weight, abs_tol=1e-5)


class TestOrderedDecodingEdges:
    """Tests for ordered collections of decoding edges."""

    @pytest.fixture(
        params=[
            OrderedDecodingEdges(),
            OrderedDecodingEdges([DecodingEdge(i, i + 1) for i in range(100)]),
            OrderedDecodingEdges([DecodingEdge(i, i + 1) for i in range(55, 10, -2)]),
            OrderedDecodingEdges([DecodingEdge(3, 5)]),
            OrderedDecodingEdges([DecodingEdge(3, 4)]),
            OrderedDecodingEdges(
                [
                    DecodingEdge(1, 0),
                    DecodingEdge(3, 9),
                    DecodingEdge(3, 9),
                    DecodingEdge(1, 2),
                ]
            ),
            OrderedDecodingEdges(
                [
                    DecodingEdge(9, 0),
                    DecodingEdge(1, 3),
                    DecodingEdge(9, 0),
                    DecodingEdge(1, 2),
                ]
            ),
            OrderedDecodingEdges(
                [
                    DecodingEdge(9, 0),
                    DecodingEdge(9, 0),
                    DecodingEdge(9, 0),
                    DecodingEdge(1, 2),
                ],
                mod_2_filter=False,
            ),
            OrderedDecodingEdges(
                [
                    DecodingEdge(9, 0),
                    DecodingEdge(1, 3),
                    DecodingEdge(9, 0),
                    DecodingEdge(1, 2),
                ],
                mod_2_filter=False,
            ),
        ]
    )
    def ordered_decoding_edges(self, request) -> OrderedDecodingEdges:
        return request.param

    def test_decoding_edges_equality_is_dependent_on_order(
        self, ordered_decoding_edges
    ):
        if len(ordered_decoding_edges) > 1:
            assert ordered_decoding_edges != OrderedDecodingEdges(
                reversed(ordered_decoding_edges)
            )

    def test_decoding_edges_hash_is_dependent_on_order(self, ordered_decoding_edges):
        if len(ordered_decoding_edges) > 1:
            assert hash(ordered_decoding_edges) != hash(
                OrderedDecodingEdges(reversed(ordered_decoding_edges))
            )

    def test_decoding_edges_items_are_unique(self, ordered_decoding_edges):
        assert len(set(ordered_decoding_edges)) == len(ordered_decoding_edges)

    def test_edges_used_to_construct_decoding_edges_are_contained(self):
        decoding_edge_a = DecodingEdge(3, 4)
        decoding_edge_b = DecodingEdge(6, 1)
        ordered_edges = OrderedDecodingEdges([decoding_edge_a, decoding_edge_b])
        assert decoding_edge_a in ordered_edges and decoding_edge_b in ordered_edges

    def test_string_of_each_decoding_edge_is_in_string_of_ordered_collection(self):
        decoding_edge_a = DecodingEdge(3, 4)
        decoding_edge_b = DecodingEdge(6, 8)
        ordered_edges_string = str(
            OrderedDecodingEdges([decoding_edge_a, decoding_edge_b])
        )
        assert (
            str(decoding_edge_a) in ordered_edges_string
            and str(decoding_edge_b) in ordered_edges_string
        )

    @pytest.mark.parametrize(
        "edges",
        [
            [],
            [
                DecodingEdge(9, 0),
                DecodingEdge(1, 3),
                DecodingEdge(9, 0),
                DecodingEdge(1, 2),
            ],
            [DecodingEdge(1, 0), DecodingEdge(1, 0), DecodingEdge(1, 0)],
            [
                DecodingEdge(1, 0),
                DecodingEdge(1, 0),
                DecodingEdge(1, 2),
                DecodingEdge(1, 2),
            ],
        ],
    )
    def test_mod_2_disabled_construction_only_removes_duplicates(self, edges):
        ordered_edges = OrderedDecodingEdges(edges, mod_2_filter=False)
        assert set(ordered_edges) == set(edges)
        assert len(ordered_edges) == len(set(edges))

    @pytest.mark.parametrize(
        "ordered_decoding_edges, edges, expected_bitstring",
        [
            (OrderedDecodingEdges([]), [], []),
            (
                OrderedDecodingEdges(
                    [
                        DecodingEdge(9, 0),
                        DecodingEdge(1, 3),
                        DecodingEdge(9, 0),
                        DecodingEdge(1, 2),
                    ]
                ),
                [DecodingEdge(1, 3)],
                [1],
            ),
            (
                OrderedDecodingEdges(
                    [DecodingEdge(0, 1), DecodingEdge(1, 0), DecodingEdge(1, 0)]
                ),
                [DecodingEdge(0, 1), DecodingEdge(1, 0)],
                [1, 1],
            ),
            (
                OrderedDecodingEdges(
                    [
                        DecodingEdge(1, 0),
                        DecodingEdge(2, 3),
                        DecodingEdge(3, 1),
                        DecodingEdge(1, 2),
                    ]
                ),
                [DecodingEdge(0, 3), DecodingEdge(3, 1)],
                [0, 1],
            ),
        ],
    )
    def test_as_bitstring(
        self,
        ordered_decoding_edges: OrderedDecodingEdges,
        edges: List[DecodingEdge],
        expected_bitstring: List[Bit],
    ):
        assert ordered_decoding_edges.as_bitstring(edges) == expected_bitstring

    @pytest.mark.parametrize(
        "indices, expected_decoding_edges",
        [
            ([], OrderedDecodingEdges()),
            (
                [(0, 1), (1, 2)],
                OrderedDecodingEdges([DecodingEdge(0, 1), DecodingEdge(1, 2)]),
            ),
            (
                [(2, 3), (5, 6)],
                OrderedDecodingEdges([DecodingEdge(2, 3), DecodingEdge(5, 6)]),
            ),
            ([(0, 1), (0, 1)], OrderedDecodingEdges()),
        ],
    )
    def test_construct_decoding_edges_from_syndrome_indices(
        self,
        indices: Iterable[Tuple[int, int]],
        expected_decoding_edges: OrderedDecodingEdges,
    ):
        assert (
            OrderedDecodingEdges.from_syndrome_indices(indices)
            == expected_decoding_edges
        )

    def test_append_ordered_decoding_edges(self):
        edges = OrderedDecodingEdges([DecodingEdge(0, 1)])
        other_edges = OrderedDecodingEdges()
        other_edges.append(edges)
        assert other_edges == edges

    def test_append_ordered_decoding_edges_allows_repeated_edges(self):
        edges = OrderedDecodingEdges([DecodingEdge(0, 1), DecodingEdge(1, 2)])
        other_edges = OrderedDecodingEdges([DecodingEdge(1, 2)])
        other_edges.append(edges, mod_2_filter=False)
        ref = OrderedDecodingEdges(
            [DecodingEdge(1, 2), DecodingEdge(0, 1), DecodingEdge(1, 2)],
            mod_2_filter=False,
        )
        assert other_edges == ref

    def test_append_ordered_decoding_edges_applies_mod_2_filter(self):
        edges = OrderedDecodingEdges([DecodingEdge(0, 1), DecodingEdge(1, 2)])
        other_edges = OrderedDecodingEdges([DecodingEdge(1, 2)])
        other_edges.append(edges, mod_2_filter=True)
        ref = OrderedDecodingEdges([DecodingEdge(0, 1)])
        assert other_edges == ref

    def test_add_empty_ordered_decoding_edges_is_identity(self):
        nonempty_ode = OrderedDecodingEdges([DecodingEdge(0, 1), DecodingEdge(1, 2)])
        assert OrderedDecodingEdges() + nonempty_ode == OrderedDecodingEdges(
            [DecodingEdge(0, 1), DecodingEdge(1, 2)]
        )
        assert nonempty_ode + OrderedDecodingEdges() == OrderedDecodingEdges(
            [DecodingEdge(0, 1), DecodingEdge(1, 2)]
        )

    def test_add_ordered_decoding_edges_does_not_mutate_operands(self):
        left_ode = OrderedDecodingEdges([DecodingEdge(0, 1)])
        right_ode = OrderedDecodingEdges([DecodingEdge(1, 2)])
        left_ode + right_ode

        assert left_ode == OrderedDecodingEdges([DecodingEdge(0, 1)])
        assert right_ode == OrderedDecodingEdges([DecodingEdge(1, 2)])

    def test_add_ordered_decoding_edges_does_not_apply_mod_2_filter(self):
        other_edges = OrderedDecodingEdges([DecodingEdge(0, 1), DecodingEdge(1, 2)])
        edges = OrderedDecodingEdges([DecodingEdge(1, 2)])
        edges = edges + other_edges
        ref = OrderedDecodingEdges(
            [DecodingEdge(1, 2), DecodingEdge(0, 1), DecodingEdge(1, 2)],
            mod_2_filter=False,
        )
        assert edges == ref


@pytest.mark.parametrize(
    "errors, boundaries, expected_syndrome",
    [
        ([], set(), OrderedSyndrome()),
        ([], set([1]), OrderedSyndrome()),
        ([DecodingEdge(2, 0)], set(), OrderedSyndrome([0, 2])),
        (
            [
                DecodingEdge(9, 0),
                DecodingEdge(1, 3),
                DecodingEdge(9, 0),
                DecodingEdge(1, 2),
            ],
            set(),
            OrderedSyndrome([3, 2]),
        ),
        (
            [
                DecodingEdge(1, 0),
                DecodingEdge(1, 0),
                DecodingEdge(1, 2),
                DecodingEdge(1, 2),
            ],
            set(),
            OrderedSyndrome(),
        ),
        (
            OrderedDecodingEdges(
                [
                    DecodingEdge(1, 0),
                    DecodingEdge(3, 9),
                    DecodingEdge(3, 9),
                    DecodingEdge(1, 2),
                ]
            ),
            set(),
            OrderedSyndrome([0, 2]),
        ),
        (
            [DecodingEdge(9, 0), DecodingEdge(1, 3), DecodingEdge(1, 2)],
            set([0]),
            OrderedSyndrome([9, 3, 2]),
        ),
        (
            OrderedDecodingEdges(
                [DecodingEdge(1, 0), DecodingEdge(3, 9), DecodingEdge(1, 2)]
            ),
            set([1, 9]),
            OrderedSyndrome([0, 3, 2]),
        ),
        (
            [DecodingHyperEdge([1, 0, 2]), DecodingHyperEdge([1, 3])],
            set(),
            OrderedSyndrome([0, 2, 3]),
        ),
        (
            OrderedDecodingEdges(
                [DecodingHyperEdge([1, 0, 2]), DecodingEdge(3, 9), DecodingEdge(1, 2)]
            ),
            set(),
            OrderedSyndrome([0, 9, 3]),
        ),
        (
            OrderedDecodingEdges(
                [DecodingHyperEdge([1, 0, 2]), DecodingHyperEdge([1, 2, 3])]
            ),
            set([1]),
            OrderedSyndrome([0, 3]),
        ),
        (
            OrderedDecodingEdges(
                [DecodingHyperEdge([0, 1, 2]), DecodingEdge(3, 9), DecodingEdge(1, 2)]
            ),
            set([2, 9]),
            OrderedSyndrome([0, 3]),
        ),
    ],
)
def test_errors_to_syndrome(
    errors: list[EdgeT],
    boundaries: AbstractSet[int],
    expected_syndrome: OrderedSyndrome,
):
    assert errors_to_syndrome(errors, boundaries) == expected_syndrome
