# (c) Copyright Riverlane 2020-2025.
"""Tests for decoding syndrome datastructures."""

from collections.abc import Sequence

import numpy as np
import pytest
from deltakit_core.decoding_graphs import (
    Bitstring,
    DetectorRecord,
    FixedWidthBitstring,
    OrderedSyndrome,
    get_round_words,
)
from deltakit_core.decoding_graphs._syndromes import Bit


class TestOrderedSyndrome:
    """Test for ordered collection of syndromes."""

    @pytest.fixture(
        params=[
            OrderedSyndrome(),
            OrderedSyndrome([4]),
            OrderedSyndrome([0, 4, 5, 1, 3, 8]),
            OrderedSyndrome(range(100)),
            OrderedSyndrome(range(55, 25, -1)),
            OrderedSyndrome((5, 5, 2, 1, 333, 5, 1, 9, 0)),
        ]
    )
    def ordered_syndrome(self, request) -> OrderedSyndrome:
        return request.param

    def test_syndrome_equality_is_dependent_on_order(self, ordered_syndrome):
        if len(ordered_syndrome) > 1:
            assert ordered_syndrome != OrderedSyndrome(reversed(ordered_syndrome))

    def test_syndrome_hash_is_dependent_on_order(self, ordered_syndrome):
        if len(ordered_syndrome) > 1:
            assert hash(ordered_syndrome) != hash(
                OrderedSyndrome(reversed(ordered_syndrome))
            )

    def test_syndrome_items_are_unique(self, ordered_syndrome):
        assert len(set(ordered_syndrome)) == len(ordered_syndrome)

    def test_syndrome_bits_used_to_construct_decoding_edges_are_contained(self):
        syndrome_bit_a = 3
        syndrome_bit_b = 4
        syndrome = OrderedSyndrome([syndrome_bit_a, syndrome_bit_b])
        assert syndrome_bit_a in syndrome and syndrome_bit_b in syndrome

    def test_string_of_each_syndrome_bit_is_in_string_of_ordered_syndrome(self):
        syndrome_bit_a = 11
        syndrome_bit_b = 9
        syndrome_string = str(OrderedSyndrome([syndrome_bit_a, syndrome_bit_b]))
        assert (
            str(syndrome_bit_a) in syndrome_string
            and str(syndrome_bit_b) in syndrome_string
        )

    @pytest.mark.parametrize(
        "annotated_syndrome, detector_records, layer_num, layers",
        [
            (
                OrderedSyndrome([0, 1, 3]),
                [
                    DetectorRecord((4,), 5),
                    DetectorRecord((0, 5)),
                    DetectorRecord((0, 5)),
                    DetectorRecord((2, 4, 6)),
                ],
                6,
                [
                    OrderedSyndrome([1, 3]),
                    OrderedSyndrome(),
                    OrderedSyndrome(),
                    OrderedSyndrome(),
                    OrderedSyndrome(),
                    OrderedSyndrome([0]),
                ],
            ),
            (OrderedSyndrome(), [], 2, [OrderedSyndrome(), OrderedSyndrome()]),
        ],
    )
    def test_example_split_by_time_coord_is_expected_result(
        self,
        annotated_syndrome: OrderedSyndrome,
        detector_records,
        layer_num: int,
        layers: list[OrderedSyndrome],
    ):
        assert (
            annotated_syndrome.split_by_time_coord(detector_records, layer_num)
            == layers
        )

    @pytest.mark.parametrize(
        "layers, layer_size, expected_ordered_syndrome",
        [
            ([], 10, OrderedSyndrome([])),
            ([[4, 5], [0], [0], [1]], 6, OrderedSyndrome([4, 5, 6, 12, 19])),
            ([[0, 0, 1]], 2, OrderedSyndrome([1])),
            ([[], [1], [], [1, 4]], 10, OrderedSyndrome([11, 31, 34])),
            ([[], [], [], [9]], 10, OrderedSyndrome([39])),
        ],
    )
    def test_syndrome_from_layers_gives_expected_ordered_syndrome(
        self, layers, layer_size, expected_ordered_syndrome
    ):
        assert (
            OrderedSyndrome.from_layers(layers, layer_size) == expected_ordered_syndrome
        )

    @pytest.mark.parametrize(
        "layers, layer_size, expected_ordered_syndrome",
        [
            ([], [1, 2], OrderedSyndrome([])),
            ([[4, 5], [0], [0], [1]], [6, 3, 4, 5], OrderedSyndrome([4, 5, 6, 9, 14])),
            ([[0, 0, 1]], [3], OrderedSyndrome([1])),
            ([[], [1], [], [1, 4]], [1, 2, 3, 5], OrderedSyndrome([2, 7, 10])),
            ([[], [], [], [9]], [1, 2, 3, 10], OrderedSyndrome([15])),
        ],
    )
    def test_syndrome_from_layers_with_variable_layer_size_gives_expected_ordered_syndrome(
        self, layers, layer_size, expected_ordered_syndrome
    ):
        assert (
            OrderedSyndrome.from_layers(layers, layer_size) == expected_ordered_syndrome
        )

    @pytest.mark.parametrize(
        "syndrome, layer_size, expected_layers",
        [
            (OrderedSyndrome([0, 6, 2, 1, 1, 11, 19]), 4, [[0, 2], [2], [3], [], [3]]),
            (OrderedSyndrome(), 100, []),
            (OrderedSyndrome(range(11)), 5, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0]]),
            (OrderedSyndrome(range(0, 20, 3)), 20, [list(range(0, 20, 3))]),
        ],
    )
    def test_example_as_layers_is_expected_result(
        self, syndrome, layer_size, expected_layers
    ):
        assert syndrome.as_layers(layer_size) == expected_layers

    @pytest.mark.parametrize(
        "syndrome, layer_size, expected_layers",
        [
            (
                OrderedSyndrome([0, 6, 2, 1, 1, 11, 19]),
                [1, 2, 3, 4, 5, 6],
                [[0], [1], [], [0], [1], [4]],
            ),
            (OrderedSyndrome(), [1, 2, 3], [[], [], []]),
            (
                OrderedSyndrome(range(11)),
                [6, 5, 4, 3, 2, 1],
                [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [], [], [], []],
            ),
            (OrderedSyndrome(range(0, 20, 3)), [20, 20], [list(range(0, 20, 3)), []]),
        ],
    )
    def test_example_as_layers_is_expected_result_with_variable_layer_size(
        self, syndrome, layer_size, expected_layers
    ):
        assert syndrome.as_layers(layer_size) == expected_layers

    @pytest.mark.parametrize(
        "syndrome, layer_size",
        [
            (OrderedSyndrome([0, 2, 6, 11, 19]), 3),
            (OrderedSyndrome([0, 2, 6, 11, 19]), 12),
            (OrderedSyndrome(), 4),
            (OrderedSyndrome(range(11)), 2),
            (OrderedSyndrome(range(0, 20, 3)), 6),
        ],
    )
    def test_as_layers_compose_from_layers_is_identity_for_sorted_syndrome(
        self, syndrome, layer_size
    ):
        assert (
            OrderedSyndrome.from_layers(syndrome.as_layers(layer_size), layer_size)
            == syndrome
        )

    @pytest.mark.parametrize("bitstring", [[0, 1, 1], np.array([0, 1, 1])])
    def test_detector_in_syndrome_created_from_bitstring_are_python_types(
        self, bitstring: Sequence[Bit]
    ):
        for detector in OrderedSyndrome.from_bitstring(bitstring):
            assert isinstance(detector, int)

    @pytest.mark.parametrize(
        "syndrome", [OrderedSyndrome([0, 2, 3]), OrderedSyndrome(np.array([0, 2, 3]))]
    )
    def test_bits_in_bitstring_from_syndrome_are_python_types(
        self, syndrome: OrderedSyndrome
    ):
        for bit in syndrome.as_bitstring(4):
            assert isinstance(bit, int)

    def test_detector_that_appears_even_number_of_times_not_in_syndrome_by_default(
        self,
    ):
        assert 1 not in OrderedSyndrome([1, 1])

    @pytest.mark.parametrize("detectors", [[1], [1] * 2, [1] * 3])
    def test_each_detector_appears_once_without_enforcing_mod_2(
        self, detectors: Sequence[int]
    ):
        syndrome = OrderedSyndrome(detectors, enforce_mod_2=False)
        assert tuple(syndrome) == (detectors[0],)

    def test_all_detectors_appear_once_in_correct_order_when_not_enforcing_mod_2(self):
        syndrome = OrderedSyndrome([1, 2, 1, 3], enforce_mod_2=False)
        assert tuple(syndrome) == (1, 2, 3)

    @pytest.mark.parametrize(
        "syndrome, max_detector_symptom, expected_detector_events, expected_herald_events",
        [
            (OrderedSyndrome([0, 6, 2, 1, 1, 11, 11, 19]), 6, [0, 2, 6], [12]),
            (OrderedSyndrome(), 5, [], []),
            (OrderedSyndrome(range(11)), 0, [0], list(range(10))),
            (OrderedSyndrome(range(0, 20, 3)), 10, [0, 3, 6, 9], [1, 4, 7]),
        ],
    )
    def test_example_split_at_symptom_is_expected_result(
        self,
        syndrome,
        max_detector_symptom,
        expected_detector_events,
        expected_herald_events,
    ):
        det_events, herald_flags = syndrome.split_at_symptom(max_detector_symptom)
        assert sorted(det_events) == sorted(expected_detector_events)
        assert herald_flags == expected_herald_events


class TestBitstringCreation:
    def test_error_is_raised_bitstring_given_negative_number(self):
        with pytest.raises(ValueError, match=r"Bitstring cannot be a negative value."):
            Bitstring(-1)

    @pytest.mark.parametrize(
        "bitstring, expected_bits",
        [
            (Bitstring(0), [0]),
            (Bitstring(1), [1]),
            (Bitstring(2), [0, 1]),
            (Bitstring(3), [1, 1]),
        ],
    )
    def test_iterating_a_bitstring_gives_expected_bits(
        self, bitstring: Bitstring, expected_bits: list[Bit]
    ):
        assert list(bitstring) == expected_bits

    @pytest.mark.parametrize(
        "indices, expected_bitstring",
        [
            ([], Bitstring(0)),
            ([0], Bitstring(1)),
            ([1], Bitstring(2)),
            ([2, 3], Bitstring(0b1100)),
        ],
    )
    def test_creating_bitstring_from_indices_gives_expected_bitstring(
        self, indices: list[int], expected_bitstring: Bitstring
    ):
        assert Bitstring.from_indices(indices) == expected_bitstring

    @pytest.mark.parametrize(
        "bits, expected_bitstring",
        [
            ([], Bitstring(0)),
            ([0], Bitstring(0)),
            ([1], Bitstring(1)),
            ([1, 0], Bitstring(0b01)),
            ([0, 1], Bitstring(0b10)),
        ],
    )
    def test_creating_bitstring_from_bits_gives_expected_bitstring(
        self, bits: list[Bit], expected_bitstring: Bitstring
    ):
        assert Bitstring.from_bits(bits) == expected_bitstring

    @pytest.mark.parametrize(
        "_bytes, expected_bitstring",
        [
            ([], Bitstring(0)),
            (np.array([0, 255], dtype=np.uint8), Bitstring(65280)),
            (np.array([2, 124], dtype=np.uint8), Bitstring(31746)),
            (np.array([255], dtype=np.uint8), Bitstring(255)),
            (bytes([16]), Bitstring(16)),
        ],
    )
    def test_creating_bitstring_from_bytes_gives_expected_bitstring(
        self, _bytes, expected_bitstring: Bitstring
    ):
        assert Bitstring.from_bytes(_bytes) == expected_bitstring


class TestBitstringBitwiseMethods:
    def test_oring_two_bitstrings_together_gives_expected_bitstring(self):
        assert Bitstring(0b0101) | Bitstring(0b1100) == Bitstring(0b1101)

    def test_oring_two_bitstrings_inplace_gives_expected_bitstring(self):
        bitstring = Bitstring(0b1001)
        bitstring |= Bitstring(0b1011)
        assert bitstring == Bitstring(0b1011)

    def test_anding_two_bitstrings_together_gives_expected_bitstring(self):
        assert Bitstring(0b11) & Bitstring(0b10) == Bitstring(0b10)

    def test_anding_two_bitstrings_inplace_gives_expected_bitstring(self):
        bitstring = Bitstring(0b11)
        bitstring &= Bitstring(0b01)
        assert bitstring == Bitstring(0b01)

    def test_xoring_two_bitstrings_together_gives_expected_bitstring(self):
        assert Bitstring(0b11) ^ Bitstring(0b01) == Bitstring(0b10)

    def test_xoring_two_bitstrings_inplace_gives_expected_bitstring(self):
        bitstring = Bitstring(0b11)
        bitstring ^= Bitstring(0b01)
        assert bitstring == Bitstring(0b10)

    def test_left_shifting_bitstring_by_an_int_returns_expected_bitstring(self):
        assert Bitstring(0b11) << 3 == Bitstring(0b11000)

    def test_right_shifting_bitstring_by_int_gives_expected_bitstring(self):
        assert Bitstring(0b1100101011) >> 4 == Bitstring(0b110010)


class TestBitstringDunderMethods:
    @pytest.mark.parametrize(
        "bad_index",
        [
            "bad",
            0.0,
        ],
    )
    def test_error_is_raised_if_index_is_not_int_or_slice(self, bad_index):
        bitstring = Bitstring(220394)
        with pytest.raises(
            TypeError, match="Bitstring indices must be " r"integers or slices, not.*"
        ):
            bitstring[bad_index]

    def test_iterating_on_zero_bitstring_returns_zero(self):
        assert list(Bitstring()) == [0]

    def test_iterating_through_bitstring_returns_bits_in_little_endian(self):
        assert list(Bitstring(0b1010011)) == [1, 1, 0, 0, 1, 0, 1]

    def test_iterating_on_zero_bitstring_in_reverse_returns_zero(self):
        assert list(reversed(Bitstring())) == [0]

    def test_iterating_though_bitstring_in_reversed_returns_bits_in_big_endian(self):
        assert list(reversed(Bitstring(0b1010011))) == [1, 0, 1, 0, 0, 1, 1]

    @pytest.mark.parametrize(
        "bitstring, expected_length",
        [
            (Bitstring(), 1),
            (Bitstring(1), 1),
            (Bitstring(2), 2),
            (Bitstring(3), 2),
            (Bitstring(0b10010110), 8),
            (Bitstring(0b00101), 3),
        ],
    )
    def test_length_of_bitstring_is_correct_length(
        self, bitstring: Bitstring, expected_length: int
    ):
        assert len(bitstring) == expected_length

    def test_indexing_with_an_int_returns_the_single_bit(self):
        indices = [0, 3, 4, 6]
        bitstring = Bitstring.from_indices(indices)
        for index in range(len(bitstring)):
            assert bitstring[index] == (1 if index in indices else 0)

    def test_slicing_bitstring_with_start_and_stop_returns_a_bitstring(self):
        bitstring = Bitstring(0b10010101101010)
        assert bitstring[0:4] == Bitstring(0b1010)
        assert bitstring[1:5] == Bitstring(0b101)
        assert bitstring[3:9] == Bitstring(0b101101)

    def test_slicing_with_only_starts_returns_from_start_until_end(self):
        bitstring = Bitstring(0b1110100101001110)
        assert bitstring[3:] == Bitstring(0b1110100101001)
        assert bitstring[6:] == Bitstring(0b1110100101)

    def test_slicing_with_only_end_returns_start_until_the_end(self):
        bitstring = Bitstring(0b1001001101010001)
        assert bitstring[:6] == Bitstring(0b010001)
        assert bitstring[:4] == Bitstring(0b0001)

    def test_interpreting_bitstring_returns_correct_integer(self):
        int_ = 0b1010100010110100101
        assert int(Bitstring(int_)) == int_

    def test_mutating_int_representation_of_bitstring_does_not_change_bitstring(self):
        bitstring = Bitstring(0b1010100101101)
        int_ = int(bitstring)
        int_ += 20
        assert bitstring == Bitstring(0b1010100101101)

    @pytest.mark.parametrize(
        "bitstring, expected_binary_string",
        [
            (Bitstring(), "0b0"),
            (Bitstring(1), "0b1"),
            (Bitstring(0b10010110), "0b10010110"),
            (Bitstring(0b00101), "0b101"),
        ],
    )
    def test_bit_string_repr_is_binary_string(self, bitstring, expected_binary_string):
        assert repr(bitstring) == expected_binary_string

    @pytest.mark.parametrize(
        "badarg, error_msg",
        [
            (
                [0, 1],
                "Bitstring indices must be integers or slices, not <class 'list'>",
            ),
            (
                "first_bit",
                "Bitstring indices must be integers or slices, not <class 'str'>",
            ),
            (0.5, "Bitstring indices must be integers or slices, not <class 'float'>"),
        ],
    )
    def test_invalid_indexing_arg_type_raises_type_error(self, badarg, error_msg):
        bitstring = Bitstring(0b10101001101)
        with pytest.raises(TypeError, match=error_msg):
            bitstring[badarg]


class TestBitstringMethods:
    @pytest.mark.parametrize(
        "bitstring, expected_bit_count",
        [
            (Bitstring(0), 0),
            (Bitstring(1), 1),
            (Bitstring(0b11), 2),
            (Bitstring(0b1011011), 5),
        ],
    )
    def test_bit_count_correctly_returns_the_expected_number_of_ones(
        self, bitstring: Bitstring, expected_bit_count: int
    ):
        assert bitstring.bit_count() == expected_bit_count

    @pytest.mark.parametrize(
        "bitstring, expected_indices",
        [
            (Bitstring(0), []),
            (Bitstring(1), [0]),
            (Bitstring(0b11), [0, 1]),
            (Bitstring(0b1101010111), [0, 1, 2, 4, 6, 8, 9]),
        ],
    )
    def test_converting_bitstring_into_list_of_indices_gives_expected_indices(
        self, bitstring: Bitstring, expected_indices: list[int]
    ):
        assert bitstring.to_indices() == expected_indices

    def test_bitstring_as_indices_has_same_length_as_bit_count(self):
        bitstring = Bitstring(0b100101001010011010010100101)
        assert len(bitstring.to_indices()) == bitstring.bit_count()

    def test_converting_bitstring_to_indices_and_back_is_the_same_bitstring(self):
        bitstring = Bitstring(0b10100110101)
        assert Bitstring.from_indices(bitstring.to_indices()) == bitstring

    @pytest.mark.parametrize(
        "bitstring, num_bits_per_word, expected_words",
        [
            (Bitstring(), 1, [Bitstring()]),
            (Bitstring(1), 1, [Bitstring(1)]),
            (Bitstring(0b10), 1, [Bitstring(0), Bitstring(0b1)]),
            (Bitstring(0b11), 1, [Bitstring(0b1), Bitstring(0b1)]),
            (
                Bitstring(0b1001011001),
                2,
                [
                    Bitstring(0b01),
                    Bitstring(0b10),
                    Bitstring(0b01),
                    Bitstring(0b01),
                    Bitstring(0b10),
                ],
            ),
        ],
    )
    def test_converting_bitstring_to_words_gives_something(
        self,
        bitstring: Bitstring,
        num_bits_per_word: int,
        expected_words: list[Bitstring],
    ):
        assert list(bitstring.to_words(num_bits_per_word)) == expected_words


class TestFixedWidthBitstringCreation:
    @pytest.mark.parametrize("bad_width", (0, -1))
    def test_error_is_raised_if_width_is_less_than_one(self, bad_width: int):
        with pytest.raises(
            ValueError, match="Width of bitstring must be greater than zero."
        ):
            FixedWidthBitstring(bad_width)

    def test_creating_from_indices_has_width_of_max_index(self):
        assert len(FixedWidthBitstring.from_indices([3, 5, 20])) == 21

    def test_creating_from_indices_gives_correct_bitstring(self):
        assert FixedWidthBitstring.from_indices([2, 4, 5]) == FixedWidthBitstring(
            6, 0b110100
        )

    def test_creating_from_bits_has_width_the_same_length_of_iterable(self):
        assert len(FixedWidthBitstring.from_bits([1, 1, 0, 1, 0])) == 5

    def test_creating_from_bits_gives_correct_bitstring(self):
        assert FixedWidthBitstring.from_bits(
            [1, 0, 1, 1, 0, 1, 0, 1]
        ) == FixedWidthBitstring(8, 0b10101101)

    def test_creating_from_bytes_has_width_same_as_bits_in_given_bytes(self):
        assert (
            len(
                FixedWidthBitstring.from_bytes(
                    np.array([123, 144, 0, 1], dtype=np.uint8)
                )
            )
            == 32
        )

    def test_creating_from_bytes_gives_correct_bitstring(self):
        assert FixedWidthBitstring.from_bytes(
            np.array([2, 128], dtype=np.uint8)
        ) == FixedWidthBitstring(16, 0b1000000000000010)

    def test_creating_from_empty_raises_width_error(self):
        with pytest.raises(
            ValueError, match="Width of bitstring must be greater than zero."
        ):
            FixedWidthBitstring.from_bits(())


class TestFixedWidthBitstringDunderMethods:
    @pytest.mark.parametrize("initial_value", (0, 0b1, 0b1111))
    def test_length_of_bitstring_list_is_same_as_width_of_bitstring(
        self, initial_value: int
    ):
        bitstring = FixedWidthBitstring(3, initial_value)
        assert len(list(bitstring)) == len(bitstring)

    def test_iterating_through_bitstring_gives_expected_bits_and_then_zeros(self):
        bitstring = FixedWidthBitstring(3, 0b11)
        assert list(bitstring) == [1, 1, 0]

    def test_only_bits_less_than_width_are_kept_in_iteration(self):
        bitstring = FixedWidthBitstring(3, 0b1010111)
        assert list(bitstring) == [1, 1, 1]

    def test_reversed_bitstring_gives_bits_in_little_endian(self):
        bitstring = FixedWidthBitstring(3, 0b110)
        assert list(reversed(bitstring)) == [1, 1, 0]

    def test_reversed_bitstring_only_shows_bits_in_width(self):
        bitstring = FixedWidthBitstring(3, 0b10101001)
        assert list(reversed(bitstring)) == [0, 0, 1]

    def test_inverting_bitstring_with_value_as_wide_as_width_gives__inverted_value(
        self,
    ):
        assert ~FixedWidthBitstring(4, 0b1001) == FixedWidthBitstring(4, 0b0110)

    def test_inverting_bitstring_with_value_less_than_width_adds_width_ones(self):
        assert ~FixedWidthBitstring(4, 0b1) == FixedWidthBitstring(4, 0b1110)

    @pytest.mark.parametrize("bad_index", ["bad", 0.1])
    def test_error_is_raised_if_index_is_not_int_or_slice(self, bad_index):
        bitstring = Bitstring(220395)
        with pytest.raises(
            TypeError, match="Bitstring indices must be " r"integers or slices, not.*"
        ):
            bitstring[bad_index]

    def test_indexing_with_an_int_returns_the_single_bit(self):
        indices = [0, 4, 6, 8]
        bitstring = FixedWidthBitstring.from_indices(indices)
        for index in range(len(bitstring)):
            assert bitstring[index] == (1 if index in indices else 0)

    def test_slicing_bitstring_with_start_and_stop_returns_a_bistring(self):
        bitstring = FixedWidthBitstring(width=10, value=0b0101000101)
        assert bitstring[0:4] == FixedWidthBitstring(width=4, value=0b0101)
        assert bitstring[1:5] == FixedWidthBitstring(width=4, value=0b0010)
        assert bitstring[3:9] == FixedWidthBitstring(width=6, value=0b0101000)

    def test_slicing_bitstring_out_of_range_is_clipped(self):
        bitstring = FixedWidthBitstring(width=4, value=0b0110)
        assert bitstring[0:6] == FixedWidthBitstring(width=4, value=0b0110)


class TestFixedWidthBitstringMethods:
    def test_converting_to_words_of_fewer_bits_gives_fixed_width_bitstrings(self):
        bitstring = FixedWidthBitstring(4, 0b1001)
        bitstring_words = list(bitstring.to_words(2))
        assert bitstring_words == [
            FixedWidthBitstring(2, 0b01),
            FixedWidthBitstring(2, 0b10),
        ]
        assert len(bitstring_words[0]) == 2
        assert len(bitstring_words[1]) == 2

    def test_converting_to_words_of_more_bits_gives_wider_fixed_width_bitstring(self):
        bitstring = FixedWidthBitstring(4, 0b1101)
        bitstring_words = list(bitstring.to_words(6))
        assert bitstring_words == [FixedWidthBitstring(6, int(bitstring))]
        assert len(bitstring_words[0]) == 6

    def test_converting_to_words_with_upper_bits_as_zero_gives_correct_words(self):
        bitstring = FixedWidthBitstring(7, 0b101)
        assert list(bitstring.to_words(2)) == [
            FixedWidthBitstring(2, 0b01),
            FixedWidthBitstring(2, 0b01),
            FixedWidthBitstring(2, 0b00),
            FixedWidthBitstring(2, 0b00),
        ]

    def test_making_width_of_bitstring_smaller_removes_bits_over_new_width(self):
        bitstring = FixedWidthBitstring(5, 0b10110)
        bitstring.change_width(2)
        assert bitstring == FixedWidthBitstring(2, 0b10)

    def test_making_bitstring_width_bigger_increases_the_length_of_the_bitstring(self):
        bitstring = FixedWidthBitstring(4, 0b1001)
        bitstring.change_width(6)
        assert len(bitstring) == 6

    def test_fixed_width_bitstring_concatenation(self):
        bitstring = FixedWidthBitstring(4, 0b0110) + FixedWidthBitstring(4, 0b0110)
        assert bitstring == FixedWidthBitstring(8, 0b01100110)


class TestFixedWidthBitstringArithmetic:
    @pytest.mark.parametrize(
        "left, right, expected_bitstring",
        [
            (
                FixedWidthBitstring(3, 0b001),
                FixedWidthBitstring(3, 0b100),
                FixedWidthBitstring(3, 0b101),
            ),
            (
                FixedWidthBitstring(4, 0b1001),
                FixedWidthBitstring(2, 0b10),
                FixedWidthBitstring(4, 0b1011),
            ),
            (
                FixedWidthBitstring(3, 0b100),
                FixedWidthBitstring(5, 0b10010),
                FixedWidthBitstring(3, 0b110),
            ),
            (
                FixedWidthBitstring(5, 0b10011),
                Bitstring(0b11000101001),
                FixedWidthBitstring(5, 0b11011),
            ),
        ],
    )
    def test_oring_together_different_width_bitstrings_takes_left_hand_width(
        self,
        left: FixedWidthBitstring,
        right: FixedWidthBitstring,
        expected_bitstring: FixedWidthBitstring,
    ):
        actual = left | right
        assert actual == expected_bitstring
        assert len(actual) == len(left)

    @pytest.mark.parametrize(
        "left, right, expected_bitstring",
        [
            (
                FixedWidthBitstring(3, 0b001),
                FixedWidthBitstring(3, 0b100),
                FixedWidthBitstring(3, 0b101),
            ),
            (
                FixedWidthBitstring(4, 0b1001),
                FixedWidthBitstring(2, 0b10),
                FixedWidthBitstring(4, 0b1011),
            ),
            (
                FixedWidthBitstring(3, 0b100),
                FixedWidthBitstring(5, 0b10010),
                FixedWidthBitstring(3, 0b110),
            ),
            (
                FixedWidthBitstring(5, 0b10011),
                Bitstring(0b11000101001),
                FixedWidthBitstring(5, 0b11011),
            ),
        ],
    )
    def test_oring_inplace_does_mutates_internal_state_in_expected_way(
        self,
        left: FixedWidthBitstring,
        right: Bitstring,
        expected_bitstring: FixedWidthBitstring,
    ):
        bitstring = left
        bitstring |= right
        assert bitstring == expected_bitstring
        assert len(bitstring) == len(left)

    @pytest.mark.parametrize(
        "left, right, expected_bitstring",
        [
            (
                FixedWidthBitstring(3, 0b001),
                FixedWidthBitstring(3, 0b011),
                FixedWidthBitstring(3, 0b010),
            ),
            (
                FixedWidthBitstring(4, 0b0011),
                FixedWidthBitstring(6, 0b111001),
                FixedWidthBitstring(4, 0b1010),
            ),
            (
                FixedWidthBitstring(4, 0b1011),
                FixedWidthBitstring(2, 0b10),
                FixedWidthBitstring(4, 0b1001),
            ),
            (
                FixedWidthBitstring(5, 0b10110),
                Bitstring(0b11011011),
                FixedWidthBitstring(5, 0b01101),
            ),
        ],
    )
    def test_xoring_together_different_width_bitstrings_takes_left_hand_width(
        self,
        left: FixedWidthBitstring,
        right: Bitstring,
        expected_bitstring: FixedWidthBitstring,
    ):
        actual = left ^ right
        assert actual == expected_bitstring
        assert len(actual) == len(left)

    @pytest.mark.parametrize(
        "left, right, expected_bitstring",
        [
            (
                FixedWidthBitstring(3, 0b001),
                FixedWidthBitstring(3, 0b011),
                FixedWidthBitstring(3, 0b010),
            ),
            (
                FixedWidthBitstring(4, 0b0011),
                FixedWidthBitstring(6, 0b111001),
                FixedWidthBitstring(4, 0b1010),
            ),
            (
                FixedWidthBitstring(4, 0b1011),
                FixedWidthBitstring(2, 0b10),
                FixedWidthBitstring(4, 0b1001),
            ),
            (
                FixedWidthBitstring(5, 0b10110),
                Bitstring(0b11011011),
                FixedWidthBitstring(5, 0b01101),
            ),
        ],
    )
    def test_xoring_inplace_does_mutates_internal_state_in_expected_way(
        self,
        left: FixedWidthBitstring,
        right: Bitstring,
        expected_bitstring: FixedWidthBitstring,
    ):
        bitstring = left
        bitstring ^= right
        assert bitstring == expected_bitstring
        assert len(bitstring) == len(left)

    @pytest.mark.parametrize(
        "left, right, expected_bitstring",
        [
            (
                FixedWidthBitstring(3, 0b001),
                FixedWidthBitstring(3, 0b011),
                FixedWidthBitstring(3, 0b001),
            ),
            (
                FixedWidthBitstring(4, 0b1011),
                FixedWidthBitstring(6, 0b101010),
                FixedWidthBitstring(4, 0b1010),
            ),
            (
                FixedWidthBitstring(4, 0b1101),
                FixedWidthBitstring(2, 0b01),
                FixedWidthBitstring(4, 0b0001),
            ),
            (
                FixedWidthBitstring(4, 0b1011),
                Bitstring(0b10010100010101),
                FixedWidthBitstring(4, 0b0001),
            ),
        ],
    )
    def test_anding_together_different_width_bitstrings_takes_left_hand_width(
        self,
        left: FixedWidthBitstring,
        right: Bitstring,
        expected_bitstring: FixedWidthBitstring,
    ):
        actual = left & right
        assert actual == expected_bitstring
        assert len(actual) == len(left)

    @pytest.mark.parametrize(
        "left, right, expected_bitstring",
        [
            (
                FixedWidthBitstring(3, 0b001),
                FixedWidthBitstring(3, 0b011),
                FixedWidthBitstring(3, 0b001),
            ),
            (
                FixedWidthBitstring(4, 0b1011),
                FixedWidthBitstring(6, 0b101010),
                FixedWidthBitstring(4, 0b1010),
            ),
            (
                FixedWidthBitstring(4, 0b1101),
                FixedWidthBitstring(2, 0b01),
                FixedWidthBitstring(4, 0b0001),
            ),
            (
                FixedWidthBitstring(4, 0b1011),
                Bitstring(0b10010100010101),
                FixedWidthBitstring(4, 0b0001),
            ),
        ],
    )
    def test_anding_inplace_does_mutates_internal_state_in_expected_way(
        self,
        left: FixedWidthBitstring,
        right: Bitstring,
        expected_bitstring: FixedWidthBitstring,
    ):
        bitstring = left
        bitstring &= right
        assert bitstring == expected_bitstring
        assert len(bitstring) == len(left)


class TestGetRoundWords:
    @pytest.fixture(
        params=[
            ([], 10, 10),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9], 1, 20),
            ([22, 33, 89], 100, 1),
        ]
    )
    def sparse_form_data(self, request):
        return request.param

    def test_round_aligned_words_have_correct_width(self, sparse_form_data):
        sparse_bits, det_per_round, num_rounds = sparse_form_data
        round_bitstrings = get_round_words(sparse_bits, det_per_round, num_rounds)
        for round_bitstring in round_bitstrings:
            assert len(round_bitstring) == det_per_round

    def test_round_aligned_words_have_correct_round_num(self, sparse_form_data):
        sparse_bits, det_per_round, num_rounds = sparse_form_data
        round_bitstrings = get_round_words(sparse_bits, det_per_round, num_rounds)
        assert len(list(round_bitstrings)) == num_rounds

    def test_round_aligned_words_contain_correct_number_of_1s(self, sparse_form_data):
        sparse_bits, det_per_round, num_rounds = sparse_form_data
        round_bitstrings = get_round_words(sparse_bits, det_per_round, num_rounds)
        number_of_ones = 0
        for round_bitstring in round_bitstrings:
            number_of_ones += sum(iter(round_bitstring))

        assert number_of_ones == len(sparse_bits)

    def test_example_round_aligned_words_is_correct(self):
        # Input 0010 1000 0000 0001 0100
        round_bitstrings = get_round_words([2, 4, 15, 17], 4, 5)
        # Output is each of the above nibbles, where MSB is on the right of the nibble
        assert list(round_bitstrings) == [
            FixedWidthBitstring(4, 4),
            FixedWidthBitstring(4, 1),
            FixedWidthBitstring(4, 0),
            FixedWidthBitstring(4, 8),
            FixedWidthBitstring(4, 2),
        ]


@pytest.mark.parametrize("class_", [Bitstring, FixedWidthBitstring])
@pytest.mark.parametrize(
    "method_name",
    [
        "__or__",
        "__ior__",
        "__xor__",
        "__ixor__",
        "__and__",
        "__iand__",
        "__lshift__",
        "__rshift__",
        "__eq__",
    ],
)
def test_bitstrings_invalid_operands(class_, method_name):
    bitstring = class_(0b11)
    other = object()
    method = getattr(bitstring, method_name)
    assert method(other) is NotImplemented
