# (c) Copyright Riverlane 2020-2025.
import pytest

from deltakit_circuit._parse_stim import group_targets


@pytest.mark.parametrize(
    ("qubit_indices", "expected_groups"),
    [
        ([0, 1, 2, 3], [{0, 1, 2, 3}]),
        ([0, 1, 2, 2], [{0, 1, 2}, {2}]),
        ([0, 1, 2, 1, 2, 2], [{0, 1, 2}, {1, 2}, {2}]),
    ],
)
def test_grouping_single_duplicated_item_returns_two_sets(
    qubit_indices, expected_groups
):
    assert group_targets(qubit_indices) == expected_groups
