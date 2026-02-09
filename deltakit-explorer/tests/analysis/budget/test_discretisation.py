import itertools

import numpy as np
import numpy.typing as npt
import pytest

from deltakit_explorer.analysis.error_budget._discretisation import (
    GradientFitDiscretisationGenerator,
    get_linear_points,
    get_logarithmic_points,
)


def _assert_is_linear(arr: npt.NDArray[np.floating]) -> None:
    diff = np.abs(arr[1:] - arr[:-1])
    np.testing.assert_allclose(diff - diff[0], 0, atol=1e-7)


@pytest.mark.parametrize(
    ("a", "b", "c", "num_points", "degree"),
    itertools.product([-1, 0, 0.1], [1, 2], [0.5], [5, 10, 1000], [1, 2, 3]),
)
def test_linear_points(
    a: float, b: float, c: float, num_points: int, degree: int
) -> None:
    ret = get_linear_points(a, b, c, num_points, degree)
    assert len(ret) == num_points
    assert np.all(np.logical_and(a <= ret, ret <= b))
    _assert_is_linear(ret)


@pytest.mark.parametrize(
    ("a", "b", "c", "num_points", "degree"),
    itertools.product([0.1, 0.5, 1.0], [1.1, 2.0, 5.0], [1.05], [5, 10], [1, 2, 3]),
)
def test_logarithmic_points(
    a: float, b: float, c: float, num_points: int, degree: int
) -> None:
    ret = get_logarithmic_points(a, b, c, num_points, degree)
    assert len(ret) == num_points
    eps = 1e-7
    assert np.all(np.logical_and(a <= ret + eps, ret <= b + eps))
    _assert_is_linear(np.log10(ret))


@pytest.mark.parametrize(
    ("func", "abc"),
    itertools.product(
        [get_linear_points, get_logarithmic_points],
        [
            (1, 2, 3),  # a < b < c
            (2, 1, 3),  # b < a < c
            (3, 1, 2),  # b < c < a
            (2, 3, 1),  # c < a < b
            (3, 2, 1),  # c < b < a
        ],
    ),
)
def test_raises_on_invalid_inputs(
    func: GradientFitDiscretisationGenerator, abc: tuple[float, float, float]
) -> None:
    a, b, c = abc
    with pytest.raises(ValueError, match=f"Expected {a=} < {c=} < {b=}"):
        func(a, b, c, 5, 3)


def test_raise_on_negative_inputs_log() -> None:
    with pytest.raises(
        ValueError,
        match="Cannot get logarithmically-spaced points for negative values.*",
    ):
        get_logarithmic_points(-1, 1, 0, 5, 3)
