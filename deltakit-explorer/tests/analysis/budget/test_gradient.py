import itertools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from deltakit_explorer.analysis.error_budget._gradient import (
    _approximate_derivative_at_point_from_values,
    _variate_ith_parameter_by,
)


def _arr(array: Sequence[float]) -> npt.NDArray[np.floating]:
    return np.asarray(array)


@pytest.mark.parametrize(
    ("parameters", "variations", "i"),
    [
        (_arr([0.0]), _arr([0.0, 1.0, 2.0]), 0),
        (_arr([0.0, 1.0, 2.0]), _arr([0.0, 1.0, 2.0]), 2),
        (np.arange(10), np.arange(5), 8),
    ],
)
def test_variate_ith_parameter_by(
    parameters: npt.NDArray[np.floating],
    variations: npt.NDArray[np.floating],
    i: int,
) -> None:
    res = np.array(list(_variate_ith_parameter_by(parameters, variations, i)))
    assert len(res.shape) == 2
    n, m = res.shape
    assert n == variations.size
    assert m == parameters.size
    all_entries_except_i_mask = np.arange(m) != i
    for array, variation in zip(res, variations, strict=True):
        np.testing.assert_allclose(
            array[all_entries_except_i_mask], parameters[all_entries_except_i_mask]
        )
        assert pytest.approx(variation) == array[i]


@pytest.mark.parametrize(
    ("degree", "func_and_expected_derivative"),
    itertools.product(
        [3, 4, 5],
        [
            (lambda x: x**2, 1),
            (lambda x: x**3, 3 / 4),
            (lambda x: x**4, 1 / 2),
            (lambda x: 4 * x + 0.5 * x**2, 4.5),
            (lambda x: -30 + 4 * x + 0.5 * x**2, 4.5),
        ],
    ),
)
def test_approximate_derivative_at_point_from_values(
    degree: int,
    func_and_expected_derivative: tuple[
        Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]], float
    ],
) -> None:
    func, expected_derivative = func_and_expected_derivative
    x = np.linspace(0, 1, 100)
    y = func(x)
    gradient_approximation_point = 0.5
    stddevs = 1e-10 + np.zeros_like(x)

    derivative, stddev = _approximate_derivative_at_point_from_values(
        x, y, stddevs, gradient_approximation_point, degree
    )
    assert pytest.approx(expected_derivative) == derivative
    assert pytest.approx(0, abs=1e-7) == stddev
