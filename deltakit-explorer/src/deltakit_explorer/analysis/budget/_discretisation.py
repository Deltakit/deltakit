from enum import Enum, auto
from typing import Protocol

import numpy as np
import numpy.typing as npt


class GradientFitDiscretisationGenerator(Protocol):
    def __call__(
        self, a: float, b: float, c: float, num_points: int, degree: int, /
    ) -> npt.NDArray[np.floating]:
        """
        Find a good set of points to estimate the gradient at point ``c`` by fitting a
        polynomial of degree ``degree`` on the interval ``[a, b]``.

        Functions implementing that interface find ``num_points`` unique points ``x``
        between ``a`` and ``b`` that will be used to fit a polynomial of degree
        ``degree``.

        Different implementation might choose to minimise different quantities. For
        example:

        - minimise code complexity, by simply returning linear (or log-spaced) points,
        - minimise the variance of the resulting gradient estimation, for example using
          a D-optimal design (sees
          https://en.wikipedia.org/wiki/Optimal_experimental_design).

        Args:
            a (float): lower bound of the interval in which fitting points should be
                computed.
            b (float): upper bound of the interval in which fitting points should be
                computed.
            c (float): point at which the gradient will be estimated.
            num_points (int): number of points to return within ``[a, b]``.
            degree (int): degree of the polynomial that will be used to fit the values
                computed at each of the points returned by this function.

        Returns:
            a sorted array of ``num_points`` points within ``[a, b]`` and without duplicates
            that should be used to evaluate the function to fit with a degree ``degree``
            polynomial.

        Raises:
            ValueError: if ``a < c < b`` is not verified.
        """
        ...


def _check_interval(a: float, b: float, c: float) -> None:
    if not a < c < b:
        msg = f"Expected {a=} < {c=} < {b=}"
        raise ValueError(msg)


def get_linear_points(
    a: float, b: float, c: float, num_points: int, _: int
) -> npt.NDArray[np.floating]:
    """Returns ``num_points`` linearly spaced between ``a`` and ``b``."""
    _check_interval(a, b, c)
    return np.linspace(a, b, num_points)


def get_logarithmic_points(
    a: float, b: float, c: float, num_points: int, _: int
) -> npt.NDArray[np.floating]:
    """Returns ``num_points`` logarithmically spaced between ``a`` and ``b``."""
    _check_interval(a, b, c)
    if a <= 0:
        msg = (
            "Cannot get logarithmically-spaced points for negative values. "
            f"Got [{a}, {b}]."
        )
        raise ValueError(msg)
    return np.logspace(np.log10(a), np.log10(b), num_points, base=10)


class DiscretisationStrategy(Enum):
    """Strategy to use to generate discretisation point for fitting a noisy function
    with a polynomial."""

    LINEAR = auto()
    """Linearly spaced points between the discretisation space boundaries."""
    LOGARITHMIC = auto()
    """Logarithmically spaced points between the discretisation space boundaries."""

    def __call__(
        self, a: float, b: float, c: float, num_points: int, degree: int, /
    ) -> npt.NDArray[np.floating]:
        match self:
            case DiscretisationStrategy.LINEAR:
                return get_linear_points(a, b, c, num_points, degree)
            case DiscretisationStrategy.LOGARITHMIC:
                return get_logarithmic_points(a, b, c, num_points, degree)
            case _:
                msg = f"Discretisation {self} is not implemented yet."
                raise NotImplementedError(msg)
