import itertools

import numpy
import pytest
from typing import Literal

from deltakit_explorer.analysis.lambda_ import calculate_lambda_and_lambda_stddev

@pytest.fixture
def rng() -> numpy.random.Generator:
    return numpy.random.default_rng(4957349587)

class TestCalculateLambda:
    @pytest.mark.parametrize(
        "method,distances,lambda_,lambda0,relative_stddev",
        itertools.product(
            ("d", "(d+1)/2", "direct"),
            ((5, 7, 9), (5, 9, 13), tuple(range(5, 50, 6))),
            (0.7, 0.9, 1 - 1e-7, 1, 1 + 1e-7, 1.1, 1.5, 2, 10, 20),
            (0.01, 0.1, 1 - 1e-7, 1, 1 + 1e-7, 2, 10, 100),
            (10**(-x) for x in (1, 3, 5, 7, 9))
        ),
    )
    def test_synthetic_values(
        self,
        method: Literal["d", "(d+1)/2", "direct"],
        distances: tuple[int, ...],
        lambda_: float,
        lambda0: float,
        relative_stddev: float,
        rng: numpy.random.Generator,
    ) -> None:
        lepprs = numpy.asarray([1 / (lambda0 * lambda_ ** ((d + 1) / 2)) for d in distances])
        relative_stddevs = rng.normal(0, relative_stddev, size=len(distances))
        lepprs_stddev = (1 + relative_stddevs) * lepprs
        res = calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stddev, method)
        # Test that the estimated quantities are within 3*sigma of the real one.
        assert pytest.approx(res.lambda_, abs=3 * res.lambda_stddev) == lambda_
        assert pytest.approx(res.lambda0, abs=3 * res.lambda0_stddev) == lambda0
        assert isinstance(res.lambda_, float)
        assert isinstance(res.lambda_stddev, float)
        assert isinstance(res.lambda0, float)
        assert isinstance(res.lambda0_stddev, float)

    def test_non_unique_distances_raises(self):
        distances = [5, 5, 7]
        lepprs = [0.01, 0.01, 0.001]
        lepprs_stddevs = [1e-10, 1e-10, 1e-10]
        with pytest.raises(ValueError, match="^Multiple entries were provided"):
            calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stddevs)

    def test_even_distances_raises(self):
        distances = [2, 4, 6]
        lepprs = [0.01, 0.001, 0.0001]
        lepprs_stddevs = [1e-10, 1e-10, 1e-10]
        with pytest.raises(ValueError, match="^Found at least one even distance"):
            calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stddevs)

    @pytest.mark.parametrize(
        "lamb,distances",
        itertools.product(
            (0.1, 0.5, 0.9, 1 - 1e-7, 1 + 1e-7, 1.1, 1.2, 1.3, 1.4),
            ([3, 5, 7], list(range(3, 20, 4))),
        ),
    )
    def test_small_lambda_and_low_distance_warns(
        self, lamb: float, distances: list[int]
    ) -> None:
        lepprs = [0.1 * lamb ** (-(d + 1) / 2) for d in distances]
        lepprs_stddevs = [1e-10 for _ in distances]
        msg = "^Lambda estimation is unreliable at low code distances and low values of lambda."
        with pytest.warns(UserWarning, match=msg):
            calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stddevs)

    @pytest.mark.parametrize(
        "methods,distances,lambda_,lambda0,relative_stddev",
        itertools.product(
            itertools.combinations(["d", "(d+1)/2", "direct"], 2),
            ((5, 7, 9), (5, 9, 13)),
            (0.7, 0.9, 1 - 1e-7, 1, 1 + 1e-7, 1.1, 1.5, 2, 10, 20),
            (0.01, 0.1, 1 - 1e-7, 1, 1 + 1e-7, 2, 10, 100),
            (10**(-x) for x in (1, 3, 5, 7, 9))
        ),
    )
    def test_different_methods_agree(
        self,
        methods: tuple[Literal["d", "(d+1)/2", "direct"], Literal["d", "(d+1)/2", "direct"]],
        distances: tuple[int, ...],
        lambda_: float,
        lambda0: float,
        relative_stddev: float,
        rng: numpy.random.Generator,
    ) -> None:
        m1, m2 = methods
        lepprs = numpy.asarray([1 / (lambda0 * lambda_ ** ((d + 1) / 2)) for d in distances])
        relative_stddevs = rng.normal(0, relative_stddev, size=len(distances))
        lepprs_stddev = (1 + relative_stddevs) * lepprs
        res1 = calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stddev, m1)
        res2 = calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stddev, m2)
        # Estimations of lambda and lambda0 are already checked by another test, so we
        # only check that the standard deviations actually agree here.
        assert pytest.approx(res1.lambda_stddev, rel=1e-6) == res2.lambda_stddev
        assert pytest.approx(res1.lambda0_stddev, rel=1e-6) == res2.lambda0_stddev
