import itertools
from math import floor
from typing import Literal

import numpy as np
import pytest

from deltakit_explorer.analysis import calculate_lambda_and_lambda_stddev


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(4957349587)


class TestCalculateLambda:
    def test_raise_error_for_mismatch_inputs(self) -> None:
        distances = [5, 7, 9]
        leppr = [1.992e-04, 4.314e-05, 7.556e-06]
        leppr_std = [1.2e-05, 9.3e-06, 3.9e-06]

        with pytest.raises(ValueError):
            calculate_lambda_and_lambda_stddev(
                distances=[5, 7],
                leppr=leppr,
                leppr_std=leppr_std,
            )

        with pytest.raises(ValueError):
            calculate_lambda_and_lambda_stddev(
                distances=distances,
                leppr=[1.992e-04, 4.314e-05],
                leppr_std=leppr_std,
            )

        with pytest.raises(ValueError):
            calculate_lambda_and_lambda_stddev(
                distances=distances,
                leppr=leppr,
                leppr_std=[1.2e-05, 9.3e-06],
            )

    @pytest.mark.parametrize(
        ("method", "distances", "lambda_", "lambda0", "relative_std"),
        list(
            itertools.product(
                ("shifted", "lin", "curve"),
                ((5, 7, 9), (5, 6, 7, 8, 9), tuple(range(5, 50, 6))),
                (0.7, 0.9, 1 - 1e-7, 1, 1 + 1e-7, 1.1, 1.5, 2, 10, 20),
                (0.01, 0.1, 1 - 1e-7, 1, 1 + 1e-7, 2, 10, 100),
                (10 ** (-x) for x in (1, 3, 5, 7, 9)),
            )
        ),
    )
    def test_synthetic_values(
        self,
        method: Literal["shifted", "lin", "curve"],
        distances: tuple[int, ...],
        lambda_: float,
        lambda0: float,
        relative_std: float,
        rng: np.random.Generator,
    ) -> None:
        lepprs = np.asarray(
            [1 / (lambda0 * lambda_ ** floor((d + 1) / 2)) for d in distances]
        )
        relative_stds = rng.normal(0, relative_std, size=len(distances))
        lepprs_std = (1 + relative_stds) * lepprs
        res = calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_std, method)
        # Test that the estimated quantities are within 3*sigma of the real one.
        assert pytest.approx(res.lambda_, abs=3 * res.lambda_std) == lambda_
        assert pytest.approx(res.lambda0, abs=3 * res.lambda0_std) == lambda0
        assert isinstance(res.lambda_, float)
        assert isinstance(res.lambda_std, float)
        assert isinstance(res.lambda0, float)
        assert isinstance(res.lambda0_std, float)

    def test_non_unique_distances_raises(self):
        distances = [5, 5, 7]
        lepprs = [0.01, 0.01, 0.001]
        lepprs_stds = [1e-10, 1e-10, 1e-10]
        with pytest.raises(ValueError, match="^Multiple entries were provided"):
            calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stds)

    @pytest.mark.parametrize(
        ("lamb", "distances"),
        list(
            itertools.product(
                (0.1, 0.5, 0.9, 1 - 1e-7, 1 + 1e-7, 1.1, 1.2, 1.3, 1.4),
                ([3, 5, 7], list(range(3, 20, 4))),
            )
        ),
    )
    def test_small_lambda_and_low_distance_warns(
        self, lamb: float, distances: list[int]
    ) -> None:
        lepprs = [0.1 * lamb ** (-(d + 1) / 2) for d in distances]
        lepprs_stds = [1e-10 for _ in distances]
        msg = "^Lambda estimation is unreliable at low code distances and low values of lambda."
        with pytest.warns(UserWarning, match=msg):
            calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_stds)

    @pytest.mark.parametrize(
        ("methods", "distances", "lambda_", "lambda0", "relative_std"),
        list(
            itertools.product(
                itertools.combinations(["shifted", "lin", "curve"], 2),
                ((5, 7, 9), (5, 9, 13)),
                (0.7, 0.9, 1 - 1e-7, 1, 1 + 1e-7, 1.1, 1.5, 2, 10, 20),
                (0.01, 0.1, 1 - 1e-7, 1, 1 + 1e-7, 2, 10, 100),
                (10 ** (-x) for x in (1, 3, 5, 7, 9)),
            )
        ),
    )
    def test_different_methods_agree(
        self,
        methods: tuple[
            Literal["shifted", "lin", "curve"], Literal["shifted", "lin", "curve"]
        ],
        distances: tuple[int, ...],
        lambda_: float,
        lambda0: float,
        relative_std: float,
        rng: np.random.Generator,
    ) -> None:
        m1, m2 = methods
        lepprs = np.asarray(
            [1 / (lambda0 * lambda_ ** ((d + 1) / 2)) for d in distances]
        )
        relative_stds = rng.normal(0, relative_std, size=len(distances))
        lepprs_std = (1 + relative_stds) * lepprs
        res1 = calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_std, m1)
        res2 = calculate_lambda_and_lambda_stddev(distances, lepprs, lepprs_std, m2)
        # Estimations of lambda and lambda0 are already checked by another test, so we
        # only check that the standard deviations actually agree here.
        assert pytest.approx(res1.lambda_std, rel=1e-6) == res2.lambda_std
        assert pytest.approx(res1.lambda0_std, rel=1e-6) == res2.lambda0_std
