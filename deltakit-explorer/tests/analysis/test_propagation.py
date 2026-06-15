import itertools
from math import sqrt

import numpy as np
import pytest

from deltakit_explorer.analysis import (
    calculate_lep_and_lep_stddev,
    compute_logical_error_per_round,
)
from deltakit_explorer.analysis._lambda import (
    lambda_from_curve_fit,
    lambda_from_lin_fit,
    lambda_from_shifted_fit,
)
from deltakit_explorer.analysis._leppr import (
    epsilon_and_spam_from_log_fit,
    leppr_from_single_point,
    log_fidelity_stddev,
)
from deltakit_explorer.analysis.error_budget._gradient import (
    polynomial_derivative_stddev,
)
from deltakit_explorer.analysis.error_budget._lambda import reciprocal_stddev


def _legacy_leppr_from_single_point(
    lep: float, lep_stddev: float, rounds: int
) -> tuple[float, float]:
    leppr = (1 - (1 - 2 * lep) ** (1 / rounds)) / 2
    leppr_stddev = lep_stddev * (1 - 2 * lep) ** (1 / rounds - 1) / rounds
    return leppr, leppr_stddev


def _legacy_log_fidelity_stddev(lep: np.ndarray, lep_stddev: np.ndarray) -> np.ndarray:
    fidelities = 1 - 2 * lep
    return 2 * lep_stddev / fidelities


def _legacy_epsilon_and_spam_from_log_fit(
    slope: float, offset: float, cov: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    slope_stddev, offset_stddev = np.sqrt(np.diagonal(cov))
    leppr = (1 - np.exp(slope)) / 2
    leppr_stddev = (1 - 2 * leppr) * slope_stddev / 2
    spam = (1 - np.exp(offset)) / 2
    spam_stddev = (1 - 2 * spam) * offset_stddev / 2
    return (leppr, leppr_stddev), (spam, spam_stddev)


def _legacy_lambda_from_shifted_fit(
    slope: float, offset: float, cov: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    slope_std, offset_std = np.sqrt(np.diagonal(cov))
    estimated_lambda = float(np.exp(-2 * slope))
    estimated_lambda_std = float(estimated_lambda * 2 * slope_std)
    estimated_lambda0 = float(np.exp(-offset - np.log(estimated_lambda) / 2))
    estimated_lambda0_std = float(
        estimated_lambda0
        * np.sqrt(
            offset_std**2
            + estimated_lambda_std**2 / (4 * estimated_lambda**2)
            - 2 * cov[0, 1]
        )
    )
    return (estimated_lambda, estimated_lambda_std), (
        estimated_lambda0,
        estimated_lambda0_std,
    )


def _legacy_lambda_from_lin_fit(
    slope: float, offset: float, cov: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    slope_std, offset_std = np.sqrt(np.diagonal(cov))
    estimated_lambda = float(np.exp(-slope))
    estimated_lambda_std = float(estimated_lambda * slope_std)
    estimated_lambda0 = float(np.exp(-offset))
    estimated_lambda0_std = float(estimated_lambda0 * offset_std)
    return (estimated_lambda, estimated_lambda_std), (
        estimated_lambda0,
        estimated_lambda0_std,
    )


def _legacy_lambda_from_curve_fit(
    lamb0: float, lamb: float, cov: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    lamb0_std, lamb_std = np.sqrt(np.diagonal(cov))
    return (lamb, lamb_std), (lamb0, lamb0_std)


def _legacy_variance_of_gradient_at_point(cov: np.ndarray, c: float) -> float:
    n = cov.shape[0]
    coeff_matrix = np.array(
        [[(i + 1) * (j + 1) * c ** (i + j) for i in range(n - 1)] for j in range(n - 1)]
    )
    return float(np.sum(coeff_matrix * cov[1:, 1:]))


class TestPropagationParity:
    @pytest.mark.parametrize("leppr", [5e-5, 1e-3, 5e-2])
    def test_leppr_from_single_point(self, leppr: float) -> None:
        rounds = 30
        fidelity = (1 - 2 * leppr) ** rounds
        lep = (1 - fidelity) / 2
        lep_stddev = lep * (1 - lep) / sqrt(100_000)
        legacy = _legacy_leppr_from_single_point(lep, lep_stddev, rounds)
        current = leppr_from_single_point(lep, lep_stddev, rounds)
        np.testing.assert_allclose(current, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("p", [1e-4, 1e-2, 0.2, 0.45])
    def test_log_fidelity_stddev(self, p: float) -> None:
        lep = np.array([p])
        lep_stddev = np.sqrt(p * (1 - p) / 10_000)
        legacy = _legacy_log_fidelity_stddev(lep, lep_stddev)
        current = log_fidelity_stddev(lep, lep_stddev)
        np.testing.assert_allclose(current, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize(
        ("slope", "offset", "cov"),
        [
            (-0.01, -0.05, np.array([[1e-6, 2e-7], [2e-7, 3e-6]])),
            (-0.5, -1.2, np.array([[1e-2, 4e-3], [4e-3, 2e-2]])),
            (-2.0, -0.3, np.array([[5e-3, -1e-3], [-1e-3, 8e-3]])),
            (-0.2, -1.5, np.array([[3e-4, 0.0], [0.0, 7e-4]])),
        ],
    )
    def test_epsilon_and_spam_from_log_fit(
        self, slope: float, offset: float, cov: np.ndarray
    ) -> None:
        legacy = _legacy_epsilon_and_spam_from_log_fit(slope, offset, cov)
        current = epsilon_and_spam_from_log_fit(slope, offset, cov)
        np.testing.assert_allclose(current[0], legacy[0], rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(current[1], legacy[1], rtol=1e-10, atol=1e-15)

    def test_lambda_from_shifted_fit_with_cross_covariance(self) -> None:
        slope, offset = -0.5, -1.2
        cov = np.array([[0.01, 0.004], [0.004, 0.02]])
        legacy = _legacy_lambda_from_shifted_fit(slope, offset, cov)
        current = lambda_from_shifted_fit(slope, offset, cov)
        np.testing.assert_allclose(current[0], legacy[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(current[1], legacy[1], rtol=1e-10, atol=1e-12)

    def test_lambda_from_lin_fit(self) -> None:
        slope, offset = -0.3, -2.0
        cov = np.array([[0.002, 0.0], [0.0, 0.005]])
        legacy = _legacy_lambda_from_lin_fit(slope, offset, cov)
        current = lambda_from_lin_fit(slope, offset, cov)
        np.testing.assert_allclose(current, legacy, rtol=1e-12, atol=1e-15)

    def test_lambda_from_curve_fit(self) -> None:
        lamb0, lamb = 0.1, 2.5
        cov = np.array([[1e-4, 1e-5], [1e-5, 2e-4]])
        legacy = _legacy_lambda_from_curve_fit(lamb0, lamb, cov)
        current = lambda_from_curve_fit(lamb0, lamb, cov)
        np.testing.assert_allclose(current, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize(
        ("degree", "point"),
        itertools.product([3, 4, 5], [0.25, 0.5, 0.75]),
    )
    def test_polynomial_derivative_stddev(self, degree: int, point: float) -> None:
        x = np.linspace(0, 1, degree + 5)
        y = x**2 + 0.5 * x
        stddevs = 1e-8 + np.zeros_like(x)
        coefficients, cov = np.polyfit(x, y, deg=degree, cov="unscaled", w=1 / stddevs)
        coefficients, cov = np.flip(coefficients), np.flip(cov)
        derivative, stddev = polynomial_derivative_stddev(coefficients, cov, point)
        legacy_variance = _legacy_variance_of_gradient_at_point(cov, point)
        expected_derivative = sum(
            coefficient * (power + 1) * point**power
            for power, coefficient in enumerate(coefficients[1:])
        )
        assert pytest.approx(expected_derivative) == derivative
        assert pytest.approx(sqrt(legacy_variance), rel=1e-9) == stddev

    @pytest.mark.parametrize(
        ("value", "stddev"),
        [(2.0, 0.1), (10.0, 0.5), (0.7, 0.02)],
    )
    def test_reciprocal_stddev(self, value: float, stddev: float) -> None:
        legacy = abs(stddev / value**2)
        current = reciprocal_stddev(value, stddev)
        np.testing.assert_allclose(current, legacy, rtol=1e-12, atol=1e-15)

    def test_compute_logical_error_per_round_end_to_end(self) -> None:
        num_failed_shots = [9949, 8434, 9649, 9926]
        num_shots = [50000, 20000, 20000, 20000]
        num_rounds = [5, 10, 15, 20]
        lep, lep_stddev = calculate_lep_and_lep_stddev(num_failed_shots, num_shots)
        res = compute_logical_error_per_round(num_rounds, lep, lep_stddev)
        assert pytest.approx(res.leppr, 3 * res.leppr_stddev) == 0.11912
