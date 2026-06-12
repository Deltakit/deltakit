"""Regression tests verifying that the ``uncertainties``-based error propagation
produces results identical to the previously used manual formulas.

Each test computes the standard deviation both ways (manual formula and
``uncertainties`` package) and asserts they match to high precision. This ensures
the refactor from manual to automatic propagation introduced no numerical
regressions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from uncertainties import correlated_values, ufloat
from uncertainties import unumpy as unp
from uncertainties.umath import exp as uexp
from uncertainties.umath import log as ulog


class TestLepprPropagation:
    """Tests for the error propagation used in ``_leppr.py``."""

    @pytest.mark.parametrize(
        ("lep", "lep_stddev", "rounds"),
        [
            (0.1, 0.01, 5),
            (0.01, 0.001, 10),
            (0.3, 0.05, 3),
            (0.001, 0.0001, 20),
        ],
    )
    def test_single_data_point_leppr_stddev(
        self, lep: float, lep_stddev: float, rounds: int
    ) -> None:
        """The single-data-point LEPPR stddev from ``ufloat`` must match the
        previously hand-derived formula: ``stddev * (1 - 2*lep)^(1/r - 1) / r``."""
        # Old manual formula
        manual_stddev = lep_stddev * (1 - 2 * lep) ** (1 / rounds - 1) / rounds

        # New uncertainties-based computation
        lep_u = ufloat(lep, lep_stddev)
        leppr_u = (1 - (1 - 2 * lep_u) ** (1 / rounds)) / 2

        assert pytest.approx(manual_stddev, rel=1e-10) == leppr_u.std_dev

    @pytest.mark.parametrize(
        ("leps", "lep_stddevs"),
        [
            ([0.1, 0.2, 0.3], [0.01, 0.02, 0.03]),
            ([0.01, 0.05, 0.15], [0.001, 0.005, 0.015]),
            ([0.001, 0.01, 0.1, 0.2], [0.0001, 0.001, 0.01, 0.02]),
        ],
    )
    def test_log_fidelity_stddev(
        self, leps: list[float], lep_stddevs: list[float]
    ) -> None:
        """The log-fidelity stddev from ``unumpy.log`` must match the previously
        used manual formula: ``2 * stddev / fidelities``."""
        leps_arr = np.array(leps)
        stddevs_arr = np.array(lep_stddevs)
        fidelities = 1 - 2 * leps_arr

        # Old manual formula
        manual_stddevs = 2 * stddevs_arr / fidelities

        # New uncertainties-based computation
        lep_u = unp.uarray(leps_arr, stddevs_arr)
        logfidelities_u = unp.log(1 - 2 * lep_u)
        uncertainties_stddevs = unp.std_devs(logfidelities_u)

        np.testing.assert_allclose(uncertainties_stddevs, manual_stddevs, rtol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_post_fit_leppr_and_spam_stddev(self, seed: int) -> None:
        """The post-fit LEPPR and SPAM error stddevs from ``correlated_values`` +
        ``uexp`` must match the previously used manual formulas."""
        rng = np.random.default_rng(seed)
        slope = rng.uniform(-1, -0.01)
        offset = rng.uniform(-0.5, -0.01)
        # Generate a realistic positive-definite covariance matrix.
        a = rng.uniform(0.0001, 0.01, size=(2, 2))
        cov = a @ a.T

        slope_stddev, offset_stddev = np.sqrt(np.diagonal(cov))

        # Old manual formulas
        estimated_leppr = float((1 - np.exp(slope)) / 2)
        manual_leppr_stddev = float(
            (1 - 2 * estimated_leppr) * slope_stddev / 2
        )
        estimated_spam = float((1 - np.exp(offset)) / 2)
        manual_spam_stddev = float(
            (1 - 2 * estimated_spam) * offset_stddev / 2
        )

        # New uncertainties-based computation
        slope_u, offset_u = correlated_values([slope, offset], cov)
        leppr_u = (1 - uexp(slope_u)) / 2
        spam_u = (1 - uexp(offset_u)) / 2

        assert pytest.approx(leppr_u.nominal_value, rel=1e-10) == estimated_leppr
        assert pytest.approx(spam_u.nominal_value, rel=1e-10) == estimated_spam
        # The manual formula ignores covariance between slope and offset (they are
        # independent transforms), so the results should match closely.
        assert pytest.approx(leppr_u.std_dev, rel=1e-6) == manual_leppr_stddev
        assert pytest.approx(spam_u.std_dev, rel=1e-6) == manual_spam_stddev


class TestLambdaPropagation:
    """Tests for the error propagation used in ``_lambda.py``."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_lambda_shifted_fit_stddev(self, seed: int) -> None:
        """The Λ and Λ₀ stddevs from ``correlated_values`` must match the
        previously used manual formulas for the shifted fit method."""
        rng = np.random.default_rng(seed)
        slope = rng.uniform(-2, -0.1)
        offset = rng.uniform(-3, -0.1)
        a = rng.uniform(0.001, 0.05, size=(2, 2))
        cov = a @ a.T

        slope_std, offset_std = np.sqrt(np.diagonal(cov))

        # Old manual formulas
        manual_lambda = float(np.exp(-2 * slope))
        manual_lambda_std = float(manual_lambda * 2 * slope_std)
        manual_lambda0 = float(np.exp(-offset - np.log(manual_lambda) / 2))
        manual_lambda0_std = float(
            manual_lambda0
            * np.sqrt(
                offset_std**2
                + manual_lambda_std**2 / (4 * manual_lambda**2)
                - 2 * cov[0, 1]
            )
        )

        # New uncertainties-based computation
        slope_u, offset_u = correlated_values([slope, offset], cov)
        lambda_u = uexp(-2 * slope_u)
        lambda0_u = uexp(-offset_u - ulog(lambda_u) / 2)

        assert pytest.approx(lambda_u.nominal_value, rel=1e-10) == manual_lambda
        assert pytest.approx(lambda_u.std_dev, rel=1e-6) == manual_lambda_std
        assert pytest.approx(lambda0_u.nominal_value, rel=1e-10) == manual_lambda0
        assert pytest.approx(lambda0_u.std_dev, rel=1e-6) == manual_lambda0_std

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_lambda_lin_fit_stddev(self, seed: int) -> None:
        """The Λ and Λ₀ stddevs from ``correlated_values`` must match the
        previously used manual formulas for the linear fit method."""
        rng = np.random.default_rng(seed)
        slope = rng.uniform(-2, -0.1)
        offset = rng.uniform(-3, -0.1)
        a = rng.uniform(0.001, 0.05, size=(2, 2))
        cov = a @ a.T

        slope_std, offset_std = np.sqrt(np.diagonal(cov))

        # Old manual formulas
        manual_lambda = float(np.exp(-slope))
        manual_lambda_std = float(manual_lambda * slope_std)
        manual_lambda0 = float(np.exp(-offset))
        manual_lambda0_std = float(manual_lambda0 * offset_std)

        # New uncertainties-based computation
        slope_u, offset_u = correlated_values([slope, offset], cov)
        lambda_u = uexp(-slope_u)
        lambda0_u = uexp(-offset_u)

        assert pytest.approx(lambda_u.nominal_value, rel=1e-10) == manual_lambda
        assert pytest.approx(lambda_u.std_dev, rel=1e-10) == manual_lambda_std
        assert pytest.approx(lambda0_u.nominal_value, rel=1e-10) == manual_lambda0
        assert pytest.approx(lambda0_u.std_dev, rel=1e-10) == manual_lambda0_std


class TestGradientPropagation:
    """Tests for the error propagation used in ``_gradient.py``."""

    @pytest.mark.parametrize(
        ("lambdas", "lambda_stddevs"),
        [
            ([2.0, 3.0, 4.0], [0.1, 0.15, 0.2]),
            ([0.5, 1.0, 1.5], [0.05, 0.1, 0.15]),
            ([10.0, 20.0], [1.0, 2.0]),
        ],
    )
    def test_lambda_reciprocal_stddev(
        self, lambdas: list[float], lambda_stddevs: list[float]
    ) -> None:
        """The 1/Λ stddev from ``unumpy`` must match the previously used manual
        formula: ``|stddev / Λ²|``."""
        lambdas_arr = np.array(lambdas)
        stddevs_arr = np.array(lambda_stddevs)

        # Old manual formula
        manual_stddevs = np.abs(stddevs_arr / lambdas_arr**2)

        # New uncertainties-based computation
        lambdas_u = unp.uarray(lambdas_arr, stddevs_arr)
        reciprocals_u = 1 / lambdas_u
        uncertainties_stddevs = unp.std_devs(reciprocals_u)

        np.testing.assert_allclose(uncertainties_stddevs, manual_stddevs, rtol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_polynomial_derivative_variance(self, seed: int) -> None:
        """The polynomial derivative stddev from ``correlated_values`` must match
        the previously used manual covariance-matrix formula."""
        rng = np.random.default_rng(seed)
        degree = 3
        gradient_point = 0.5

        # Generate random polynomial coefficients and a realistic covariance matrix.
        n = degree + 1
        a = rng.uniform(0.001, 0.01, size=(n, n))
        cov = a @ a.T
        coefficients = rng.uniform(-1, 1, size=n)

        # Old manual formula for the variance of the derivative.
        coeff_matrix = np.array(
            [
                [(i + 1) * (j + 1) * gradient_point ** (i + j) for i in range(n - 1)]
                for j in range(n - 1)
            ]
        )
        manual_stddev = math.sqrt(float(np.sum(coeff_matrix * cov[1:, 1:])))

        # Old manual formula for the derivative value.
        manual_derivative = float(
            sum(
                c * (power + 1) * gradient_point**power
                for power, c in enumerate(coefficients[1:])
            )
        )

        # New uncertainties-based computation
        coefficients_u = correlated_values(coefficients, cov)
        derivative_u = sum(
            c * (power + 1) * gradient_point**power
            for power, c in enumerate(coefficients_u[1:])
        )

        assert pytest.approx(manual_derivative, rel=1e-10) == derivative_u.nominal_value
        assert pytest.approx(manual_stddev, rel=1e-6) == derivative_u.std_dev