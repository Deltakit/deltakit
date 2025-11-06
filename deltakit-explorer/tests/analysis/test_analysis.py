# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import numpy as np
import pytest

from deltakit_explorer import Logging
from deltakit_explorer.analysis._analysis import (
    calculate_lep_and_lep_stddev, get_exp_fit, get_lambda_fit
)


class TestCurveFit:

    def test_get_exp_fit_fits(self):
        # the method is very fragile, as it involves 3 mathematical concepts with
        # different restrictions.
        # 1. Probabilities and fidelities,
        #       they should be in [0..0.5], as fidelity is 1-2p.
        # 2. Logarithms, arguments should be positive
        # 3. Linear fit. Matrix should not be singular
        epsilon = 0.04
        rounds = [1, 3, 5, 7, 9, 11]
        f_0 = 1.0
        fidelities = [f_0 * (1 - 2 * epsilon) ** r for r in rounds]
        prob_data = [(1.0 - y) * 0.5 for y in fidelities]
        noisy_prob_data = [
            y * (1.001 if i % 2 else 0.999) for i, y in enumerate(prob_data)
        ]
        shots = [1000] * len(noisy_prob_data)
        fails = [round(p * s) for s, p in zip(shots, noisy_prob_data)]
        with pytest.warns(DeprecationWarning):
            eps, _, y, yerr = get_exp_fit(
                logical_fails_all_rounds=fails,
                shots_all_rounds=shots,
                all_rounds=rounds,
                interpolation_points=6,
            )
        assert pytest.approx(epsilon, rel=0.05) == eps
        assert pytest.approx(y[0], rel=0.05) == prob_data[0]
        assert pytest.approx(y[-1], rel=0.05) == prob_data[-1]
        assert yerr[0] < 0.01

    def test_get_exp_fit_no_fails_raises(self):
        Logging.set_log_to_console(False)
        rounds = [1, 3, 5, 7, 9, 11]
        shots = [1000] * len(rounds)
        fails = [0] * len(rounds)
        with pytest.raises(np.linalg.LinAlgError), pytest.warns(DeprecationWarning):
            get_exp_fit(
                logical_fails_all_rounds=fails,
                shots_all_rounds=shots,
                all_rounds=rounds,
                interpolation_points=6,
            )

    def test_get_exp_fit_negative_fidelity_raises(self):
        rounds = [1, 3, 5, 7, 9, 11]
        shots = [1000] * len(rounds)
        # fildelity is 1 - 2p = 1.0 - 1.2 = -0.2
        fails = [495 + i for i in rounds]
        with pytest.raises(AssertionError), pytest.warns(DeprecationWarning):
            get_exp_fit(
                logical_fails_all_rounds=fails,
                shots_all_rounds=shots,
                all_rounds=rounds,
                interpolation_points=6,
            )


class TestCalculateLep:
    def test_calculate_lep_no_fails_raises(self):
        fails = [500, 200, 25, 0]
        shots = 50000
        with pytest.raises(ValueError):
            calculate_lep_and_lep_stddev(fails=fails, shots=shots)

    def test_calculate_lep_returns_correct_values_with_scalars(self):
        true_lep = 0.1
        true_lep_stddev = 0.00948683  # copied from above
        lep, lep_stddev = calculate_lep_and_lep_stddev(fails=100, shots=1000)
        np.testing.assert_allclose(lep, true_lep)
        np.testing.assert_allclose(lep_stddev, true_lep_stddev, atol=1e-8)

    def test_calculate_lep_returns_correct_values(self):
        true_leps = [0.1, 0.02, 0.005]
        true_lep_stddevs = [0.00948683, 0.00442719, 0.00223047]
        leps, lep_stddevs = calculate_lep_and_lep_stddev(
            fails=[100, 20, 5], shots=1000
        )
        np.testing.assert_allclose(leps, true_leps)
        np.testing.assert_allclose(lep_stddevs, true_lep_stddevs, atol=1e-8)


class TestGetLambdaFit:
    def test_get_lambda_fit_returns_correct_values(self):
        true_lep_fit = [0.000201, 0.000039, 0.00000758]
        with pytest.warns(DeprecationWarning):
            lep_fit = get_lambda_fit(
                distances=[5, 7, 9],
                lep_per_round=[1.992e-04, 4.314e-05, 7.556e-06],
                lep_stddev_per_round=[1.99579718e-05, 9.28881002e-06, 3.88728658e-07],
            )
        assert pytest.approx(lep_fit, rel=0.002) == true_lep_fit
