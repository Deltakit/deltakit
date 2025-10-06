# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations
import itertools
import re

from deltakit_explorer.analysis._analysis import compute_logical_error_per_round
import numpy as np
import pytest
from deltakit_explorer import Logging, analysis


class TestLEPPerRoundComputation:
    @pytest.mark.parametrize(
        "leppr,spam_error,is_noisy",
        itertools.product(
            (1e-5, 1e-4, 1e-3, 1e-2), (1e-5, 1e-4, 1e-3, 1e-2), (False, True)
        ),
    )
    def test_on_synthetic_inputs(self, leppr: float, spam_error: float, is_noisy: bool):
        f_0 = 1 - 2 * spam_error
        rounds = np.arange(2, np.ceil(np.log(0.3 / f_0) / np.log(1 - 2 * leppr)), 2)
        num_shots = 100_000 + np.zeros_like(rounds)
        fidelities = f_0 * (1 - 2 * leppr) ** rounds
        lep = (1 - fidelities) / 2
        rng = np.random.default_rng(239845794235)
        if is_noisy:
            lep *= 1 - 1e-4 * rng.random(lep.size)
        nfails = np.rint(num_shots * lep)

        res = compute_logical_error_per_round(nfails, num_shots, rounds)
        # Test that the estimated quantities are within 3*sigma of the real one.
        assert pytest.approx(res.leppr, abs=3 * res.leppr_stddev) == leppr
        assert (
            pytest.approx(res.spam_error, abs=3 * res.spam_error_stddev) == spam_error
        )
        assert isinstance(res.leppr, float)
        assert isinstance(res.leppr_stddev, float)
        assert isinstance(res.spam_error, float)
        assert isinstance(res.spam_error_stddev, float)

    @pytest.mark.parametrize(
        "rounds", ([0, 1, 2, 3, 4, 3], [-2, 1, 1, 3, 3, 3], [4, 8, 4, 0, 5])
    )
    def test_raises_when_duplicated_round_number(self, rounds: list[int]):
        f_0, leppr = 0.999, 0.001
        nprounds = np.asarray(rounds)
        num_shots = 100_000 + np.zeros_like(rounds)
        fidelities = f_0 * (1 - 2 * leppr) ** nprounds
        lep = (1 - fidelities) / 2
        nfails = np.rint(num_shots * lep)

        message = "^Multiple entries were provided for the following number of rounds:"
        with pytest.raises(RuntimeError, match=message):
            compute_logical_error_per_round(nfails, num_shots, nprounds)

    @pytest.mark.parametrize(
        "rounds", ([0, 1, 2, 3, 4], [-1, 4, 5, 7], [8, 4, 0, 5, -1, -348975])
    )
    def test_raises_when_invalid_round_number(self, rounds: list[int]):
        f_0, leppr = 0.999, 0.001
        nprounds = np.asarray(rounds)
        num_shots = 100_000 + np.zeros_like(rounds)
        fidelities = f_0 * (1 - 2 * leppr) ** nprounds
        lep = (1 - fidelities) / 2
        nfails = np.rint(num_shots * lep)

        with pytest.warns(UserWarning) as reporter:
            compute_logical_error_per_round(nfails, num_shots, nprounds)
        # Check that at least the "invalid number of rounds" warning has been raised
        # once or more.
        pattern = r"Found an invalid number of rounds: -?[0-9]+"
        assert any(re.match(pattern, str(warning.message)) for warning in reporter)

    def test_real_world_example(self):
        # Note that this test fails when the ``bounds`` optional parameter is set in the
        # call to curve_fit. My best guess at the moment is that the optimiser used by
        # curve_fit when bounds are provided ("trf") behaves strangely with the
        # optimisation problem we give it, whereas the default optimiser without bounds
        # ("lm") works nicely.
        num_failed_shots = [9949, 8434, 9649, 9926]
        num_shots = [50000, 20000, 20000, 20000]
        num_rounds = [5, 10, 15, 20]
        res = compute_logical_error_per_round(num_failed_shots, num_shots, num_rounds)

        assert pytest.approx(res.leppr, 3 * res.leppr_stddev) == 0.11912

    def test_raises_when_no_fails(self):
        shots = 100_000
        message = "^Got an experiment without any errors.*"
        with pytest.raises(RuntimeError, match=message):
            compute_logical_error_per_round([0, 0, 0], [shots, shots, shots], [2, 4, 6])

    def test_raises_when_too_many_fails(self):
        shots = 100_000
        message = "^Got estimations of logical error-rates above 0.5.*"
        with pytest.raises(RuntimeError, match=message):
            compute_logical_error_per_round(
                [shots // 2 + 1] * 3, [shots] * 3, [2, 4, 6]
            )

    def test_warn_when_max_lep_is_too_small(self):
        shots = 100_000
        message = r"^The maximum estimated logical error-rate \([^\)]+\) is below 0.2.*"
        with pytest.warns(UserWarning, match=message):
            compute_logical_error_per_round([2460, 4343, 6151], [shots] * 3, [2, 4, 6])

    def test_warn_when_linear_fit_is_bad(self):
        f_0 = 1 - 0.01
        rounds = np.arange(2, 61, 5)
        # Non-constant logical error-rate per round that should trigger the R2 check.
        leppr = np.array(
            [
                0.00485509,
                0.00606816,
                0.00226491,
                0.00426893,
                0.00082944,
                0.00218146,
                0.00558842,
                0.0088417,
                0.00508088,
                0.00051394,
                0.00762123,
                0.00475807,
            ]
        )
        num_shots = 100_000 + np.zeros_like(rounds)
        fidelities = f_0 * (1 - 2 * leppr) ** rounds
        lep = (1 - fidelities) / 2
        nfails = np.rint(num_shots * lep)

        message = r"Got a R2 value of -?[0-9]+\.[0-9]+ < 0.98."
        with pytest.warns(UserWarning, match=message):
            compute_logical_error_per_round(nfails, num_shots, rounds)

    def test_stddev_reduces_with_more_shots(self):
        f_0, leppr = 1 - 0.001, 0.001
        max_round = np.ceil(np.log(0.3 / f_0) / np.log(1 - 2 * leppr))
        rounds = np.arange(2, max_round, 2)
        fidelities = f_0 * (1 - 2 * leppr) ** rounds
        lep = (1 - fidelities) / 2
        leppr_stddevs = []
        for nshots in [1_000, 10_000, 100_000, 1_000_000]:
            num_shots = nshots + np.zeros_like(rounds)
            nfails = np.rint(num_shots * lep)
            res = compute_logical_error_per_round(nfails, num_shots, rounds)
            leppr_stddevs.append(res.leppr_stddev)

        assert leppr_stddevs == sorted(leppr_stddevs, reverse=True)

    @pytest.mark.parametrize("leppr", [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
    def test_single_point_fit(self, leppr: float):
        rounds = 30
        num_shots = 100_000
        fidelity = (1 - 2 * leppr) ** rounds
        lep = (1 - fidelity) / 2
        nfails = np.rint(num_shots * lep)

        with pytest.warns(UserWarning) as warning_collector:
            res = compute_logical_error_per_round([nfails], [num_shots], [rounds])
        expected_message = (
            "^Only one data-point provided for logical error probability per round. "
            "Continuing computation assuming that SPAM error is negligible.$"
        )
        assert any(
            re.match(expected_message, str(w.message)) for w in warning_collector
        ), "Expected to get a warning on single data-point being used."
        # Test that the estimated quantities are within 3*sigma of the real one.
        assert pytest.approx(res.leppr, abs=3 * res.leppr_stddev) == leppr
        assert pytest.approx(res.spam_error, abs=3 * res.spam_error_stddev) == 0
        assert isinstance(res.leppr, float)
        assert isinstance(res.leppr_stddev, float)


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
        eps, _, y, yerr = analysis.get_exp_fit(
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
        with pytest.raises(np.linalg.LinAlgError):
            analysis.get_exp_fit(
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
        with pytest.raises(AssertionError):
            analysis.get_exp_fit(
                logical_fails_all_rounds=fails,
                shots_all_rounds=shots,
                all_rounds=rounds,
                interpolation_points=6,
            )


class TestCalculateLambda:

    def test_calculate_lambda_returns_correct_values(self):
        true_lambda_value = 5.155
        true_lambda_value_stddev = 0.287

        _lambda, lambda_stddev = analysis.calculate_lambda_and_lambda_stddev(
            distances=[5, 7, 9],
            lep_per_round=[1.992e-04, 4.314e-05, 7.556e-06],
            lep_stddev_per_round=[1.99579718e-05, 9.28881002e-06, 3.88728658e-07]
        )
        assert pytest.approx(_lambda, rel=0.002) == true_lambda_value
        assert pytest.approx(lambda_stddev, rel=0.002) == true_lambda_value_stddev

    def test_calculate_lambda_few_distance_raises(self):
        Logging.set_log_to_console(False)
        distances = [7, 9]
        lep = [0.000996, 0.000302, 0.00006]
        lep_stddev = [2.0e-09, 6.0e-10, 1.2e-10]
        with pytest.raises(ValueError):
            analysis.calculate_lambda_and_lambda_stddev(
                distances=distances,
                lep_per_round=lep,
                lep_stddev_per_round=lep_stddev
            )

    @pytest.mark.parametrize(
        "lambd,distances,should_warn",
        [
            (1.3, [2, 3, 4], True),
            (1.55, [2, 3, 4], False),
            (1.3, [5, 7, 9], False),
            (1.55, [5, 7, 9], False),
        ]
    )
    def test_calculate_lambda_warns(self, mocker, lambd, distances, should_warn):
        mocker.patch("deltakit_explorer._utils._logging.Logging.warn")
        blep, blep_stddev = 0.000996, 2.0e-09
        lep = [blep / lambd**((d - distances[0]) // 2) for d in distances]
        lep_stddev = [blep_stddev / lambd**((d - distances[0]) // 2) for d in distances]
        analysis.calculate_lambda_and_lambda_stddev(
            distances=distances,
            lep_per_round=lep,
            lep_stddev_per_round=lep_stddev
        )
        if should_warn:
            Logging.warn.assert_called_once()
        else:
            Logging.warn.assert_not_called()
        Logging.set_log_to_console(False)


class TestCalculateLep:

    def test_calculate_lep_no_fails_raises(self):
        fails = [500, 200, 25, 0]
        shots = 50000
        with pytest.raises(ValueError):
            analysis.calculate_lep_and_lep_stddev(
                fails=fails,
                shots=shots
            )

    def test_calculate_lep_returns_correct_values(self):
        true_leps = [0.1, 0.02, 0.005]
        true_lep_stddevs = [0.00948683, 0.00442719, 0.00223047]
        leps, lep_stddevs = analysis.calculate_lep_and_lep_stddev(
            fails=[100, 20, 5],
            shots=1000
        )
        assert leps.tolist() == true_leps
        assert [round(l_s, 8) for l_s in lep_stddevs] == true_lep_stddevs

    def test_calculate_lep_returns_correct_values_with_scalars(self):
        true_lep = 0.1
        true_lep_stddev = 0.00948683  # copied from above
        lep, lep_stddev = analysis.calculate_lep_and_lep_stddev(fails=100, shots=1000)
        np.testing.assert_allclose(lep, true_lep)
        np.testing.assert_allclose(lep_stddev, true_lep_stddev, atol=1e-8)


class TestGetLambdaFit:
    def test_get_lambda_fit_returns_correct_values(self):
        true_lep_fit = [0.000201, 0.000039, 0.00000758]
        lep_fit = analysis.get_lambda_fit(
            distances=[5, 7, 9],
            lep_per_round=[1.992e-04, 4.314e-05, 7.556e-06],
            lep_stddev_per_round=[1.99579718e-05, 9.28881002e-06, 3.88728658e-07],
        )
        assert pytest.approx(lep_fit, rel=0.002) == true_lep_fit
