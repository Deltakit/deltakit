import itertools
import re
from math import sqrt

import numpy as np
import pytest

from deltakit_explorer.analysis import (
    calculate_lep_and_lep_stddev,
    compute_logical_error_per_round,
)


class TestLEPPerRoundComputation:
    def test_raise_exception_for_mismatch_inputs(self) -> None:
        fails = [498, 151, 34]
        shots = [500000] * 3

        with pytest.raises(ValueError, match="do not match lengths."):
            calculate_lep_and_lep_stddev(fails=[498, 151], shots=shots)

        with pytest.raises(ValueError, match="do not match lengths."):
            calculate_lep_and_lep_stddev(fails=fails, shots=[500000] * 2)

        with pytest.raises(ValueError, match="must be strictly positive"):
            calculate_lep_and_lep_stddev(fails=[498, 151, -34], shots=shots)

        with pytest.raises(ValueError, match="must be strictly positive"):
            calculate_lep_and_lep_stddev(
                fails=fails, shots=[500_000, -500_000, 500_000]
            )

    @pytest.mark.parametrize(
        ("leppr", "spam_error"),
        list(itertools.product((1e-5, 1e-4, 1e-3, 1e-2), (1e-5, 1e-4, 1e-3, 1e-2))),
    )
    def test_on_synthetic_inputs(
        self,
        leppr: float,
        spam_error: float,
        random_generator: np.random.Generator,
    ) -> None:
        f_0 = 1 - 2 * spam_error
        rounds = np.arange(2, np.ceil(np.log(0.3 / f_0) / np.log(1 - 2 * leppr)), 2)
        fidelities = f_0 * (1 - 2 * leppr) ** rounds
        lep = (1 - fidelities) / 2
        lep *= 1 - random_generator.normal(0, 1e-4, lep.size)
        lep_stddev = lep * (1 - lep) / sqrt(100_000)

        res = compute_logical_error_per_round(rounds, lep, lep_stddev)
        # Test that the estimated quantities are within 3*sigma of the real one.
        assert pytest.approx(res.leppr, abs=4 * res.leppr_stddev) == leppr
        assert (
            pytest.approx(res.spam_error, abs=4 * res.spam_error_stddev) == spam_error
        )
        assert isinstance(res.leppr, float)
        assert isinstance(res.leppr_stddev, float)
        assert isinstance(res.spam_error, float)
        assert isinstance(res.spam_error_stddev, float)

    @pytest.mark.parametrize(
        "rounds", [[0, 1, 2, 3, 4, 3], [-2, 1, 1, 3, 3, 3], [4, 8, 4, 0, 5]]
    )
    def test_raises_when_duplicated_round_number(self, rounds: list[int]) -> None:
        f_0, leppr = 0.999, 0.001
        nprounds = np.asarray(rounds)
        fidelities = f_0 * (1 - 2 * leppr) ** nprounds
        lep = (1 - fidelities) / 2
        lep_stddev = lep * (1 - lep) / sqrt(100_000)

        message = "^Multiple entries were provided for the following number of rounds:"
        with pytest.raises(RuntimeError, match=message):
            compute_logical_error_per_round(nprounds, lep, lep_stddev)

    @pytest.mark.parametrize(
        "rounds", [[0, 1, 2, 3, 4], [-1, 4, 5, 7], [8, 4, 0, 5, -1, -34]]
    )
    def test_raises_when_invalid_round_number(self, rounds: list[int]) -> None:
        f_0, leppr = 0.999, 0.001
        nprounds = np.asarray(rounds)
        fidelities = f_0 * (1 - 2 * leppr) ** nprounds
        lep = (1 - fidelities) / 2
        lep_stddev = lep * (1 - lep) / sqrt(100_000)

        with pytest.warns(UserWarning) as reporter:
            compute_logical_error_per_round(nprounds, lep, lep_stddev)
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
        lep, lep_stddev = calculate_lep_and_lep_stddev(num_failed_shots, num_shots)
        res = compute_logical_error_per_round(num_rounds, lep, lep_stddev)

        assert pytest.approx(res.leppr, 3 * res.leppr_stddev) == 0.11912

    def test_warn_when_max_lep_is_too_small(self):
        message = (
            "^The maximum estimated logical error probability "
            r"\([^\)]+\) is below 0.2.*"
        )
        with pytest.warns(UserWarning, match=message):
            compute_logical_error_per_round(
                [2, 4, 6], [0.01, 0.02, 0.03], [1e-12, 1e-12, 1e-12]
            )

    def test_warn_when_invalid_lep(self):
        message = (
            r"Found at least one invalid \(i.e., > 0.5\) logical error probability. "
            "Ignoring all the provided logical error probabilities above 0.5."
        )
        with pytest.warns(UserWarning, match=message):
            compute_logical_error_per_round(
                [2, 4, 6], [0.2, 0.4, 0.6], [1e-12, 1e-12, 1e-12]
            )

    def test_raise_on_empty_input(self):
        msg = "No valid data was provided\\..+"
        with pytest.raises(ValueError, match=msg):
            compute_logical_error_per_round([], [], [])

        with pytest.raises(ValueError, match=msg), pytest.warns(UserWarning):
            compute_logical_error_per_round([0], [0.1], [0.001])

    def test_warn_when_linear_fit_is_bad(self):
        f_0 = 1 - 0.01
        rounds = np.arange(2, 61, 5)
        # Non-constant logical error probability per round that should trigger the R2
        # check.
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
        fidelities = f_0 * (1 - 2 * leppr) ** rounds
        lep = (1 - fidelities) / 2
        lep_stddev = lep * (1 - lep) / sqrt(100_000)
        message = r"Got a R2 value of -?[0-9]+\.[0-9]+ < 0.98."
        with pytest.warns(UserWarning, match=message):
            compute_logical_error_per_round(rounds, lep, lep_stddev)

    @pytest.mark.parametrize("leppr", [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
    def test_single_point_fit(self, leppr: float) -> None:
        rounds = 30
        fidelity = (1 - 2 * leppr) ** rounds
        lep = (1 - fidelity) / 2
        lep_stddev = lep * (1 - lep) / sqrt(100_000)

        with pytest.warns(UserWarning) as warning_collector:
            res = compute_logical_error_per_round([rounds], [lep], [lep_stddev])
        expected_message = (
            "^Only one valid data-point provided for logical error probability per "
            "round. Continuing computation assuming that SPAM error is negligible.$"
        )
        assert any(
            re.match(expected_message, str(w.message)) for w in warning_collector
        ), "Expected to get a warning on single data-point being used."
        # Test that the estimated quantities are within 3*sigma of the real one.
        assert pytest.approx(res.leppr, abs=3 * res.leppr_stddev) == leppr
        assert pytest.approx(res.spam_error, abs=3 * res.spam_error_stddev) == 0
        assert isinstance(res.leppr, float)
        assert isinstance(res.leppr_stddev, float)


class TestCalculateLep:
    def test_calculate_lep_no_fails_raises(self):
        fails = [500, 200, 25, 0]
        shots = 50000
        with pytest.raises(ValueError, match="do not match lengths."):
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
            fails=[100, 20, 5], shots=[1000] * 3
        )
        np.testing.assert_allclose(leps, true_leps)
        np.testing.assert_allclose(lep_stddevs, true_lep_stddevs, atol=1e-8)
