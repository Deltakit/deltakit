import itertools
import re

import numpy as np
import pytest

from deltakit_explorer.analysis import compute_logical_error_per_round


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
        message = "^Got estimations of logical error probability above 0.5.*"
        with pytest.raises(RuntimeError, match=message):
            compute_logical_error_per_round(
                [shots // 2 + 1] * 3, [shots] * 3, [2, 4, 6]
            )

    def test_warn_when_max_lep_is_too_small(self):
        shots = 100_000
        message = (
            "^The maximum estimated logical error probability "
            r"\([^\)]+\) is below 0.2.*"
        )
        with pytest.warns(UserWarning, match=message):
            compute_logical_error_per_round([2460, 4343, 6151], [shots] * 3, [2, 4, 6])

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
