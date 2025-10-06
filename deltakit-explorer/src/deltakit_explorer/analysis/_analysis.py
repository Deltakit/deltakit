# (c) Copyright Riverlane 2020-2025.
"""`analysis` module aggregates client-side analytical tools.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import floor
import warnings

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from deltakit_explorer._utils._logging import Logging


def get_exp_fit(
    logical_fails_all_rounds: npt.NDArray[np.int_] | list[int],
    shots_all_rounds: npt.NDArray[np.int_] | list[int],
    all_rounds: list[int],
    interpolation_points: int = 26,
) -> tuple[
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Implement logical error rate fit as described in
    https://arxiv.org/pdf/2310.05900.pdf (p.40) and
    https://arxiv.org/pdf/2207.06431.pdf (p.21). The first round (`r=0`)
    data points are excluded as the error suppression is stronger there
    than in consecutive rounds.

    Args:
        logical_fails_all_rounds (npt.NDArray | List[int]):
            Number of logical failures per round of experiment.
        shots_all_rounds (npt.NDArray | List[int]):
            Number of shots per round of experiment.
        all_rounds (List[int]):
            The number of rounds corresponding to each data
            point, e.g. [1, 2, 4, 8, 16].
        interpolation_points: int
            How many points to use in the interpolated plot.
            Default is 26.

    Returns:
        Tuple[float, npt.NDArray, npt.NDArray, npt.NDArray]:
            A tuple consisting of
            - an epsilon parameter value (error rate per round);
            - an interpolated graph (2 value), computed with this value;
            - error bars value.

    Examples:
        Calling the method::

            eps, x, y, yerr = Analysis.get_exp_fit(
                logical_fails_all_rounds=[20, 20, 30, 40, 50],
                shots_all_rounds=[500, 500, 500, 500, 500],
                all_rounds=[1, 3, 5, 7, 9],
                interpolation_points=25+1,
            )

    """
    query_id = Logging.info_and_generate_uid(locals())
    try:
        # Calculate logical fidelity (F = 1 - 2 * p_err) and its standard error
        logical_perr_per_round = np.array(logical_fails_all_rounds) / np.array(
            shots_all_rounds
        )
        fidelity = 1 - 2 * logical_perr_per_round
        assert not np.any(
            fidelity < 0.0
        ), f"Fidelity values (1-2*p) should be non-negative, but were {fidelity}."

        yerr = np.sqrt(
            logical_perr_per_round * (1 - logical_perr_per_round) / shots_all_rounds
        )

        # Take base-10 log of fidelity and find
        # the best linear fit
        sigma = np.abs(yerr / fidelity)[1:]
        # Where `sigma = 0`, a weight will be infinite. This will cause
        # a warning and error in `polyfit`. The error is enough, so let's
        # suppress the warnings. If the error is considered confusing, we
        # can catch and re-raise with more information.
        with np.errstate(divide="ignore"):
            w = 1 / sigma
        x = np.array(all_rounds[1:])
        y = np.log10(fidelity[1:])

        # cov=True may return the covariance matrix
        with np.errstate(invalid="ignore"):
            pf = np.polyfit(
                x=x,
                y=y,
                deg=1,
                w=w,
            )

        # Exponentiate curve fit params to obtain decay rate
        # epsilon and initial fidelity f_0
        epsilon = float(0.5 * (1.0 - 10.0 ** pf[0]))
        f_0 = float(10.0 ** pf[1])

        # Interpolate x values across the range of available rounds
        # and calculate y values based on an exponential decay with
        # rate epsilon and initial fidelity f_0.
        rounds_interpolated = np.linspace(
            all_rounds[0], all_rounds[-1], interpolation_points,
            dtype=np.float64,
        )
        y_interpolated = [f_0 * (1 - 2 * epsilon) ** r for r in rounds_interpolated]
        probs_interpolated = (1.0 - np.array(y_interpolated, dtype=np.float64)) * 0.5

        return epsilon, rounds_interpolated, probs_interpolated, yerr
    except Exception as ex:
        Logging.error(ex, query_id)
        raise

@dataclass(frozen=True)
class LogicalErrorRatePerRoundResults:
    """Named-tuple-like class containing computation results from
    :func:`compute_logical_error_per_round`.

    Attributes:
        leppr (float): Logical Error Probability Per Round (LEPPR).
        leppr_stddev (float): LEPPR standard deviation.
        spam_error (float): computed SPAM error-rate.
        spam_error_stddev (float): SPAM error-rate standard deviation.
        leppr_stddev_propagated (float): standard deviation due to the propagation of
            individual logical error probabilities standard deviations used to estimate
            the LEPPR.
        leppr_stddev_fit (float): standard deviation due to the linear fit involved in
            LEPPR computation.
        spam_error_stddev_propagated (float): standard deviation due to the propagation
            of individual logical error probabilities standard deviations used to
            estimate the SPAM error-rate.
        spam_error_stddev_fit (float): standard deviation due to the linear fit involved
            in SPAM error-rate computation.

    Note:
        attributes ending in ``_stddev_propagated`` or ``_stddev_fit`` are internal
        estimations that might be useful to understand the contribution of each process
        to the final standard-deviation estimation.
    """
    leppr: float
    leppr_stddev: float
    spam_error: float
    spam_error_stddev: float

    leppr_stddev_propagated: float
    leppr_stddev_fit: float
    spam_error_stddev_propagated: float
    spam_error_stddev_fit: float


def compute_logical_error_per_round(
    num_failed_shots: npt.NDArray[np.int_] | Sequence[int],
    num_shots: npt.NDArray[np.int_] | Sequence[int],
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    *,
    force_include_single_round: bool = False,
) -> LogicalErrorRatePerRoundResults:
    """Compute the logical error-rate per round from different logical error-rate
    computations.

    This function implements the method described in:

    1. https://arxiv.org/pdf/2310.05900.pdf (p.40)
    2. https://arxiv.org/pdf/2207.06431.pdf (p.21)
    3. https://arxiv.org/pdf/2505.09684.pdf (p.8)

    to recover an estimator of the logical error-rate per round from the estimated
    values of logical error-rates for several round durations.

    Args:
        num_failed_shots (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the number of failed shots for each of
            the number of rounds in ``num_rounds``. Should be the same length as
            ``num_rounds``.
        num_shots (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the total number of shots for each of
            the number of rounds in ``num_rounds``. Should be the same length as
            ``num_rounds``.
        num_rounds (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``. Any value
            inferior to 1 (``< 1``) is automatically removed from this list along with
            the corresponding values in ``num_shots`` and ``num_failed_shots``. Any
            value equal to 1 is removed from this list along with the corresponding
            values in ``num_shots`` and ``num_failed_shots`` iff
            ``force_include_single_round`` is ``False``. If only one data-point is
            provided (or left after the removal process described just before), the SPAM
            error is assumed to be ``0`` and an estimation will still be returned.

            Heuristically, to increase the returned estimation precision, you should try
            to provide data for rounds such that the estimated logical error-rate for
            the number of rounds ``max(num_rounds)`` is approximately ``0.4``. This
            ``0.4`` value has been set to reduce fitting errors.
        force_include_single_round (bool):
            if ``True``, data obtained from 1-round experiment will be used in the
            computation if provided in ``num_rounds``. Default to ``False`` which
            results in 1-round data being ignored due to boundary effects that affect
            the final estimation. See https://arxiv.org/pdf/2207.06431.pdf (p.21).

    Returns:
        LEPPRResults: detailed results of the computation.

    Examples:
        Calculating per-round logical error probability and its standard deviation
        given number of fails, and number of shots for several rounds::

            res = compute_logical_error_per_round(
                num_failed_shots=[34, 151, 356],
                num_shots=[500000] * 3,
                num_rounds=[2, 4, 6],
            )
            leppr, leppr_stddev = res.leppr, res.leppr_stddev
            spam, spam_stddev = res.spam_error, res.spam_error_stddev

    """
    # Get the inputs as numpy arrays.
    # Sanitization: also make sure that the inputs are sorted.
    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    num_failed_shots = np.asarray(num_failed_shots)[isort]
    num_shots = np.asarray(num_shots)[isort]

    # Check that we do not have duplicate data for the same number of rounds as that
    # will confuse the numerical methods used in this function.
    unique_counts = np.unique_counts(num_rounds)
    non_unique_entries_mask = unique_counts.counts > 1
    if np.any(non_unique_entries_mask):
        non_unique_values = unique_counts.values[non_unique_entries_mask].tolist()
        raise RuntimeError(
            "Multiple entries were provided for the following number of rounds: "
            f"{non_unique_values}. This is not supported. Please make sure you only "
            "provide one entry per number of rounds."
        )

    # Check that we do not have any num_rounds <= 0 entry.
    while num_rounds[0] <= 0:
        warnings.warn(
            f"Found an invalid number of rounds: {num_rounds[0]}. Number of rounds "
            "should be >= 1."
        )
        num_rounds = num_rounds[1:]
        num_failed_shots = num_failed_shots[1:]
        num_shots = num_shots[1:]

    # Filter out the r == 1 input if not forced to include it by the user.
    if num_rounds[0] == 1 and not force_include_single_round:
        num_rounds = num_rounds[1:]
        num_failed_shots = num_failed_shots[1:]
        num_shots = num_shots[1:]

    if np.any(num_failed_shots == 0):
        raise RuntimeError(
            "Got an experiment without any errors. You should increase the number of "
            "shots to have at least one error, else the problem is ill-formed."
        )

    logical_error_rates = num_failed_shots / num_shots
    fidelities = 1 - 2 * logical_error_rates

    if np.any(fidelities <= 0):
        raise RuntimeError(
            "Got estimations of logical error-rates above 0.5. That is not supported "
            "by this function. Please reduce your number of rounds. Estimated logical "
            f"error-rates: {logical_error_rates}."
        )
    # Check if the heuristic guideline on the number of rounds is verified.
    max_logical_error_rate = np.max(logical_error_rates)
    if max_logical_error_rate < 0.2:
        warnings.warn(
            f"The maximum estimated logical error-rate ({max_logical_error_rate}) is "
            "below 0.2. The returned estimation might be better if you add data with "
            "more rounds such that the maximum estimated logical error-rate is closer "
            "to 0.4."
        )

    # We want to do a linear regression on the log values of fidelity, and obtain the
    # per-round error rate like that.
    # Applying the logarithm function will change non-uniformly the standard deviation
    # of each variable, which makes the standard linear regression estimator biased. The
    # best linear unbiased estimator in that case is obtained by solving a weighted
    # least square problem where the weights corresponds to the reciprocal of the
    # variance of each observation.
    # See https://en.wikipedia.org/wiki/Weighted_least_squares.
    logfidelity = np.log(fidelities)
    # We approximate the standard deviation with an error propagation analysis. This
    # method has been tested against scipy and returns similar results.
    # Alias for more readability
    pl = logical_error_rates
    pl_stddev = np.sqrt(pl * (1 - pl) / num_shots)
    logfidelities_stddev = 2 * pl_stddev / fidelities

    # If the user only provided one data point, we add a noiseless data-point assuming
    # that the SPAM error is 0.
    if logfidelity.size == 1:
        warnings.warn(
            "Only one data-point provided for logical error probability per round. "
            "Continuing computation assuming that SPAM error is negligible."
        )
        num_rounds = np.hstack([[0], num_rounds])
        logfidelity = np.hstack([[0], logfidelity])
        # We cannot set the stddev to 0 here because curve_fit will divide by that
        # quantity, so make it very small.
        logfidelities_stddev = np.hstack([[1e-12], logfidelities_stddev])

    # Note that the covariance matrix is used later to estimate the logical error-rate
    # per round standard deviation.
    (slope, offset), cov = curve_fit(
        lambda x, s, o: s * x + o,
        num_rounds,
        logfidelity,
        sigma=logfidelities_stddev,
        absolute_sigma=True,
        # If the error probabilities are exactly 0, the solution should be (0, 0).
        # Because we expect the error probabilities to be close to 0, start from (0, 0)
        # as a first estimate.
        p0=(0, 0),
        # Both slope and offset are used to recover a probability with the expression
        # p = (1 - np.exp(value)) / 2. Because a probability needs to be in [0, 1], we
        # have that value <= np.log(1).
        # Note: even though the below bounds are valid, setting bounds changes the
        # default optimisation method from "lm" to "trf". There is at least one
        # real-world example where setting those bounds led to incorrect results, so not
        # including them for the moment.
        # bounds=((-np.inf, -np.inf), (np.log(1), np.log(1))),
    )

    estimated_logical_error_per_round = float((1 - np.exp(slope)) / 2)
    # Compute the standard R2 (Coefficient of determination) using the formula
    # ``R2 = 1 - SSE / SST`` where SSE is the Sum of Squares Error and SST is the Sum of
    # Square Total that are computed below.
    sse = np.sum((logfidelity - offset - slope * num_rounds) ** 2)
    sst = np.sum((logfidelity - np.mean(logfidelity)) ** 2)
    r2 = float(1 - sse / sst)
    if abs(r2) < 0.98:
        warnings.warn(
            f"Got a R2 value of {r2} < 0.98. Estimation might be imprecise. Increasing "
            "the number of shots or re-performing the computation might help in removing "
            "this warning."
        )

    # Following https://arxiv.org/pdf/2505.09684v1 (Methods - Extracting logical error
    # per cycle, page 8) we estimate the variance on the logical error-rate per cycle
    # (named Perrc below) using the formula
    #      sigma(Perrc) = (1 - Perrc) * sigma(slope)
    # The standard deviation on the linear fit parameters can be obtained through the
    # covariance matrix diagonal entries.
    slope_stddev, offset_stddev = np.sqrt(np.diagonal(cov))
    estimated_logical_error_per_round_stddev = (
        (1 - 2 * estimated_logical_error_per_round) * slope_stddev / 2
    )

    # Else
    estimated_spam_error = float((1 - np.exp(offset)) / 2)
    estimated_spam_error_stddev = (1 - 2 * estimated_spam_error) * offset_stddev / 2
    return LogicalErrorRatePerRoundResults(
        estimated_logical_error_per_round,
        estimated_logical_error_per_round_stddev,
        estimated_spam_error,
        estimated_spam_error_stddev,
        (1 - 2 * estimated_logical_error_per_round) / 2,
        slope_stddev,
        (1 - 2 * estimated_spam_error) / 2,
        offset_stddev,
    )

def simulate_different_round_numbers_for_lep_per_round_estimation(
    simulator: Callable[[int], tuple[int, int]],
    initial_round_number: int = 2,
    next_round_number_func: Callable[[int], int] = lambda x: 2 * x,
    maximum_round_number: int | None = None,
    heuristic_logical_error_lower_bound: float = 0.25,
    heuristic_logical_error_upper_bound: float = 0.45,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Compute QEC results to estimate the logical error-rate per round.

    This function aims at encapsulating the practical knowledge about logical error-rate
    per round computation to help any user computing the required logical error-rates
    for useful number of rounds.

    It repeatedly calls ``simulator`` with a number of rounds growing according to
    ``next_round_number_func``, starting from ``initial_round_number``,
    until the logical error-rate is above ``heuristic_logical_error_lower_bound``. If
    the final step returned a logical error-rate above
    ``heuristic_logical_error_upper_bound``, the algorithm then goes backward and
    replaces that last value with the first one under that limit.

    Args:
        simulator (Callable[[int], tuple[int, int]]):
            a callable that returns a tuple ``(num_fails, num_shots)`` from a number of
            rounds given as input.
        initial_round_number (int): initial value for the geometric series that will be
            used to generate the number of rounds.
        next_round_number_func (Callable[[int], int]): function used to compute the
            next round number that should be tested. Default to a linear scaling up to
            500 rounds and then an exponential scaling. The initial linear scaling is to
            avoid the nearby points generated at the beginning of the exponential
            scaling whereas the final exponential scaling is to avoid spending too much
            time if the noise is really low.
        maximum_round_number (int): if set, this function will stop once the next
            number of rounds (computed with ``next_round_number_func``) is above that
            threshold. If not set, only the other stopping criterions apply.
        heuristic_logical_error_lower_bound (float): minimum target logical error-rate
            for the final round. Might not be verified by the return of this function if
            ``maximum_round_number`` is set and reached before that minimum threshold.
        heuristic_logical_error_upper_bound (float): maximal target logical error-rate
            for the final round. Should be set sufficiently below ``0.5`` such that the
            uncertainties (mostly due to finite sampling) on the computed logical
            error-rate (LER) are low enough to not introduce a plateau in the log-plot
            of the fidelity log(F) = log(1 - 2*LER). Experimentally, ``0.45`` seems to
            check that.

    Returns:
        tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
            A tuple consisting of
            - the different number of rounds corresponding to the two other entries,
            - the number of failed shots for the corresponding number of rounds,
            - the total number of shots for the corresponding number of rounds.

    Examples:
        Calculating per-round logical error probability and its standard deviation
        given number of fails, and number of shots for several rounds::

            def perfect_simulator(num_rounds: int) -> tuple[int, int]:
                error_per_round: float = 0.001
                total_error: float = (1 - error_per_round) ** num_rounds
                num_shots: int = 100_000
                num_fails = total_error * num_shots
                return num_fails, num_shots


            nrounds, nfails, nshots = (
                simulate_different_round_numbers_for_lep_per_round_estimation(
                    simulator=perfect_simulator,
                    initial_round_number=2,
                    geometric_factor=1.7,
                )
            )
    """
    if maximum_round_number is None:
        maximum_round_number = 2**30

    nrounds: list[int] = [initial_round_number]
    nfails: list[int] = []
    nshots: list[int] = []

    nfail, nshot = simulator(nrounds[-1])
    nfails.append(nfail)
    nshots.append(nshot)

    # Generate experiments until the number of repetitions is large enough (which is
    # heuristically determined as
    # ``logical error-rate > heuristic_logical_error_lower_bound``).
    while (nfails[-1] / nshots[-1]) < heuristic_logical_error_lower_bound:
        new_round_number = next_round_number_func(nrounds[-1])
        if new_round_number > maximum_round_number:
            break
        nrounds.append(new_round_number)
        nfail, nshot = simulator(nrounds[-1])
        nfails.append(nfail)
        nshots.append(nshot)

    # We do not want to include logical error-rates above
    # ``heuristic_logical_error_upper_bound``.
    # We go back using smaller steps until we find a last point that is over
    # ``heuristic_logical_error_lower_bound`` but under
    # ``heuristic_logical_error_upper_bound``.
    maximum_number_of_backward_steps: int = 5
    backward_arithmetic_factor: int = int(floor(
        (nrounds[-1] - nrounds[-2]) / (maximum_number_of_backward_steps + 1)
    ))
    while (nfails[-1] / nshots[-1]) > heuristic_logical_error_upper_bound:
        out_of_bound_round_value = nrounds[-1]
        nrounds, nfails, nshots = nrounds[:-1], nfails[:-1], nshots[:-1]
        nrounds.append(out_of_bound_round_value - backward_arithmetic_factor)
        nfail, nshot = simulator(nrounds[-1])
        nfails.append(nfail)
        nshots.append(nshot)

    return np.asarray(nrounds), np.asarray(nfails), np.asarray(nshots)


def calculate_lep_and_lep_stddev(
    fails: npt.NDArray[np.int_] | Sequence[int] | int,
    shots: npt.NDArray[np.int_] | Sequence[int] | int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate the logical error probability and its standard deviation.

    Args:
        fails (npt.NDArray[np.int\\_] | Sequence[int] | int):
            The number of logical failures.
        shots (npt.NDArray[np.int\\_] | Sequence[int] | int):
            The number of shots the experiment was run for.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            A tuple consisting of
            - the logical error probability;
            - the standard deviation of the logical error probability.

    Examples:
        Calculating logical error probability and standard deviation
        given number of fails, and number of shots::

            lep, lep_stddev = Analysis.calculate_lep_and_lep_stddev(
                fails=[498, 151, 34],
                shots=[500000] * 3,
            )
    """
    fails, shots = np.asarray(fails), np.asarray(shots)
    query_id = Logging.info_and_generate_uid(locals())
    lep = fails / shots
    if np.any(lep <= 0):
        error = ValueError("Must have > 0 fails to calculate"
                            " logical error probability.")
        Logging.error(error, query_id)
        raise error
    lep_stddev = np.sqrt(lep * (1 - lep) / shots)

    return lep, lep_stddev


def calculate_lambda_and_lambda_stddev(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float],
) -> tuple[float, float]:
    """
    Calculate the error suppression factor (lambda) and its standard deviation.

    Accepts the logical error probability (LEP) per round, which may be approximated
    as LEP / num_rounds (for small LEP), and equally for lep_stddev.

    By providing the logical error probability for increasing code distances,
    one can obtain an estimate for how error suppression scales with distances.
    A minimum of three distances is required to calculate lambda.
    Note that lambda is a "rule of thumb". This approximation is unreliable near
    threshold and for low code distances.

    Args:
        distances (npt.NDArray[np.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_stddev_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The standard deviation of the logical error probabilities per round.

    Returns:
        Tuple[float, float]:
            A tuple consisting of::
            - the exponential error rate suppression factor, Lambda;
            - the standard deviation of the Lambda value;


    Examples:
        Fitting the Lambda value given information for 5, 7, and 9 round of
        a QEC experiment::

            lambda_, lambda_stddev = Analysis.calculate_lambda_and_lambda_stddev(
                distances=[5, 7, 9],
                lep_per_round=[1.992e-04, 4.314e-05, 7.556e-06],
                lep_stddev_per_round=[1.2e-05, 9.3e-06, 3.9e-06],
            )
            lambda_, lambda_stddev = res.lambda_, res.lambda_stddev

    """
    query_id = Logging.info_and_generate_uid(locals())
    # Make sure that the inputs are numpy arrays sorted by distance
    isort = np.argsort(distances)
    distances = np.asarray(distances)[isort]
    lep_per_round = np.asarray(lep_per_round)[isort]
    lep_stddev_per_round = np.asarray(lep_stddev_per_round)[isort]

    if len(distances) < 3:
        error = ValueError("Minimum of 3 distances are required to calculate lambda.")
        Logging.error(error, query_id)
        raise error
    params, cov = np.polyfit(
        x=distances,
        y=np.log(lep_per_round),
        deg=1,
        w=[lep_pr / lep_std_pr for lep_pr, lep_std_pr in zip(
            lep_per_round, lep_stddev_per_round)],
        cov='unscaled'
    )
    lambda_value = float(np.exp(-2 * params[0]))
    lambda_value_stddev = float(lambda_value * 2 * np.sqrt(np.diag(cov))[0])

    # See Fig. S15 of Supplementary information of "Quantum error correction below the
    # surface code threshold" (https://www.nature.com/articles/s41586-024-08449-y#Sec8).
    if lambda_value < 1.5 and min(distances) < 5:
        Logging.warn(
            "Lambda estimation is unreliable at low code distances and low "
            "values of lambda. Please use distance 5 as a minimum.",
            query_id,
        )

    return lambda_value, lambda_value_stddev


def get_lambda_fit(
    distances: list[int],
    lep_per_round: list[float],
    lep_stddev_per_round: list[float]
) -> npt.NDArray[np.float64]:
    """
    Get the best fit line with gradient lambda for plotting purposes.

    Accepts the logical error probability (LEP) per round, which may be approximated
    as LEP / num_rounds (for small LEP), and equally for lep_stddev.

    Args:
        distances (List[int]):
            The distances of the code.
        lep_per_round (List[float]):
            The logical error probabilities per round.
        lep_stddev_per_round (List[float]):
            The standard deviation of the logical error probabilities per round.

    Returns:
        npt.NDArray:
            The best fit line of log(lep_per_round) vs distance, from which
            the gradient is lambda.

    Examples:
        Fit exponential curve given logical error probability for 5, 7, and
        9 rounds of QEC experiment::

            lep_per_round_fit = Analysis.get_lambda_fit(
                distances=[5, 7, 9],
                lep_per_round=[1.992e-04, 4.314e-05, 7.556e-06],
                lep_stddev_per_round=[1.2e-05, 9.3e-06, 3.9e-06]
            )

    """
    params, _ = np.polyfit(
        x=distances,
        y=np.log(lep_per_round),
        deg=1,
        w=[lep_pr / lep_std_pr for lep_pr, lep_std_pr in zip(
            lep_per_round, lep_stddev_per_round)],
        cov='unscaled'
    )

    return np.exp(params[0]*np.array(distances) + params[1])
