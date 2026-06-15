import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import floor

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from uncertainties import correlated_values, ufloat, unumpy
from uncertainties.umath import exp as uexp

from deltakit_explorer.analysis._estimate import Estimate


def leppr_from_single_point(lep: float, lep_stddev: float, rounds: int) -> Estimate:
    """Propagate uncertainty for the single-point LEPPR.

    See:
        - Eq. 4, arXiv:2310.05900.

    Args:
        lep: Logical error probability.
        lep_stddev: Standard deviation of the logical error probability.
        rounds: Number of QEC rounds.

    Returns:
        LEPPR and its standard deviation.
    """
    uncertain_lep = ufloat(lep, lep_stddev)
    leppr = (1 - (1 - 2 * uncertain_lep) ** (1 / rounds)) / 2
    return Estimate.from_ufloat(leppr)


def log_fidelity_stddev(
    lep: npt.NDArray[np.floating] | Sequence[float],
    lep_stddev: npt.NDArray[np.floating] | Sequence[float],
) -> npt.NDArray[np.float64]:
    """Standard deviation of ``log(1 - 2*lep)`` for the weighted least-squares fit.

    Args:
        lep: Logical error probabilities.
        lep_stddev: Standard deviation of each logical error probability.

    Returns:
        Standard deviation of log-fidelity for each input point.
    """
    lep_arr = np.asarray(lep, dtype=np.float64)
    std_arr = np.asarray(lep_stddev, dtype=np.float64)
    uncertain_log_fidelity = unumpy.log(1 - 2 * unumpy.uarray(lep_arr, std_arr))
    return unumpy.std_devs(uncertain_log_fidelity).astype(np.float64)


def epsilon_and_spam_from_log_fit(
    slope: float,
    offset: float,
    cov: npt.NDArray[np.floating],
) -> tuple[Estimate, Estimate]:
    """LEPPR and SPAM error from the correlated log-linear fit parameters.

    Following https://arxiv.org/pdf/2505.09684v1 (Methods - Extracting logical
    error per cycle, page 8), the per-round error and the SPAM error are recovered
    as ``(1 - exp(slope)) / 2`` and ``(1 - exp(offset)) / 2``, with standard
    deviations propagated from the fit covariance matrix.

    Args:
        slope: Slope from the log-fidelity linear fit.
        offset: Offset from the log-fidelity linear fit.
        cov: Covariance matrix of the fit parameters.

    Returns:
        ``(Estimate(leppr, leppr_stddev), Estimate(spam_error, spam_error_stddev))``.
    """
    uncertain_slope, uncertain_offset = correlated_values([slope, offset], cov)
    uncertain_leppr = (1 - uexp(uncertain_slope)) / 2
    uncertain_spam = (1 - uexp(uncertain_offset)) / 2
    return (
        Estimate.from_ufloat(uncertain_leppr),
        Estimate.from_ufloat(uncertain_spam),
    )


@dataclass(frozen=True)
class LogicalErrorProbabilityPerRoundData:
    """Container class to hold `compute_logical_error_per_round` results.

    Attributes:
        leppr: Logical Error Probability Per Round (LEPPR).
        leppr_stddev: LEPPR standard deviation.
        num_rounds: Array containing the number of rounds.
        spam_error: Computed SPAM error probability.
        spam_error_stddev: SPAM error probability standard deviation.
    """

    leppr: float
    leppr_stddev: float
    num_rounds: npt.NDArray[np.int_]
    spam_error: float
    spam_error_stddev: float


def compute_logical_error_per_round(
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probabilities: npt.NDArray[np.floating] | Sequence[float],
    logical_error_probabilities_stddev: npt.NDArray[np.floating] | Sequence[float],
    *,
    force_include_single_round: bool = False,
) -> LogicalErrorProbabilityPerRoundData:
    """Compute the logical error probability per round from different logical error
    probability computations.

    This function implements the method described in:

    1. https://arxiv.org/pdf/2310.05900.pdf (p.40)
    2. https://arxiv.org/pdf/2207.06431.pdf (p.21)
    3. https://arxiv.org/pdf/2505.09684.pdf (p.8)

    to recover an estimator of the logical error probability per round from the
    estimated values of logical error probabilities for several round durations.

    Args:
        num_rounds (npt.NDArray[numpy.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``logical_error_probabilities`` and
            ``logical_error_probabilities_stddev``. Any value below 1 (``< 1``) is
            automatically removed from this list along with the corresponding values in
            ``logical_error_probabilities`` and ``logical_error_probabilities_stddev``.
            Any value equal to 1 is removed from this list along with the corresponding
            values in ``logical_error_probabilities`` and
            ``logical_error_probabilities_stddev`` iff ``force_include_single_round`` is
            ``False``. If only one data-point is provided (or left after the removal
            process described just before), the SPAM error is assumed to be ``0`` and an
            estimation will still be returned.

            Heuristically, to increase the returned estimation precision, you should try
            to provide data for rounds such that the estimated logical error probability
            for the number of rounds ``max(num_rounds)`` is approximately ``0.4``. This
            ``0.4`` value has been set to reduce fitting errors.
        logical_error_probabilities (npt.NDArray[numpy.floating] | Sequence[float]):
            logical error probabilities computed for each of the provided
            ``num_rounds``. Should be the same length as ``num_rounds``.
        logical_error_probabilities_stddev (npt.NDArray[numpy.floating] | Sequence[float]):
            standard deviation of the logical error probabilities provided in
            ``logical_error_probabilities``. Should be the same length as
            ``num_rounds``.
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
    # Sanitisation: also make sure that the inputs are sorted.
    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    logical_error_probabilities = np.asarray(logical_error_probabilities)[isort]
    logical_error_probabilities_stddev = np.asarray(logical_error_probabilities_stddev)[
        isort
    ]

    # Check that we do not have duplicate data for the same number of rounds as that
    # will confuse the numerical methods used in this function.
    unique_counts = np.unique_counts(num_rounds)
    non_unique_entries_mask = unique_counts.counts > 1
    if np.any(non_unique_entries_mask):
        non_unique_values = unique_counts.values[non_unique_entries_mask].tolist()
        msg = (
            "Multiple entries were provided for the following number of rounds: "
            f"{non_unique_values}. This is not supported. Please make sure you only "
            "provide one entry per number of rounds."
        )
        raise RuntimeError(msg)

    # Check that we do not have any num_rounds <= 0 entry.
    while num_rounds.size > 0 and num_rounds[0] <= 0:
        warnings.warn(
            f"Found an invalid number of rounds: {num_rounds[0]}. Number of rounds "
            "should be >= 1."
        )
        num_rounds = num_rounds[1:]
        logical_error_probabilities = logical_error_probabilities[1:]
        logical_error_probabilities_stddev = logical_error_probabilities_stddev[1:]

    # Filter out the r == 1 input if not forced to include it by the user.
    if num_rounds.size > 0 and num_rounds[0] == 1 and not force_include_single_round:
        num_rounds = num_rounds[1:]
        logical_error_probabilities = logical_error_probabilities[1:]
        logical_error_probabilities_stddev = logical_error_probabilities_stddev[1:]

    # Filter out logical error probabilities above 0.5 as that will lead to negative
    # fidelities.
    invalid_lep_indices = logical_error_probabilities > 0.5
    if np.any(invalid_lep_indices):
        warnings.warn(
            "Found at least one invalid (i.e., > 0.5) logical error probability. "
            "Ignoring all the provided logical error probabilities above 0.5."
        )
        valid_lep_indices = np.logical_not(invalid_lep_indices)
        num_rounds = num_rounds[valid_lep_indices]
        logical_error_probabilities = logical_error_probabilities[valid_lep_indices]
        logical_error_probabilities_stddev = logical_error_probabilities_stddev[
            valid_lep_indices
        ]

    # Checking the validity of the filtered data.
    if num_rounds.size == 0:
        msg = (
            "No valid data was provided. Please ensure that the data provided is "
            "correct. If you provided data, look at the warnings to understand why it "
            "was considered invalid and ignored by this function."
        )
        raise ValueError(msg)

    # If the user only provided one data point, we can use a direct approximate formula
    # without having to call a fitting function.
    if logical_error_probabilities.size == 1:
        warnings.warn(
            "Only one valid data-point provided for logical error probability per "
            "round. Continuing computation assuming that SPAM error is negligible."
        )
        rounds = num_rounds[0]
        lep = float(logical_error_probabilities[0])
        lep_stddev = float(logical_error_probabilities_stddev[0])
        # Implement Eq. (4) from section A.2.2. at page 40 of
        # https://arxiv.org/pdf/2310.05900.
        estimated_logical_error_per_round, estimated_logical_error_per_round_stddev = (
            leppr_from_single_point(lep, lep_stddev, rounds)
        )
        return LogicalErrorProbabilityPerRoundData(
            leppr=estimated_logical_error_per_round,
            leppr_stddev=estimated_logical_error_per_round_stddev,
            num_rounds=rounds,
            spam_error=0,
            spam_error_stddev=0,
        )

    # Check if the heuristic guideline on the number of rounds is verified.
    max_logical_error_probability = np.max(logical_error_probabilities)
    if max_logical_error_probability < 0.2:
        warnings.warn(
            "The maximum estimated logical error probability "
            f"({max_logical_error_probability}) is below 0.2. The returned estimation "
            "might be better if you add data with more rounds such that the maximum "
            "estimated logical error probability is closer to 0.4."
        )

    fidelities = 1 - 2 * logical_error_probabilities
    # We want to do a linear regression on the log values of fidelity, and obtain the
    # per-round error probability like that.
    # Applying the logarithm function will change non-uniformly the standard deviation
    # of each variable, which makes the standard linear regression estimator biased. The
    # best linear unbiased estimator in that case is obtained by solving a weighted
    # least square problem where the weights corresponds to the reciprocal of the
    # variance of each observation.
    # See https://en.wikipedia.org/wiki/Weighted_least_squares.
    logfidelity = np.log(fidelities)
    logfidelities_stddev = log_fidelity_stddev(
        logical_error_probabilities, logical_error_probabilities_stddev
    )

    # Note that the covariance matrix is used later to estimate the logical error
    # probability per round standard deviation.
    (slope, offset), cov = curve_fit(
        lambda x, s, o: s * x + o,
        num_rounds,
        logfidelity,
        sigma=logfidelities_stddev,
        absolute_sigma=True,
        # If the error probabilities are exactly 0, the solution should be (0, 0).
        # Because we expect the error probabilities to be close to 0, start from (0, 0)
        # as a first estimate.
        p0=(0, 0),
        # Both slope and offset are used to recover a probability with the expression
        # p = (1 - numpy.exp(value)) / 2. Because a probability needs to be in [0, 1], we
        # have that value <= numpy.log(1).
        # Note: even though the below bounds are valid, setting bounds changes the
        # default optimisation method from "lm" to "trf". There is at least one
        # real-world example where setting those bounds led to incorrect results, so not
        # including them for the moment.
        # bounds=((-numpy.inf, -numpy.inf), (numpy.log(1), numpy.log(1))),
    )

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
    # per cycle, page 8) we estimate the variance on the logical error probability per
    # round (named Perrc below) using the formula
    #      sigma(Perrc) = (1 - Perrc) * sigma(slope)
    # The standard deviation on the linear fit parameters is obtained from the
    # covariance matrix and propagated by the helper below.
    (
        (estimated_logical_error_per_round, estimated_logical_error_per_round_stddev),
        (
            estimated_spam_error,
            estimated_spam_error_stddev,
        ),
    ) = epsilon_and_spam_from_log_fit(slope, offset, cov)
    return LogicalErrorProbabilityPerRoundData(
        leppr=estimated_logical_error_per_round,
        leppr_stddev=estimated_logical_error_per_round_stddev,
        num_rounds=num_rounds,
        spam_error=estimated_spam_error,
        spam_error_stddev=estimated_spam_error_stddev,
    )


def simulate_different_round_numbers_for_lep_per_round_estimation(
    simulator: Callable[[int], tuple[int, int]],
    initial_round_number: int = 2,
    next_round_number_func: Callable[[int], int] = lambda x: 2 * x,
    maximum_round_number: int | None = None,
    heuristic_logical_error_lower_bound: float = 0.25,
    heuristic_logical_error_upper_bound: float = 0.45,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Compute QEC results to estimate the logical error probability per round.

    This function aims at encapsulating the practical knowledge about logical error
    probability per round computation to help any user computing the required logical
    error probabilities for useful number of rounds.

    It repeatedly calls ``simulator`` with a number of rounds growing according to
    ``next_round_number_func``, starting from ``initial_round_number``,
    until the logical error probability is above
    ``heuristic_logical_error_lower_bound``. If the final step returned a logical error
    probability above ``heuristic_logical_error_upper_bound``, the algorithm then goes
    backward and replaces that last value with the first one under that limit.

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
        heuristic_logical_error_lower_bound (float): minimum target logical error
            probability for the final round. Might not be verified by the return of this
            function if ``maximum_round_number`` is set and reached before that minimum
            threshold.
        heuristic_logical_error_upper_bound (float): maximal target logical error
            probability for the final round. Should be set sufficiently below ``0.5``
            such that the uncertainties (mostly due to finite sampling) on the computed
            logical error probability (LEP) are low enough to not introduce a plateau in
            the log-plot of the fidelity log(F) = log(1 - 2*LEP). Experimentally,
            ``0.45`` seems to check that.

    Returns:
        tuple[npt.NDArray[numpy.int_], npt.NDArray[numpy.int_], npt.NDArray[numpy.int_]]:
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
    # ``logical error probability > heuristic_logical_error_lower_bound``).
    while (nfails[-1] / nshots[-1]) < heuristic_logical_error_lower_bound:
        new_round_number = next_round_number_func(nrounds[-1])
        if new_round_number > maximum_round_number:
            break
        nrounds.append(new_round_number)
        nfail, nshot = simulator(nrounds[-1])
        nfails.append(nfail)
        nshots.append(nshot)

    # We do not want to include logical error probabilities above
    # ``heuristic_logical_error_upper_bound``.
    # We go back using smaller steps until we find a last point that is over
    # ``heuristic_logical_error_lower_bound`` but under
    # ``heuristic_logical_error_upper_bound``.
    maximum_number_of_backward_steps: int = 5
    backward_arithmetic_factor: int = floor(
        (nrounds[-1] - nrounds[-2]) / (maximum_number_of_backward_steps + 1)
    )
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
    """Calculate the logical error probability (lep) and its standard deviation.

    Args:
        fails: The number of logical failures.
        shots: The number of shots the experiment was run for.

    Returns:
        A tuple consisting of the logical error probability
        and its standard deviation.

    Raises:
        ValueError: When inputs do not match lengths or have non-positive entries.

    Examples:
        Calculating logical error probability and standard deviation
        given number of fails, and number of shots:

        >>> lep, lep_stddev = analysis.calculate_lep_and_lep_stddev(
        ...     fails=[498, 151, 34],
        ...     shots=[500000] * 3,
        ... )

    """
    fails = np.asarray([fails]) if isinstance(fails, int) else np.asarray(fails)
    shots = np.asarray([shots]) if isinstance(shots, int) else np.asarray(shots)
    if len(fails) != len(shots):
        msg = "Input data do not match lengths."
        raise ValueError(msg)
    if np.any(fails <= 0) or np.any(shots <= 0):
        msg = "Both `fail` and `shots` must be strictly positive entries to calculate the logical error probability."
        raise ValueError(msg)
    # Wald approximation for binomial LEP; asymmetric intervals are tracked separately
    # (see issue #134).
    lep = fails / shots
    lep_stddev = np.sqrt(lep * (1 - lep) / shots)
    return lep, lep_stddev
