from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import floor

import numpy as np
import numpy.typing as npt
import scipy.optimize


@dataclass(frozen=True)
class LogicalErrorProbabilityPerRoundResults:
    """Named-tuple-like class containing computation results from
    :func:`compute_logical_error_per_round`.

    Attributes:
        leppr (float): Logical Error Probability Per Round (LEPPR).
        leppr_stddev (float): LEPPR standard deviation.
        spam_error (float): computed SPAM error probability.
        spam_error_stddev (float): SPAM error probability standard deviation.
        leppr_error_high (float): High (upper) asymmetric 1-sigma error on LEPPR.
        leppr_error_low (float): Low (lower) asymmetric 1-sigma error on LEPPR.
        spam_error_high (float): High (upper) asymmetric 1-sigma error on SPAM error.
        spam_error_low (float): Low (lower) asymmetric 1-sigma error on SPAM error.
    """

    leppr: float
    leppr_stddev: float
    spam_error: float
    spam_error_stddev: float
    leppr_error_high: float
    leppr_error_low: float
    spam_error_high: float
    spam_error_low: float


def compute_logical_error_per_round(
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probabilities: npt.NDArray[np.floating] | Sequence[float],
    logical_error_probabilities_low: npt.NDArray[np.floating] | Sequence[float],
    logical_error_probabilities_high: npt.NDArray[np.floating] | Sequence[float],
    *,
    force_include_single_round: bool = False,
    num_sigmas: int = 1,
) -> LogicalErrorProbabilityPerRoundResults:
    """Compute the logical error probability per round from different logical error
    probability computations.

    This function implements the method described in:

    1. https://arxiv.org/pdf/2310.05900.pdf (p.40)
    2. https://arxiv.org/pdf/2207.06431.pdf (p.21)
    3. https://arxiv.org/pdf/2505.09684.pdf (p.8)

    to recover an estimator of the logical error probability per round from the
    estimated values of logical error probabilities for several round durations.

    The logical error probabilities carry *asymmetric* lower and upper errors (for
    example the Wilson-score interval returned by
    :func:`calculate_lep_and_lep_stddev`). These are consumed directly: the weighted
    least-square fit uses Barlow's dynamic ("variable variance") weighting so that the
    effective variance of each point depends on which side of the fitted curve it falls
    on. See R. Barlow, "Asymmetric Statistical Errors" (arXiv:physics/0406120).

    Args:
        num_rounds (npt.NDArray[numpy.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``logical_error_probabilities``,
            ``logical_error_probabilities_low`` and
            ``logical_error_probabilities_high``. Any value below 1 (``< 1``) is
            automatically removed from this list along with the corresponding values in
            ``logical_error_probabilities``, ``logical_error_probabilities_low`` and
            ``logical_error_probabilities_high``. Any value equal to 1 is removed from
            this list along with the corresponding values in
            ``logical_error_probabilities``, ``logical_error_probabilities_low`` and
            ``logical_error_probabilities_high`` iff ``force_include_single_round`` is
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
        logical_error_probabilities_low (npt.NDArray[numpy.floating] | Sequence[float]):
            lower (low-side) asymmetric error on the logical error probabilities provided
            in ``logical_error_probabilities``, such that the lower bound of the interval
            is ``logical_error_probabilities - logical_error_probabilities_low``. Should
            be the same length as ``num_rounds``.
        logical_error_probabilities_high (npt.NDArray[numpy.floating] | Sequence[float]):
            upper (high-side) asymmetric error on the logical error probabilities provided
            in ``logical_error_probabilities``, such that the upper bound of the interval
            is ``logical_error_probabilities + logical_error_probabilities_high``. Should
            be the same length as ``num_rounds``.
        force_include_single_round (bool):
            if ``True``, data obtained from 1-round experiment will be used in the
            computation if provided in ``num_rounds``. Default to ``False`` which
            results in 1-round data being ignored due to boundary effects that affect
            the final estimation. See https://arxiv.org/pdf/2207.06431.pdf (p.21).
        num_sigmas (int): number of standard deviations defining the confidence level of
            the returned asymmetric errors (``leppr_error_high``/``leppr_error_low`` and
            ``spam_error_high``/``spam_error_low``). For example, ``num_sigmas=3``
            returns a 3σ asymmetric confidence interval. Defaults to ``1``. Note that the
            symmetric ``leppr_stddev``/``spam_error_stddev`` fields are always 1σ and are
            not affected by this parameter.

    Returns:
        LogicalErrorProbabilityPerRoundResults: detailed results of the computation.

    Examples:
        Calculating per-round logical error probability and its asymmetric error given
        the number of fails and the number of shots for several rounds::

            lep, lep_error_low, lep_error_high = calculate_lep_and_lep_stddev(
                fails=[34, 151, 356], shots=[500000] * 3
            )
            res = compute_logical_error_per_round(
                num_rounds=[2, 4, 6],
                logical_error_probabilities=lep,
                logical_error_probabilities_low=lep_error_low,
                logical_error_probabilities_high=lep_error_high,
            )
            leppr, leppr_stddev = res.leppr, res.leppr_stddev
            spam, spam_stddev = res.spam_error, res.spam_error_stddev
            leppr_high, leppr_low = res.leppr_error_high, res.leppr_error_low

    """
    # Get the inputs as numpy arrays.
    # Sanitisation: also make sure that the inputs are sorted.
    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    logical_error_probabilities = np.asarray(logical_error_probabilities)[isort]
    logical_error_probabilities_low = np.asarray(logical_error_probabilities_low)[isort]
    logical_error_probabilities_high = np.asarray(logical_error_probabilities_high)[isort]

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
        logical_error_probabilities_low = logical_error_probabilities_low[1:]
        logical_error_probabilities_high = logical_error_probabilities_high[1:]

    # Filter out the r == 1 input if not forced to include it by the user.
    if num_rounds.size > 0 and num_rounds[0] == 1 and not force_include_single_round:
        num_rounds = num_rounds[1:]
        logical_error_probabilities = logical_error_probabilities[1:]
        logical_error_probabilities_low = logical_error_probabilities_low[1:]
        logical_error_probabilities_high = logical_error_probabilities_high[1:]

    # Filter out logical error probabilities of 0.5 or above as that will lead to a
    # null-or-negative fidelity (1 - 2 * lep), whose logarithm is undefined.
    invalid_lep_indices = logical_error_probabilities >= 0.5
    if np.any(invalid_lep_indices):
        warnings.warn(
            "Found at least one invalid (i.e., >= 0.5) logical error probability. "
            "Ignoring all the provided logical error probabilities at or above 0.5."
        )
        valid_lep_indices = np.logical_not(invalid_lep_indices)
        num_rounds = num_rounds[valid_lep_indices]
        logical_error_probabilities = logical_error_probabilities[valid_lep_indices]
        logical_error_probabilities_low = logical_error_probabilities_low[
            valid_lep_indices
        ]
        logical_error_probabilities_high = logical_error_probabilities_high[
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
        lep_low = float(logical_error_probabilities_low[0])
        lep_high = float(logical_error_probabilities_high[0])
        # Implement Eq. (4) from section A.2.2. at page 40 of
        # https://arxiv.org/pdf/2310.05900.
        estimated_logical_error_per_round = (1 - (1 - 2 * lep) ** (1 / rounds)) / 2
        # The transform ``(1 - (1 - 2*lep) ** (1/r)) / 2`` is monotonically increasing in
        # ``lep``, so the low/high LEP errors map onto the low/high LEPPR errors directly
        # (no need to symmetrise). A symmetric 1σ standard deviation is reported using the
        # average of the two asymmetric errors as a representative width.
        leppr_error_low = (
            estimated_logical_error_per_round
            - (1 - (1 - 2 * (lep - lep_low)) ** (1 / rounds)) / 2
        )
        leppr_error_high = (
            (1 - (1 - 2 * (lep + lep_high)) ** (1 / rounds)) / 2
            - estimated_logical_error_per_round
        )
        estimated_logical_error_per_round_stddev = (
            leppr_error_low + leppr_error_high
        ) / 2
        return LogicalErrorProbabilityPerRoundResults(
            estimated_logical_error_per_round,
            estimated_logical_error_per_round_stddev,
            0,
            0,
            leppr_error_high,
            leppr_error_low,
            0.0,
            0.0,
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
    # We propagate the asymmetric LEP errors onto the log-fidelity through an error
    # propagation analysis. ``log(1 - 2*lep)`` is monotonically *decreasing* in ``lep``,
    # so a high-side LEP error maps onto a low-side log-fidelity error and vice-versa. The
    # ``2 / fidelity`` factor is the magnitude of the derivative ``d log(F) / d lep``.
    logfidelities_low = 2 * logical_error_probabilities_high / fidelities
    logfidelities_high = 2 * logical_error_probabilities_low / fidelities

    x = num_rounds.astype(float)

    def _residual(
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        slope, offset = beta
        delta = logfidelity - (slope * x + offset)
        # Barlow's dynamic ("variable variance") weighting for asymmetric errors. The
        # effective variance of each point depends on which side of the fitted curve the
        # observation falls on: it uses the high-side error when the data is above the fit
        # (``delta > 0``) and the low-side error otherwise. Following Barlow's
        # linear-variance model (arXiv:physics/0406120), with mean error
        # ``sigma = (sigma_high + sigma_low) / 2`` and asymmetry
        # ``sigma_prime = (sigma_high - sigma_low) / 2``, the variance is
        # ``V(delta) = sigma**2 + 2 * sigma_prime * delta``. This makes the weighting
        # depend on the current fit parameters and is therefore recomputed at every
        # iteration. We clip ``V`` to a small positive value to keep it well defined for
        # large negative deltas.
        sigma = (logfidelities_high + logfidelities_low) / 2
        sigma_prime = (logfidelities_high - logfidelities_low) / 2
        variance = np.clip(sigma**2 + 2 * sigma_prime * delta, 1e-300, None)
        return delta / np.sqrt(variance)

    result = scipy.optimize.least_squares(_residual, x0=np.array([0.0, 0.0]))

    slope = float(result.x[0])
    offset = float(result.x[1])

    # Recover the parameter covariance matrix from the Jacobian at the solution. The
    # residual is already scaled by 1/sqrt(variance), so the covariance is (JᵀJ)⁻¹. We
    # invert it through the SVD of ``J`` (rather than forming ``JᵀJ`` directly) to avoid
    # squaring the condition number, matching what ``scipy.optimize.curve_fit`` does.
    _, sv, vt = np.linalg.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * sv[0]
    sv = sv[sv > threshold]
    vt = vt[: sv.size]
    cov = (vt.T / sv**2) @ vt
    slope_stddev, offset_stddev = (float(v) for v in np.sqrt(np.diagonal(cov)))

    estimated_logical_error_per_round = float((1 - np.exp(slope)) / 2)
    estimated_spam_error = float((1 - np.exp(offset)) / 2)

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
    # The standard deviation on the linear fit parameters can be obtained through the
    # covariance matrix diagonal entries.
    estimated_logical_error_per_round_stddev = float(
        (1 - 2 * estimated_logical_error_per_round) * slope_stddev / 2
    )
    estimated_spam_error_stddev = float(
        (1 - 2 * estimated_spam_error) * offset_stddev / 2
    )

    # Asymmetric num_sigmas CI obtained by propagating the (symmetric) num_sigmas-scaled
    # slope/offset standard deviations through the (1 - exp(.)) / 2 transform. The interval
    # is symmetric in slope/offset space but becomes asymmetric after the transform.
    # ``slope`` is negative; a lower (more negative) slope → higher leppr (high side).
    slope_delta = num_sigmas * slope_stddev
    offset_delta = num_sigmas * offset_stddev
    leppr_error_high = max(
        0.0, float((np.exp(slope) - np.exp(slope - slope_delta)) / 2)
    )
    leppr_error_low = max(
        0.0, float((np.exp(slope + slope_delta) - np.exp(slope)) / 2)
    )
    spam_error_high = max(
        0.0, float((np.exp(offset) - np.exp(offset - offset_delta)) / 2)
    )
    spam_error_low = max(
        0.0, float((np.exp(offset + offset_delta) - np.exp(offset)) / 2)
    )
    return LogicalErrorProbabilityPerRoundResults(
        estimated_logical_error_per_round,
        estimated_logical_error_per_round_stddev,
        estimated_spam_error,
        estimated_spam_error_stddev,
        leppr_error_high,
        leppr_error_low,
        spam_error_high,
        spam_error_low,
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
