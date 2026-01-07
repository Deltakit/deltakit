# (c) Copyright Riverlane 2020-2025.
"""`analysis` module aggregates client-side analytical tools."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from deltakit_core.deprecation import deprecated
from packaging.version import Version


@deprecated(
    reason="A better function is now available.",
    replaced_by="deltakit.explorer.analysis.compute_logical_error_per_round",
    removed_in_version=Version("1.0.0"),
)
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
    Implement logical error probability per round fit as described in
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
            - an epsilon parameter value (error probability per round);
            - an interpolated graph (2 value), computed with this value;
            - error bars value.

    Examples:
        Calling the method::

            eps, x, y, yerr = Analysis.get_exp_fit(
                logical_fails_all_rounds=[20, 20, 30, 40, 50],
                shots_all_rounds=[500, 500, 500, 500, 500],
                all_rounds=[1, 3, 5, 7, 9],
                interpolation_points=25 + 1,
            )

    """
    # Calculate logical fidelity (F = 1 - 2 * p_err) and its standard error
    logical_perr_per_round = np.array(logical_fails_all_rounds) / np.array(
        shots_all_rounds
    )
    fidelity = 1 - 2 * logical_perr_per_round
    assert not np.any(fidelity < 0.0), (
        f"Fidelity values (1-2*p) should be non-negative, but were {fidelity}."
    )

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
        all_rounds[0],
        all_rounds[-1],
        interpolation_points,
        dtype=np.float64,
    )
    y_interpolated = [f_0 * (1 - 2 * epsilon) ** r for r in rounds_interpolated]
    probs_interpolated = (1.0 - np.array(y_interpolated, dtype=np.float64)) * 0.5

    return epsilon, rounds_interpolated, probs_interpolated, yerr


def calculate_lep_and_lep_stddev(
    fails: npt.NDArray[np.int_] | Sequence[int] | int,
    shots: npt.NDArray[np.int_] | Sequence[int] | int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the logical error probability and its standard deviation.

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
    lep = fails / shots
    if np.any(lep <= 0):
        msg = "Must have > 0 fails to calculate logical error probability."
        raise ValueError(msg)
    lep_stddev = np.sqrt(lep * (1 - lep) / shots)

    return lep, lep_stddev



@deprecated(
    reason=(
        "A better function is now available. The output of this function can now be "
        "computed with error bars and any number of points."
    ),
    replaced_by="deltakit.explorer.analysis.calculate_lambda_and_lambda_stddev",
    removed_in_version=Version("1.0.0"),
)
def get_lambda_fit(
    distances: list[int], lep_per_round: list[float], lep_stddev_per_round: list[float]
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
                lep_stddev_per_round=[1.2e-05, 9.3e-06, 3.9e-06],
            )

    """
    params, _ = np.polyfit(
        x=distances,
        y=np.log(lep_per_round),
        deg=1,
        w=[
            lep_pr / lep_std_pr
            for lep_pr, lep_std_pr in zip(lep_per_round, lep_stddev_per_round)
        ],
        cov="unscaled",
    )

    return np.exp(params[0] * np.array(distances) + params[1])
