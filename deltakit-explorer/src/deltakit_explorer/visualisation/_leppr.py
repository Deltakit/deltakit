import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy
import numpy.typing as npt
from deltakit_core.constants import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundResults


def plot_logical_error_probability_per_round(
    leppr_data: LogicalErrorProbabilityPerRoundResults,
    num_rounds: npt.NDArray[numpy.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[numpy.float64] | Sequence[float],
    logical_error_probability_stddev: npt.NDArray[numpy.float64] | Sequence[float] | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_data (LogicalErrorProbabilityPerRoundResults):
            Data class containing logical error probability per round fit results.
        num_rounds (npt.NDArray[numpy.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``.
        logical_error_probability (npt.NDArray[numpy.float64] | Sequence[float]):
            a sequence of floats representing the logical error probabilities
            corresponding to the number of rounds in ``num_rounds``.
        logical_error_probability_stddev (npt.NDArray[numpy.float64] | Sequence[float] | None, optional):
            a sequence of floats representing the standard deviation of the logical
            error probabilities corresponding to the number of rounds in ``num_rounds``.
            If None, no error bars will be plotted. Default is None.
        fig (Figure | None, optional):
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax (Axes | None, optional):
            a matplotlib Axes object to plot on. If None, a new axes will be created.
            Default is None.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example
        from deltakit_explorer.analysis import calculate_lep_and_lep_stddev, compute_logical_error_per_round
        num_failed_shots=[34, 151, 356]
        num_shots=[500000] * 3
        num_rounds=[2, 4, 6]

        res = compute_logical_error_per_round(
                    num_failed_shots=num_failed_shots,
                    num_shots=num_shots,
                    num_rounds=num_rounds,
                )

        # plot logical error probabilities
        lep, lep_stddev = calculate_lep_and_lep_stddev(fails=num_failed_shots, shots=num_shots)
        fig, ax = plot_logical_error_probability_per_round(
            res,
            num_rounds=num_rounds,
            logical_error_probability=lep,
            logical_error_probability_stddev=lep_stddev,
            )
        plt.show()
    """
    if (fig is None) ^ (ax is None):
        raise ValueError("The 'fig' and 'ax' parameters should either both be set or unset. Got only one set, which is not handled.")
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if not len(num_rounds) == len(logical_error_probability) == len(logical_error_probability_stddev):
        raise ValueError(
            "The lengths of 'num_rounds', 'logical_error_probability' and "
            "'logical_error_probability_stddev' must be the same. "
            f"Got lengths {len(num_rounds)}, {len(logical_error_probability)}, "
            f"and {len(logical_error_probability_stddev)} respectively."
        )

    isort = numpy.argsort(num_rounds)
    num_rounds = numpy.asarray(num_rounds)[isort]
    logical_error_probability = numpy.asarray(logical_error_probability)[isort]
    logical_error_probability_stddev = numpy.asarray(logical_error_probability_stddev)[isort]
    while num_rounds[0] <= 0:
        warnings.warn(
            f"Found an invalid number of rounds: {num_rounds[0]}. Number of rounds "
            "should be >= 1."
        )
        num_rounds = num_rounds[1:]
        logical_error_probability = logical_error_probability[1:]
        logical_error_probability_stddev = logical_error_probability_stddev[1:]

    if numpy.any(logical_error_probability <= 0) or numpy.any(logical_error_probability >= 1):
        raise RuntimeError(
            "Found an invalid logical error probability. Probabilities must be between 0 and 1"
            "Logical error probabilities: "
            f"{logical_error_probability}."
        )

    leppr, leppr_stddev = leppr_data.leppr, leppr_data.leppr_stddev
    spam, spam_stddev = leppr_data.spam_error, leppr_data.spam_error_stddev

    if leppr < 0 or spam < 0 or leppr >= 0.5 or spam >= 0.5:
        warnings.warn(
            "LEPPR or SPAM error is not within [0, 0.5)."
            f"LEPPR: {leppr}, SPAM error: {spam}."
        )
    if leppr_stddev < 0 or spam_stddev < 0:
        raise RuntimeError(
            "LEPPR or SPAM error standard deviation is negative. Standard deviations must be non-negative."
            f"LEPPR stddev: {leppr_stddev}, SPAM error stddev: {spam_stddev}."
        )

    # plot logical error probabilities
    ax.errorbar(num_rounds, logical_error_probability, yerr=logical_error_probability_stddev, fmt="o", color=RIVERLANE_PLOT_COLOURS[0], label = "Logical error probabilities")
    # plot fitted LEPPR curve
    interpolation_points = 200
    rounds_interpolated = numpy.linspace(
        num_rounds[0], num_rounds[-1], interpolation_points,
        dtype=numpy.float64,
    )
    y_interpolated = (1 - 2 * spam) * (1 - 2 * leppr) ** rounds_interpolated

    lep_interpolated = (1 - y_interpolated) /2
    ax.plot(rounds_interpolated, lep_interpolated,label=f"Fit, ε={leppr:.4f}" + r"$\pm$" + f"{leppr_stddev:.4f}", color=RIVERLANE_PLOT_COLOURS[0])

    # add error band to LEPPR curve
    lep_interpolated_err = numpy.array([numpy.sqrt(((2*spam_stddev)/(1-2*spam_stddev))**2 + ((2*r*leppr_stddev)/(1-2*leppr_stddev))**2) for r in rounds_interpolated])
    upper_error = lep_interpolated + lep_interpolated_err
    lower_error = lep_interpolated - lep_interpolated_err
    lower_error = numpy.clip(lower_error, 0, 1)  # ensure no negative values for log scale
    ax.fill_between(rounds_interpolated, lower_error, upper_error, color=RIVERLANE_PLOT_COLOURS[0], alpha=0.2)

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()

    return fig, ax
