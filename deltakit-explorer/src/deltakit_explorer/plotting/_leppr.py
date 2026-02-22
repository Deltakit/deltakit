from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting.result import LogicalErrorProbabilityPerRoundResults


def plot_logical_error_probability_per_round(
    leppr_results: LogicalErrorProbabilityPerRoundResults,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: npt.NDArray[np.float64] | Sequence[float] | None = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_results (LogicalErrorProbabilityPerRoundResults):
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
        num_sigmas (int): number of sigmas to consider when plotting error bars.
        fig (Figure | None, optional):
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax (Axes | None, optional):
            a matplotlib Axes object to plot on. If None, a new axes will be created.
            Default is None.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example:

        >>> from deltakit_explorer.analysis import (
        ...     calculate_lep_and_lep_stddev, compute_logical_error_per_round,
        ... )
        >>> num_failed_shots=[34, 151, 356]
        >>> num_shots=[500000] * 3
        >>> num_rounds=[2, 4, 6]
        >>> res = compute_logical_error_per_round(
        ...     num_failed_shots=num_failed_shots,
        ...     num_shots=num_shots,
        ...     num_rounds=num_rounds,
        ... )
        >>> lep, lep_stddev = calculate_lep_and_lep_stddev(
        ...     fails=num_failed_shots, shots=num_shots
        ... )
        >>> fig, ax = plot_logical_error_probability_per_round(
        ...     res,
        ...     num_rounds=num_rounds,
        ...     logical_error_probability=lep,
        ...     logical_error_probability_stddev=lep_stddev,
        ... )
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    lens = {len(num_rounds), len(logical_error_probability)}
    if logical_error_probability_stddev is not None:
        lens.add(len(logical_error_probability_stddev))
    if len(lens) > 1:
        msg = (
            "The lengths of 'num_rounds', 'logical_error_probability' and "
            "'logical_error_probability_stddev' must be the same. Got the following "
            f"lengths: {lens}."
        )
        raise ValueError(msg)

    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    logical_error_probability = np.asarray(logical_error_probability)[isort]
    if logical_error_probability_stddev is not None:
        logical_error_probability_stddev = (
            num_sigmas * np.asarray(logical_error_probability_stddev)[isort]
        )


    # Plot the logical error probabilities
    ax.errorbar(
        num_rounds,
        logical_error_probability,
        yerr=logical_error_probability_stddev,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=f"Logical error probabilities (±{num_sigmas}σ)"  # noqa: RUF001
    )

    leppr_results.distance_grid = np.linspace(num_rounds[0], num_rounds[-1], 200)
    lep_interpolated = leppr_results._interpolate()

    # Plot the fitted logical error probability per round curve
    ax.plot(
        leppr_results.distance_grid,
        lep_interpolated,
        label=f"Fit, ε={leppr_results.leppr:.4f} ± {num_sigmas * leppr_results.leppr_stddev:.4f} ({num_sigmas}σ)",  # noqa: RUF001
        color=RIVERLANE_PLOT_COLOURS[1]
    )

    lep_interpolated_low, lep_interpolated_high  = leppr_results._interpolate_error(num_sigmas)

    ax.fill_between(
        leppr_results.distance_grid,
        np.clip(lep_interpolated_low, 0, 1),
        np.clip(lep_interpolated_high, 0, 1),
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2
    )

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()

    return fig, ax
