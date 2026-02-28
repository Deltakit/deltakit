from collections.abc import Sequence
from functools import singledispatch

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting.result import LambdaPlotResults, LepprPlotResult


@singledispatch
def interpolation_plot():
    return


@interpolation_plot.register(LepprPlotResult)
def _(
    plot_results: LepprPlotResult,
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_std: npt.NDArray[np.float64] | Sequence[float] | None = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        plot_results:
            Data class containing logical error probability per round fit results.
        distances:
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``.
        lep_per_round:
            a sequence of floats representing the logical error probabilities
            corresponding to the number of rounds in ``distances``.
        lep_per_round_std:
            a sequence of floats representing the standard deviation of the logical
            error probabilities corresponding to the number of rounds in ``distances``.
            If None, no error bars will be plotted. Default is None.
        num_sigmas: number of sigmas to consider when plotting error bars.
        fig:
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax:
            a matplotlib Axes object to plot on. If None, a new axes will be created.
            Default is None.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

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
        ...
        >>> lep, lep_stddev = calculate_lep_and_lep_stddev(
        ...     fails=num_failed_shots, shots=num_shots
        ... )
        >>> fig, ax = plot_lep_per_round_per_round(
        ...     res,
        ...     num_rounds=num_rounds,
        ...     lep_per_round=lep,
        ...     lep_per_round_std=lep_stddev,
        ... )
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    lens = {len(distances), len(lep_per_round)}
    if lep_per_round_std is not None:
        lens.add(len(lep_per_round_std))
    if len(lens) > 1:
        msg = (
            "The lengths of 'distances', 'lep_per_round' and "
            "'lep_per_round_std' must be the same. Got the following "
            f"lengths: {lens}."
        )
        raise ValueError(msg)

    isort = np.argsort(distances)
    distances = np.asarray(distances)[isort]
    lep_per_round = np.asarray(lep_per_round)[isort]
    if lep_per_round_std is not None:
        lep_per_round_std = num_sigmas * np.asarray(lep_per_round_std)[isort]

    # Plot the logical error probabilities
    ax.errorbar(
        distances,
        lep_per_round,
        yerr=lep_per_round_std,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=f"Logical error probabilities (±{num_sigmas}σ)",  # noqa: RUF001
    )

    distance_grid = np.linspace(distances[0], distances[-1], 200)
    plot_results.set_distances(distance_grid=distance_grid)
    lep_interpolated = plot_results._interpolate(
        plot_results.spam_error, plot_results.leppr, plot_results.distance_grid
    )

    # Plot the fitted logical error probability per round curve
    ax.plot(
        plot_results.distance_grid,
        lep_interpolated,
        label=f"Fit, ε={plot_results.leppr:.4f} ± {num_sigmas * plot_results.leppr_stddev:.4f} ({num_sigmas}σ)",  # noqa: RUF001
        color=RIVERLANE_PLOT_COLOURS[1],
    )

    lep_interpolated_low, lep_interpolated_high = plot_results._interpolate_error(
        num_sigmas
    )

    ax.fill_between(
        plot_results.distance_grid,
        np.clip(lep_interpolated_low, 0, 1),
        np.clip(lep_interpolated_high, 0, 1),
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()

    return fig, ax


@interpolation_plot.register(LambdaPlotResults)
def _(
    plot_results: LambdaPlotResults,
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_std: npt.NDArray[np.float64] | Sequence[float] | None = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot Λ-fitting data.

    This function plots both the logical error-probability per round that has been used
    to compute Λ, the associated error-rates if provided, and the resulting fit, showing
    how close the fit is from actual data.

    Args:
        plot_results (LambdaResults): Object that contains the results data
        distances (npt.NDArray[np.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_per_round_std (npt.NDArray[np.float64] | Sequence[float] | None):
            The standard deviation of the logical error probabilities per round.
        num_sigmas (int):
        fig (Figure):
        ax (Axes):

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:
        fig, ax = plot_lambda(
            result = LambdaResult,
            distances = [5, 7, 9],
            lep_per_round = [0.15, 0.1, 0.05],
            lep_stddev_per_round = [0.01, 0.008, 0.005],
        )
        ax.set_yscale("log")
        plt.show()
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    lengths = {len(distances), len(lep_per_round)}
    if lep_per_round_std is not None:
        lengths.add(len(lep_per_round_std))
    if len(lengths) > 1:
        msg = (
            "The lengths of 'distances', 'lep_per_round' and 'lep_per_round_std' "
            f"must be the same. Got the following lengths: {lengths}."
        )
        raise ValueError(msg)

    isort = np.argsort(distances)
    distances = np.asarray(distances)[isort]
    lep_per_round = np.asarray(lep_per_round)[isort]
    if lep_per_round_std is not None:
        lep_per_round_std = num_sigmas * np.asarray(lep_per_round_std)[isort]

    # Plot the logical error probabilities per round
    ax.errorbar(
        distances,
        lep_per_round,
        yerr=lep_per_round_std,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=f"Logical error probabilities per round (±{num_sigmas}σ)",  # noqa: RUF001
    )

    # Plot the fitted lambda curve
    distance_grid = np.linspace(distances[0], distances[-1], 200)
    plot_results.set_distances(distance_grid=distance_grid)
    lambda_interpolated = plot_results._interpolate(
        plot_results.lambda_, plot_results.lambda0, plot_results.distance_grid
    )

    ax.plot(
        plot_results.distance_grid,
        lambda_interpolated,
        label=f"Fit, Λ={plot_results.lambda_:.4f} ± {num_sigmas * plot_results.lambda_stddev:.4f} ({num_sigmas}σ)",  # noqa: RUF001
        color=RIVERLANE_PLOT_COLOURS[1],
    )

    lambda_interpolated_low, lambda_interpolated_high = plot_results._interpolate_error(
        num_sigmas=num_sigmas
    )

    ax.fill_between(
        plot_results.distance_grid,
        lambda_interpolated_low,
        lambda_interpolated_high,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Error suppression factor Λ")
    ax.legend()
    return fig, ax
