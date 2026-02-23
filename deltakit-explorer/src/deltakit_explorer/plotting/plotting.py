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
def _(leppr_results: LepprPlotResult,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: (
        npt.NDArray[np.float64] | Sequence[float] | None
    ) = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_results:
            Data class containing logical error probability per round fit results.
        num_rounds:
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``.
        logical_error_probability:
            a sequence of floats representing the logical error probabilities
            corresponding to the number of rounds in ``num_rounds``.
        logical_error_probability_stddev:
            a sequence of floats representing the standard deviation of the logical
            error probabilities corresponding to the number of rounds in ``num_rounds``.
            If None, no error bars will be plotted. Default is None.
        num_sigmas (int): number of sigmas to consider when plotting error bars.
        fig:
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax:
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
        label=f"Logical error probabilities (±{num_sigmas}σ)",  # noqa: RUF001
    )

    distance_grid = np.linspace(num_rounds[0], num_rounds[-1], 200)
    leppr_results.set_distances(distance_grid=distance_grid)
    lep_interpolated = leppr_results._interpolate(
        leppr_results.leppr, leppr_results.spam_error, leppr_results.distance_grids
    )

    # Plot the fitted logical error probability per round curve
    ax.plot(
        leppr_results.distance_grid,
        lep_interpolated,
        label=f"Fit, ε={leppr_results.leppr:.4f} ± {num_sigmas * leppr_results.leppr_stddev:.4f} ({num_sigmas}σ)",  # noqa: RUF001
        color=RIVERLANE_PLOT_COLOURS[1],
    )

    lep_interpolated_low, lep_interpolated_high = leppr_results._interpolate_error(
        num_sigmas
    )

    ax.fill_between(
        leppr_results.distance_grid,
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
    lambda_results: LambdaPlotResults,
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_stddev: npt.NDArray[np.float64] | Sequence[float] | None = None,
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
        lambda_results (LambdaResults): Object that contains the results data
        distances (npt.NDArray[np.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_per_round_stddev (npt.NDArray[np.float64] | Sequence[float] | None):
            The standard deviation of the logical error probabilities per round.
        num_sigmas (int):
        fig (Figure):
        ax (Axes):

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example:
        fig, ax = plot_lambda(
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

    # These should be already checked by the above code, but type checkers are not able
    # to infer that information, so including the asserts explicitly for type checkers
    # to understand.
    assert ax is not None
    assert fig is not None

    lengths = {len(distances), len(lep_per_round)}
    if lep_per_round_stddev is not None:
        lengths.add(len(lep_per_round_stddev))
    if len(lengths) > 1:
        msg = (
            "The lengths of 'distances', 'lep_per_round' and 'lep_per_round_stddev' "
            f"must be the same. Got the following lengths: {lengths}."
        )
        raise ValueError(msg)

    isort = np.argsort(distances)
    distances = np.asarray(distances)[isort]
    lep_per_round = np.asarray(lep_per_round)[isort]
    if lep_per_round_stddev is not None:
        lep_per_round_stddev = num_sigmas * np.asarray(lep_per_round_stddev)[isort]

    # Plot the logical error probabilities per round
    ax.errorbar(
        distances,
        lep_per_round,
        yerr=lep_per_round_stddev,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[1],
        label=f"Logical error probabilities per round (±{num_sigmas}σ)",  # noqa: RUF001
    )

    # Plot the fitted lambda curve
    distance_grid = np.linspace(distances[0], distances[-1], 200)
    lambda_results.set_distances(distance_grid=distance_grid)
    lambda_interpolated = lambda_results._interpolate(
        lambda_results.lambda_, lambda_results.lambda_0, lambda_results.distance_grid
    )

    ax.plot(
        lambda_results.distance_grid,
        lambda_interpolated,
        label=f"Fit, Λ={lambda_results.lambda_:.4f} ± {num_sigmas * lambda_results.lambda_stddev:.4f} ({num_sigmas}σ)",  # noqa: RUF001
        color=RIVERLANE_PLOT_COLOURS[1],
    )

    lambda_interpolated_low, lambda_interpolated_high = (
        lambda_results._interpolate_error(num_sigmas=num_sigmas)
    )

    ax.fill_between(
        lambda_results.distance_grid,
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
