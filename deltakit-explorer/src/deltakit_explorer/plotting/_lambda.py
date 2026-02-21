from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis._lambda import LambdaResults
from deltakit_explorer.plotting.results import LambdaPlot, compute_lambda_plot


def plot_lambda(
    lambda_data: LambdaResults,
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
        lambda_data (LambdaResults): Results from
            :func:`~deltakit_explorer.analysis.calculate_lambda_and_lambda_stddev`.
        distances (npt.NDArray[numpy.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[numpy.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_per_round_stddev (npt.NDArray[numpy.float64] | Sequence[float] | None):
            The standard deviation of the logical error probabilities per round.
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
        label=f"Logical error probabilities per round (±{num_sigmas}σ)"  # noqa: RUF001
    )

    # Compute the interpolated fit data using the result type
    lambda_plot: LambdaPlot = compute_lambda_plot(
        lambda_data, distances, num_sigmas=num_sigmas
    )

    # Plot the fitted lambda curve
    lambda_, lambda_stddev = lambda_data.lambda_, lambda_data.lambda_stddev
    ax.plot(
        lambda_plot.distances,
        lambda_plot.interpolated,
        label=f"Fit, Λ={lambda_:.4f} ± {num_sigmas * lambda_stddev:.4f} ({num_sigmas}σ)",  # noqa: RUF001
        color=RIVERLANE_PLOT_COLOURS[1]
    )

    # Add error band to lambda curve
    ax.fill_between(
        lambda_plot.distances,
        lambda_plot.lower_boundary,
        lambda_plot.upper_boundary,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2
    )

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Error suppression factor Λ")
    ax.legend()
    return fig, ax
