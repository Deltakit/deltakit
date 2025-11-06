from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy
import numpy.typing as npt
from deltakit_core.constants import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis._analysis import get_lambda_fit
from deltakit_explorer.analysis.lambda_ import calculate_lambda_and_lambda_stddev


# next step: plot lambda
def plot_lambda(
    distances: npt.NDArray[numpy.int_] | Sequence[int],
    lep_per_round: npt.NDArray[numpy.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[numpy.float64] | Sequence[float],
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Args:
        distances (npt.NDArray[numpy.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[numpy.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_stddev_per_round (npt.NDArray[numpy.float64] | Sequence[float]):
            The standard deviation of the logical error probabilities per round.

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
        raise ValueError("The 'fig' and 'ax' parameters should either both be set or unset. Got only one set, which is not handled.")
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    res = calculate_lambda_and_lambda_stddev(
        distances=distances,
        lep_per_round=lep_per_round,
        lep_stddev_per_round=lep_stddev_per_round,
    )
    lambda_val, lambda_val_stddev = res.lambda_, res.lambda_stddev
    ax.errorbar(distances, lep_per_round, yerr=lep_stddev_per_round, fmt="o", color=RIVERLANE_PLOT_COLOURS[0])

    y_vals = get_lambda_fit(distances, lep_per_round, lep_stddev_per_round)
    ax.plot(distances,y_vals,label=f"Fit, λ={lambda_val:.4f}" + r"$\pm$" + f"{lambda_val_stddev:.4f}", color=RIVERLANE_PLOT_COLOURS[0])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Logical error probability per round")
    ax.legend()
    return fig, ax
