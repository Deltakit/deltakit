# (c) Copyright Riverlane 2020-2025.
"""Plot Logical Error Probability per round vs code distance with Λ fit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy
import numpy.typing as npt
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from deltakit_explorer.analysis.lambda_ import (
    LambdaResults,
    calculate_lambda_and_lambda_stddev,
)
from deltakit_explorer.plotting._plot_style import (
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
    apply_publication_style,
)


def plot_lambda(
    distances: npt.NDArray[numpy.int_] | Sequence[int],
    lep_per_round: npt.NDArray[numpy.floating] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[numpy.floating] | Sequence[float],
    *,
    method: Literal["d", "(d+1)/2", "direct"] = "(d+1)/2",
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot LEP per round vs code distance with Λ fit curve.

    Calls :func:`calculate_lambda_and_lambda_stddev` to obtain Λ and Λ_0, then
    plots the data points with error bars and the fitted curve
    ε_d = 1 / (Λ_0 * Λ^((d+1)/2)). The y-axis uses a log scale. Styling is
    publication-ready with grid, legible fonts, and legend placement that avoids
    obscuring data.

    Args:
        distances: Code distances for each data point (odd integers).
        lep_per_round: Logical error probability per round at each distance.
        lep_stddev_per_round: Standard deviation of LEP per round at each distance.
        method: Fitting method: "d", "(d+1)/2", or "direct". Default "(d+1)/2".
        ax: Axes to draw on. If None, creates a new figure and axes.
        label: Optional label for the data series (used in legend).

    Returns:
        The Axes instance used for plotting (for further customization).

    Examples:
        Plot LEP per round vs distance with Λ fit::

            from deltakit_explorer.plotting import plot_lambda

            distances = [5, 7, 9]
            lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
            lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
            ax = plot_lambda(distances, lep_per_round, lep_stddev)
            ax.figure.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    res: LambdaResults = calculate_lambda_and_lambda_stddev(
        distances,
        lep_per_round,
        lep_stddev_per_round,
        method=method,
    )

    distances_arr = numpy.asarray(distances)
    lep_arr = numpy.asarray(lep_per_round)
    lep_std_arr = numpy.asarray(lep_stddev_per_round)

    data_label = label if label is not None else "Data"
    ax.errorbar(
        distances_arr,
        lep_arr,
        yerr=lep_std_arr,
        fmt="o",
        linestyle="-",
        capsize=3,
        label=data_label,
    )

    # Fitted curve: ε_d = 1 / (Λ_0 * Λ^((d+1)/2))
    d_min, d_max = int(distances_arr.min()), int(distances_arr.max())
    d_smooth = numpy.linspace(d_min, d_max, max(50, (d_max - d_min) * 5))
    lep_fit = 1 / (res.lambda0 * res.lambda_ ** ((d_smooth + 1) / 2))

    fit_label = f"Λ = {res.lambda_:.3f} ± {res.lambda_stddev:.3f}"
    ax.plot(d_smooth, lep_fit, linestyle="--", label=fit_label)

    ax.set_yscale("log")
    ax.set_xlabel("Code Distance $d$", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Logical Error Probability per Round", fontsize=FONTSIZE_LABEL)
    ax.legend(loc="best", fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.set_xticks(distances_arr)

    apply_publication_style(ax)
    return ax
