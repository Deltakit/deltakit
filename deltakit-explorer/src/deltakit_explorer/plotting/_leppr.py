# (c) Copyright Riverlane 2020-2025.
"""Plot Logical Error Probability vs rounds with LEPPR fit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy.typing as npt

from deltakit_explorer.analysis._leppr import (
    LogicalErrorProbabilityPerRoundResults,
    compute_logical_error_per_round,
)
from deltakit_explorer.plotting._plot_style import (
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
    apply_publication_style,
)


def plot_leppr(
    num_rounds: npt.NDArray[numpy.int_] | Sequence[int],
    logical_error_probabilities: npt.NDArray[numpy.floating] | Sequence[float],
    logical_error_probabilities_stddev: npt.NDArray[numpy.floating] | Sequence[float],
    *,
    force_include_single_round: bool = False,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot Logical Error Probability vs number of rounds with fitted LEPPR curve.

    Calls :func:`compute_logical_error_per_round` to obtain the fit, then plots
    the data points with error bars and the fitted exponential curve. The y-axis
    uses a log scale. Styling is publication-ready with grid, legible fonts,
    and legend placement that avoids obscuring data.

    Args:
        num_rounds: Number of rounds for each data point.
        logical_error_probabilities: Logical error probability at each round count.
        logical_error_probabilities_stddev: Standard deviation of each LEP value.
        force_include_single_round: If True, include r=1 in the LEPPR fit.
        ax: Axes to draw on. If None, creates a new figure and axes.
        label: Optional label for the data series (used in legend).

    Returns:
        The Axes instance used for plotting (for further customization).

    Examples:
        Plot LEP vs rounds with fit::

            import numpy as np
            from deltakit_explorer.analysis import calculate_lep_and_lep_stddev
            from deltakit_explorer.plotting import plot_leppr

            num_rounds = [2, 4, 6, 8]
            fails, shots = [34, 151, 356, 512], [500000] * 4
            lep, lep_std = calculate_lep_and_lep_stddev(fails, shots)
            ax = plot_leppr(num_rounds, lep, lep_std)
            ax.figure.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    res: LogicalErrorProbabilityPerRoundResults = compute_logical_error_per_round(
        num_rounds,
        logical_error_probabilities,
        logical_error_probabilities_stddev,
        force_include_single_round=force_include_single_round,
    )

    num_rounds_arr = numpy.asarray(num_rounds)
    lep_arr = numpy.asarray(logical_error_probabilities)
    lep_std_arr = numpy.asarray(logical_error_probabilities_stddev)

    data_label = label if label is not None else "Data"
    ax.errorbar(
        num_rounds_arr,
        lep_arr,
        yerr=lep_std_arr,
        fmt="o",
        linestyle="-",
        capsize=3,
        label=data_label,
    )

    # Fitted curve: LEP(r) = (1 - (1-2*spam)*(1-2*leppr)^r) / 2
    r_min, r_max = int(num_rounds_arr.min()), int(num_rounds_arr.max())
    r_smooth = numpy.linspace(r_min, r_max, max(50, (r_max - r_min) * 5))
    fidelity = (1 - 2 * res.spam_error) * (1 - 2 * res.leppr) ** r_smooth
    lep_fit = (1 - fidelity) / 2

    fit_label = f"Fit, ε={res.leppr:.4f} ± {res.leppr_stddev:.4f}"
    ax.plot(r_smooth, lep_fit, linestyle="--", label=fit_label)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Rounds", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Logical Error Probability", fontsize=FONTSIZE_LABEL)
    ax.legend(loc="best", fontsize=FONTSIZE_LEGEND, framealpha=0.9)

    apply_publication_style(ax)
    return ax
