# Copyright Riverlane 2020-2025.
"""Plot logical error probability per round vs code distance with optional Λ fit.

This module provides a styling-focused wrapper around matplotlib for
visualising logical error probabilities per round as a function of code
distance. It intentionally does **not** compute Λ or Λ₀ itself; callers are
expected to pass any pre-computed values explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy
import numpy.typing as npt
from matplotlib.axes import Axes

from deltakit_explorer.plotting._plot_style import (
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
    apply_publication_style,
)


def plot_lambda(
    distances: npt.NDArray[numpy.int_] | Sequence[int],
    lep_per_round: npt.NDArray[numpy.floating] | Sequence[float],
    lep_stddev_per_round: (
        npt.NDArray[numpy.floating] | Sequence[float] | None
    ) = None,
    *,
    lambda_value: float | None = None,
    lambda_stddev: float | None = None,
    lambda0: float | None = None,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot LEP per round vs code distance with optional Λ fit curve.

    The function expects **pre-computed** logical error probabilities per
    round and, optionally, pre-computed values of the error suppression
    factor Λ and multiplicative offset Λ₀. It does **not** perform any
    fitting itself, which allows callers to use any workflow for computing
    Λ and simply rely on this helper for plotting.

    When both ``lambda_value`` and ``lambda0`` are provided, a fitted curve
    of the form

    .. math::

        \\epsilon_d = \\frac{1}{\\Lambda_0 \\Lambda^{(d+1)/2}}

    is drawn in addition to the data points.

    Args:
        distances: Code distances for each data point (odd integers).
        lep_per_round: Logical error probability per round at each distance.
        lep_stddev_per_round: Standard deviation of LEP per round at each
            distance. If ``None``, no error bars are drawn.
        lambda_value: Pre-computed error suppression factor Λ. This can be
            obtained, for example, with
            :func:`deltakit_explorer.analysis.calculate_lambda_and_lambda_stddev`,
            or with any other analysis code.
        lambda_stddev: Standard deviation of ``lambda_value``. Used only for
            labelling the fit in the legend.
        lambda0: Pre-computed multiplicative offset Λ₀ used in the fit
            formula above. If ``None``, no fitted curve is drawn.
        ax: Axes to draw on. If ``None``, a new figure and axes are created.
        label: Optional label for the data series (used in the legend).

    Returns:
        The Axes instance used for plotting (for further customization).

    Examples:
        Plot LEP per round vs distance with externally-computed Λ::

            from deltakit_explorer.plotting import plot_lambda
            from deltakit_explorer.analysis import calculate_lambda_and_lambda_stddev

            distances = [5, 7, 9]
            lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
            lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
            res = calculate_lambda_and_lambda_stddev(
                distances, lep_per_round, lep_stddev
            )
            ax = plot_lambda(
                distances,
                lep_per_round,
                lep_stddev,
                lambda_value=res.lambda_,
                lambda_stddev=res.lambda_stddev,
                lambda0=res.lambda0,
            )
            ax.figure.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    distances_arr = numpy.asarray(distances)
    lep_arr = numpy.asarray(lep_per_round)
    lep_std_arr = (
        None
        if lep_stddev_per_round is None
        else numpy.asarray(lep_stddev_per_round)
    )

    data_label = label if label is not None else "Data"
    if lep_std_arr is not None:
        ax.errorbar(
            distances_arr,
            lep_arr,
            yerr=lep_std_arr,
            fmt="o",
            linestyle="-",
            capsize=3,
            label=data_label,
        )
    else:
        ax.plot(
            distances_arr,
            lep_arr,
            marker="o",
            linestyle="-",
            label=data_label,
        )

    # Optional fitted curve if Λ and Λ₀ are provided.
    if lambda_value is not None and lambda0 is not None:
        d_min, d_max = int(distances_arr.min()), int(distances_arr.max())
        d_smooth = numpy.linspace(d_min, d_max, max(50, (d_max - d_min) * 5))
        lep_fit = 1 / (lambda0 * lambda_value ** ((d_smooth + 1) / 2))

        if lambda_stddev is not None:
            fit_label = f"Λ = {lambda_value:.3f} ± {lambda_stddev:.3f}"
        else:
            fit_label = f"Λ = {lambda_value:.3f}"

        ax.plot(d_smooth, lep_fit, linestyle="--", label=fit_label)

    ax.set_yscale("log")
    ax.set_xlabel("Code Distance $d$", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Logical Error Probability per Round", fontsize=FONTSIZE_LABEL)
    ax.legend(loc="best", fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.set_xticks(distances_arr)

    apply_publication_style(ax)
    return ax

