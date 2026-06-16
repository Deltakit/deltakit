# (c) Copyright Riverlane 2020-2025.
"""Plotting helpers for error-suppression factor results."""

from __future__ import annotations

from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting._utils import get_figure_and_axes
from deltakit_explorer.plotting.results import LambdaResult


def plot_lambda(
    lambda_result: LambdaResult,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot an interpolated Lambda result.

    This specialised plotter owns only the Lambda-specific rendering logic. It
    expects a ready-to-plot :class:`~deltakit_explorer.plotting.results.LambdaResult`.
    Higher-level data preparation, such as interpolation from raw Lambda data,
    should be handled by :func:`deltakit_explorer.plotting.plot` before dispatch.

    Args:
        lambda_result: Interpolated Lambda result to plot.
        fig: A matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: A matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.
        title: An optional custom title for the plot. If None, the default
            Lambda title will be used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Examples:

        Plotting an interpolated Lambda result::

            from deltakit_explorer.plotting import interpolate_lambda, plot_lambda

            lambda_result = interpolate_lambda(lambda_data)
            fig, ax = plot_lambda(lambda_result)

    """
    fig, ax = get_figure_and_axes(fig, ax)
    ax.plot(
        lambda_result.distances,
        lambda_result.interpolated,
        label=lambda_result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        lambda_result.distances,
        lambda_result.lower_boundary,
        lambda_result.upper_boundary,
        label=lambda_result.confidence_interval_label,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.set_title(title if title is not None else "Error Suppression Factor Λ")
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Logical Error Probability per Round")
    ax.legend()
    return fig, ax
