# (c) Copyright Riverlane 2020-2025.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

import matplotlib.pyplot as plt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting.results import (
    LambdaResult,
)
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)


def plot(
    result: LambdaResult | LEPPRResult,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Generic plot function that dispatches to specialised plotting based on the
    result type.

    This function inspects the type of ``result`` and calls the appropriate
    rendering logic:

    - :class:`~deltakit_explorer.plotting.results.LambdaResult` -- renders the
      error-suppression factor Λ fit curve with error bands.
    - :class:`~deltakit_explorer.plotting.results.LEPPRResult` -- renders the
      logical error probability per round fit curve with error bands.

    This enables users to compute the plot data separately (via
    :meth:`~deltakit_explorer.plotting.results.interpolate_lambda` or
    :meth:`~deltakit_explorer.plotting.results.interpolate_leppr`) and then
    render with a single call.

    Args:
        result: The precomputed plot data.
        fig: An existing matplotlib Figure. If None,
            a new figure will be created. Default is None.
        ax: An existing matplotlib Axes. If None, a new
            axes will be created. Default is None.
        title: An optional custom title for the plot. If None,
            a default title based on the result type will be used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        ValueError: If ``fig`` and ``ax`` are not both None or both set.
        TypeError: If the ``result`` type is not supported.

    Examples:
        Plotting a Lambda fit curve::

            from deltakit_explorer.plotting.results import interpolate_lambda

            lambda_result = interpolate_lambda(lambda_data, distances)
            fig, ax = plot(lambda_result)

        Plotting a LEPPR fit curve::

            from deltakit_explorer.plotting.results import interpolate_leppr

            leppr_result = interpolate_leppr(leppr_data, num_rounds)
            fig, ax = plot(leppr_result)
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both `None` or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    match result:
        case LambdaResult():
            x_vals = result.distances
            xlabel = "Code distance"
            ylabel = "Logical Error Probability per Round"
            default_title = "Error Suppression Factor Λ"
        case LEPPRResult():
            x_vals = result.rounds
            xlabel = "Rounds"
            ylabel = "Logical Error Probability per Round"
            default_title = "Logical Error Probability per Round"
        case _:
            msg = (
                f"Unsupported result type: {type(result).__name__}. "
                "Expected LambdaResult or LEPPRResult."
            )
            raise TypeError(msg)

    ax.plot(
        x_vals,
        result.interpolated,
        label=result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        x_vals,
        result.lower_boundary,
        result.upper_boundary,
        label=result.confidence_interval_label,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.set_title(title if title is not None else default_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig, ax
