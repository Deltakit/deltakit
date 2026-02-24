# (c) Copyright Riverlane 2020-2025.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

import matplotlib.pyplot as plt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting.results import (
    LambdaResult,
    LEPPRResult,
)


def plot(
    result: LambdaResult | LEPPRResult,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Generic plot function that dispatches to specialised plotting based on the
    result type.

    This function inspects the type of ``result`` and calls the appropriate
    rendering logic:

    - :class:`~deltakit_explorer.plotting.results.LambdaResult` → renders the
      error-suppression factor Λ fit curve with error bands.
    - :class:`~deltakit_explorer.plotting.results.LEPPRResult` → renders the
      logical error probability per round fit curve with error bands.

    This enables users to compute the plot data separately (via
    :meth:`~deltakit_explorer.plotting.results._lambda_interpolate` or
    :meth:`~deltakit_explorer.plotting.results._leppr_interpolate`) and then
    render with a single call.

    Args:
        result: The precomputed plot data.
        fig: An existing matplotlib Figure. If None,
            a new figure will be created. Default is None.
        ax: An existing matplotlib Axes. If None, a new
            axes will be created. Default is None.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        ValueError: If ``fig`` and ``ax`` are not both None or both set, or if
            the ``result`` type is not supported.

    Examples:
        Plotting a Lambda fit curve::

            from deltakit_explorer.plotting.results import _lambda_interpolate
            lambda_result = _lambda_interpolate(lambda_data, distances)
            fig, ax = plot(lambda_result)

        Plotting a LEPPR fit curve::

            from deltakit_explorer.plotting.results import _leppr_interpolate
            leppr_result = _leppr_interpolate(leppr_data, num_rounds)
            fig, ax = plot(leppr_result)
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    if isinstance(result, LambdaResult):
        _plot_lambda(result, ax)
    elif isinstance(result, LEPPRResult):
        _plot_leppr(result, ax)
    else:
        msg = (
            f"Unsupported result type: {type(result).__name__}. "
            "Expected LambdaResult or LEPPRResult."
        )
        raise ValueError(msg)

    return fig, ax


def _plot_lambda(result: LambdaResult, ax: Axes) -> None:
    """Render a :class:`LambdaPlot` on the given axes."""
    ax.plot(
        result.distances,
        result.interpolated,
        label=result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        result.distances,
        result.lower_boundary,
        result.upper_boundary,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Error suppression factor Λ")
    ax.legend()


def _plot_leppr(result: LEPPRResult, ax: Axes) -> None:
    """Render a :class:`LogicalErrorProbabilityPerRoundPlot` on the given axes."""
    ax.plot(
        result.rounds,
        result.interpolated,
        label=result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        result.rounds,
        result.lower_boundary,
        result.upper_boundary,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()
