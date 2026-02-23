# (c) Copyright Riverlane 2020-2025.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

import matplotlib.pyplot as plt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting.results import LEPPRPlot, LambdaPlot


def plot(
    result: LambdaPlot | LEPPRPlot,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Generic plot function that dispatches to specialised plotting based on the
    result type.

    This function inspects the type of ``result`` and calls the appropriate
    rendering logic:

    - :class:`~deltakit_explorer.plotting.results.LambdaPlot` → renders the
      error-suppression factor Λ fit curve with error bands.
    - :class:`~deltakit_explorer.plotting.results.LEPPRPlot` → renders the
      logical error probability per round fit curve with error bands.

    This enables users to compute the plot data separately (via
    :func:`~deltakit_explorer.plotting.results.compute_lambda_plot` or
    :func:`~deltakit_explorer.plotting.results.compute_leppr_plot`) and then
    render with a single call.

    Args:
        result (LambdaPlot | LEPPRPlot): The precomputed plot data.
        fig (Figure | None, optional): An existing matplotlib Figure. If None,
            a new figure will be created. Default is None.
        ax (Axes | None, optional): An existing matplotlib Axes. If None, a new
            axes will be created. Default is None.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Raises:
        ValueError: If ``fig`` and ``ax`` are not both None or both set, or if
            the ``result`` type is not supported.

    Examples:
        Plotting a Lambda fit curve::

            from deltakit_explorer.plotting.results import compute_lambda_plot
            lambda_plot = compute_lambda_plot(lambda_data, distances)
            fig, ax = plot(lambda_plot)

        Plotting a LEPPR fit curve::

            from deltakit_explorer.plotting.results import compute_leppr_plot
            leppr_plot = compute_leppr_plot(leppr_data, num_rounds)
            fig, ax = plot(leppr_plot)
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    if isinstance(result, LambdaPlot):
        _plot_lambda(result, fig, ax)
    elif isinstance(result, LEPPRPlot):
        _plot_leppr(result, fig, ax)
    else:
        msg = (
            f"Unsupported result type: {type(result).__name__}. "
            "Expected LambdaPlot or LEPPRPlot."
        )
        raise ValueError(msg)

    return fig, ax


def _plot_lambda(result: LambdaPlot, fig: Figure, ax: Axes) -> None:
    """Render a :class:`LambdaPlot` on the given axes."""
    ax.plot(
        result.distances,
        result.interpolated,
        label="Λ fit",
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


def _plot_leppr(result: LEPPRPlot, fig: Figure, ax: Axes) -> None:
    """Render a :class:`LEPPRPlot` on the given axes."""
    ax.plot(
        result.rounds,
        result.interpolated,
        label="LEPPR fit",
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
