# (c) Copyright Riverlane 2020-2025.
"""Shared helpers for plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def get_figure_and_axes(
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Validate or create a matplotlib figure and axes pair.

    Args:
        fig: An existing matplotlib Figure. If None, a new figure will be
            created when ``ax`` is also None.
        ax: An existing matplotlib Axes. If None, a new axes will be created
            when ``fig`` is also None.

    Returns:
        A valid matplotlib Figure and Axes pair.

    Raises:
        ValueError: If exactly one of ``fig`` and ``ax`` is provided.
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both `None` or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert fig is not None
    assert ax is not None
    return fig, ax
