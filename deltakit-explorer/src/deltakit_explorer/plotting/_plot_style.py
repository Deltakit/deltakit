# (c) Copyright Riverlane 2020-2025.
"""Shared styling constants and helpers for publication-ready plotting."""

from __future__ import annotations

# B&W-friendly marker and linestyle cycles for multiple datasets
MARKERS = ("o", "s", "^", "D", "v", "<", ">", "p")
LINESTYLES = ("-", "--", "-.", ":")

# Publication-ready font sizes
FONTSIZE_LABEL = 12
FONTSIZE_TITLE = 14
FONTSIZE_TICK = 11
FONTSIZE_LEGEND = 11

# Grid style
GRID_KWARGS = {"which": "both", "linestyle": "--", "alpha": 0.5}


def apply_publication_style(ax: object) -> None:
    """Apply grid and font sizes to an Axes for publication-ready appearance.

    Args:
        ax: matplotlib Axes instance to style.
    """
    ax.grid(True, **GRID_KWARGS)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)


def get_cycle_item(cycle: tuple[str, ...], index: int) -> str:
    """Return the cycle element at the given index (wraps around).

    Args:
        cycle: Tuple of style strings (e.g. markers or linestyles).
        index: Zero-based index; may be larger than len(cycle).

    Returns:
        The element at index % len(cycle).
    """
    return cycle[index % len(cycle)]
