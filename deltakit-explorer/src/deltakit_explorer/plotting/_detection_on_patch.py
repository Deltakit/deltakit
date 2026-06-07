from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting._draw import _draw_code

if TYPE_CHECKING:
    from deltakit_explorer.codes._planar_code import PlanarCode


def _aggregate_probabilities(
    detection_probabilities: dict[tuple[float, ...], list[float]],
    mode: Literal["average", "median", "variance"],
) -> dict[tuple[float, float], float]:
    """Aggregate per-round detection probabilities into a single value per coordinate.

    Args:
        detection_probabilities: Mapping from detector coordinates (x, y, ...) to
            per-round detection probability lists.
        mode: Aggregation mode:
            - ``"average"``: mean of all rounds.
            - ``"median"``: median of all rounds.
            - ``"variance"``: variance excluding first and last rounds.

    Returns:
        Mapping from (x, y) coordinate pairs to the aggregated probability value.
    """
    result: dict[tuple[float, float], float] = {}
    for coord, rates in detection_probabilities.items():
        values = np.asarray(rates)
        match mode:
            case "average":
                result[coord[:2]] = float(np.mean(values))
            case "median":
                result[coord[:2]] = float(np.median(values))
            case "variance":
                if len(values) > 2:
                    values = values[1:-1]
                result[coord[:2]] = float(np.var(values))
    return result


def _match_ancilla_coords(
    aggregated: dict[tuple[float, float], float],
    code: PlanarCode,
    tolerance: float = 1e-3,
) -> dict[tuple[float, float], float]:
    """Match aggregated probabilities to stabiliser ancilla coordinates.

    Args:
        aggregated: Mapping from (x, y) coordinate pairs to probability values.
        code: The planar code whose stabiliser ancilla coordinates are used
            for matching.
        tolerance: Absolute tolerance for matching floating-point coordinates.

    Returns:
        Mapping from ancilla (x, y) coordinate pairs to their matched probability.
    """
    ancilla_values: dict[tuple[float, float], float] = {}
    for stabilisers in code._stabilisers:
        for stabiliser in stabilisers:
            if stabiliser.ancilla_qubit is None:
                continue
            anc = stabiliser.ancilla_qubit.unique_identifier
            for prob_coord, prob_val in aggregated.items():
                if (
                    abs(prob_coord[0] - anc.x) < tolerance
                    and abs(prob_coord[1] - anc.y) < tolerance
                ):
                    ancilla_values[(float(anc.x), float(anc.y))] = prob_val
                    break

    return ancilla_values


def plot_detection_probability_on_patch(
    code: PlanarCode,
    detection_probabilities: dict[tuple[float, ...], list[float]],
    *,
    mode: Literal["average", "median", "variance"] = "average",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    show_colorbar: bool = True,
) -> tuple[Figure, Axes]:
    """Plot detection probabilities as coloured circles on a surface-code patch.

    Each stabiliser ancilla qubit is drawn as a circle whose colour reflects the
    detection probability at that location.

    Args:
        code: The planar code patch to draw.
        detection_probabilities: Per-round detection probabilities for each detector
            coordinate, as returned by ``Client.defect_rates()`` or
            ``detect_and_aggregate()``. Keys are ``(x, y, ...)`` tuples, values are
            lists of per-round probabilities.
        mode: How to aggregate the per-round values into a single number per detector.
        cmap: Matplotlib colour-map name for mapping probabilities to colours.
        vmin: Lower bound of the colour scale. If ``None``, inferred from data.
        vmax: Upper bound of the colour scale. If ``None``, inferred from data.
        fig: An existing matplotlib Figure. If ``None``, a new figure is created.
        ax: An existing matplotlib Axes. If ``None``, a new axes is created.
        show_colorbar: Whether to display a colour bar alongside the plot.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        ValueError: If ``fig`` and ``ax`` are not both ``None`` or both set.

    Examples:

        Standalone usage::

            from deltakit_explorer.codes import RotatedPlanarCode
            from deltakit_explorer.plotting import plot_detection_probability_on_patch

            code = RotatedPlanarCode(3, 3)
            rates = {(0.0, 2.0): [0.05, 0.08, 0.07, 0.09]}
            fig, ax = plot_detection_probability_on_patch(code, rates)

        Using as an inset plot::

            main_fig, main_ax = plt.subplots()
            inset_ax = main_ax.inset_axes([0.1, 0.1, 0.4, 0.4])
            fig, ax = plot_detection_probability_on_patch(
                code, rates, fig=main_fig, ax=inset_ax,
            )
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    aggregated = _aggregate_probabilities(detection_probabilities, mode)

    if not aggregated:
        msg = (
            "The detection_probabilities dict is empty. Nothing to plot."
        )
        raise ValueError(msg)

    ancilla_values = _match_ancilla_coords(aggregated, code)

    if fig is None and ax is None:
        fig, ax = _draw_code(code)
    else:
        ax.set_aspect(1)

    values = np.array(list(ancilla_values.values()))
    _vmin = vmin if vmin is not None else float(np.min(values))
    _vmax = vmax if vmax is not None else float(np.max(values))

    if _vmax - _vmin < 1e-12:
        _vmin = _vmin - 0.5
        _vmax = _vmax + 0.5

    norm = plt.Normalize(vmin=_vmin, vmax=_vmax)
    cmap_obj = plt.get_cmap(cmap)

    for (anc_x, anc_y), val in ancilla_values.items():
        color = cmap_obj(norm(val))
        circle = plt.Circle(
            (anc_x, anc_y),
            0.25,
            color=color,
            zorder=3,
        )
        ax.add_artist(circle)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Detection Probability")

    return fig, ax
