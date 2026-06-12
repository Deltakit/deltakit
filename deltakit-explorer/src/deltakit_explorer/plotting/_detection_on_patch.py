from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

if TYPE_CHECKING:
    from deltakit_explorer.codes._planar_code import PlanarCode
    from deltakit_explorer.codes._stabiliser import Stabiliser


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

    Raises:
        ValueError: If ``mode`` is not one of ``"average"``, ``"median"``, or
            ``"variance"``.
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
            case _:
                msg = f"Unknown mode: {mode!r}. Expected one of 'average', 'median', 'variance'."
                raise ValueError(msg)
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


def _stabiliser_vertices(stabiliser: Stabiliser) -> list[tuple[float, float]]:
    """Return ordered polygon vertices for a stabiliser's plaquette.

    For weight-4 stabilisers, returns the four data-qubit corners forming a
    diamond. For weight-2 boundary stabilisers, returns a triangle formed by
    the two data qubits and the ancilla qubit.

    Args:
        stabiliser: The stabiliser whose plaquette vertices to compute.

    Returns:
        Vertices ordered counter-clockwise around the plaquette centre.

    Raises:
        ValueError: If the stabiliser has fewer than two data qubits or is
            missing its ancilla qubit.
    """
    coords: list[tuple[float, float]] = [
        (
            float(pauli.qubit.unique_identifier.x),
            float(pauli.qubit.unique_identifier.y),
        )
        for pauli in stabiliser.paulis
        if pauli is not None
    ]
    if len(coords) == 2:
        if stabiliser.ancilla_qubit is None:
            msg = "Boundary stabilisers need an ancilla qubit to form a plaquette."
            raise ValueError(msg)
        anc = stabiliser.ancilla_qubit.unique_identifier
        coords.append((float(anc.x), float(anc.y)))
    if len(coords) < 3:
        msg = "A stabiliser plaquette needs at least two data qubits."
        raise ValueError(msg)

    center = np.mean(coords, axis=0)
    return sorted(
        coords,
        key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]),
    )


def _heatmap_patch_position(
    stabiliser: Stabiliser,
    code: PlanarCode,
) -> tuple[str, float, float]:
    """Map a stabiliser to a square-grid heatmap position.

    Args:
        stabiliser: The stabiliser whose grid position to compute.
        code: The planar code patch.

    Returns:
        A tuple ``(patch_type, x, y)`` where ``patch_type`` is ``"square"``
        for interior weight-4 stabilisers or ``"circle"`` for boundary
        weight-2 stabilisers, and ``(x, y)`` is the heatmap cell coordinate.

    Raises:
        ValueError: If the stabiliser has no ancilla qubit or an unsupported
            weight.
    """
    if stabiliser.ancilla_qubit is None:
        msg = "Cannot draw a detector heatmap for stabilisers without ancilla qubits."
        raise ValueError(msg)

    anc = stabiliser.ancilla_qubit.unique_identifier
    weight = len([pauli for pauli in stabiliser.paulis if pauli is not None])

    if weight == 4:
        return "square", float(anc.x / 2 - 1), float(anc.y / 2 - 1)

    if weight == 2:
        if anc.x == 0:
            return "circle", 0.0, float(anc.y / 2 - 0.5)
        if anc.x == 2 * code.width:
            return "circle", float(code.width - 1), float(anc.y / 2 - 0.5)
        if anc.y == 0:
            return "circle", float(anc.x / 2 - 0.5), 0.0
        if anc.y == 2 * code.height:
            return "circle", float(anc.x / 2 - 0.5), float(code.height - 1)

    msg = f"Unsupported stabiliser weight for heatmap rendering: {weight}."
    raise ValueError(msg)


def plot_detection_probability_on_patch(
    code: PlanarCode,
    detection_probabilities: dict[tuple[float, ...], list[float]],
    *,
    mode: Literal["average", "median", "variance"] = "average",
    style: Literal["plaquette", "heatmap"] = "plaquette",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    show_colorbar: bool = True,
    show_data_qubits: bool = False,
) -> tuple[Figure, Axes]:
    """Plot detection probabilities on a surface-code patch.

    Two visual styles are available:

    - ``"plaquette"`` (default): fills the actual stabiliser plaquettes
      (diamond-shaped for interior weight-4 stabilisers, triangular for
      boundary weight-2 stabilisers) matching the code geometry.
    - ``"heatmap"``: simplified grid with square cells for interior and
      edge-attached circles for boundary stabilisers.

    Args:
        code: The planar code patch to draw.
        detection_probabilities: Per-round detection probabilities for each detector
            coordinate, as returned by ``Client.defect_rates()`` or
            ``detect_and_aggregate()``. Keys are ``(x, y, ...)`` tuples, values are
            lists of per-round probabilities.
        mode: How to aggregate the per-round values into a single number per detector.
        style: Visual style — ``"plaquette"`` fills the code plaquettes,
            ``"heatmap"`` uses a simplified square-and-circle grid.
        cmap: Matplotlib colour-map name for mapping probabilities to colours.
        vmin: Lower bound of the colour scale. If ``None``, inferred from data.
        vmax: Upper bound of the colour scale. If ``None``, inferred from data.
        fig: An existing matplotlib Figure. If ``None``, a new figure is created.
        ax: An existing matplotlib Axes. If ``None``, a new axes is created.
        show_colorbar: Whether to display a colour bar alongside the plot.
        show_data_qubits: Whether to draw data-qubit markers (only used when
            ``style="plaquette"``).

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        ValueError: If ``fig`` and ``ax`` are not both ``None`` or both set, or
            if an unknown style is given.

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
                code,
                rates,
                fig=main_fig,
                ax=inset_ax,
            )

        Compare plaquette and heatmap styles side by side::

            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_detection_probability_on_patch(code, rates, style="plaquette", ax=ax1)
            plot_detection_probability_on_patch(code, rates, style="heatmap", ax=ax2)
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if style not in ("plaquette", "heatmap"):
        msg = f"Unknown style: {style!r}. Expected 'plaquette' or 'heatmap'."
        raise ValueError(msg)

    aggregated = _aggregate_probabilities(detection_probabilities, mode)

    if not aggregated:
        msg = "The detection_probabilities dict is empty. Nothing to plot."
        raise ValueError(msg)

    ancilla_values = _match_ancilla_coords(aggregated, code)

    if fig is None and ax is None:
        fig, ax = plt.subplots()
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

    if style == "plaquette":
        all_vertices: list[tuple[float, float]] = []
        for stabilisers_group in code.stabilisers:
            for stabiliser in stabilisers_group:
                if stabiliser.ancilla_qubit is None:
                    continue
                anc = stabiliser.ancilla_qubit.unique_identifier
                val = ancilla_values.get((float(anc.x), float(anc.y)))
                if val is None:
                    continue
                vertices = _stabiliser_vertices(stabiliser)
                all_vertices.extend(vertices)
                color = cmap_obj(norm(val))
                ax.add_patch(
                    Polygon(
                        vertices,
                        closed=True,
                        facecolor=color,
                        edgecolor="none",
                        zorder=1,
                    )
                )

        if show_data_qubits:
            for qubit in code.data_qubits:
                coord = qubit.unique_identifier
                ax.add_patch(
                    Circle(
                        (float(coord.x), float(coord.y)),
                        radius=0.12,
                        facecolor="white",
                        edgecolor="#666666",
                        linewidth=0.6,
                        zorder=2,
                    )
                )

        xs = [v[0] for v in all_vertices]
        ys = [v[1] for v in all_vertices]
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    elif style == "heatmap":
        grid_width = code.width - 1
        grid_height = code.height - 1
        boundary_patches: list[Circle] = []
        interior_patches: list[Rectangle] = []

        for stabilisers_group in code.stabilisers:
            for stabiliser in stabilisers_group:
                if stabiliser.ancilla_qubit is None:
                    continue
                anc = stabiliser.ancilla_qubit.unique_identifier
                val = ancilla_values.get((float(anc.x), float(anc.y)))
                if val is None:
                    continue

                patch_type, x, y = _heatmap_patch_position(stabiliser, code)
                color = cmap_obj(norm(val))
                if patch_type == "square":
                    interior_patches.append(
                        Rectangle(
                            (x, y),
                            1,
                            1,
                            facecolor=color,
                            edgecolor="none",
                            zorder=2,
                        )
                    )
                else:
                    boundary_patches.append(
                        Circle(
                            (x, y),
                            radius=0.42,
                            facecolor=color,
                            edgecolor="none",
                            zorder=1,
                        )
                    )

        for patch in boundary_patches + interior_patches:
            ax.add_patch(patch)

        ax.set_xlim(-0.55, grid_width + 0.55)
        ax.set_ylim(-0.55, grid_height + 0.55)
    else:
        msg = f"Unknown style: {style!r}. Expected 'plaquette' or 'heatmap'."
        raise ValueError(msg)

    ax.set_aspect("equal")
    ax.axis("off")

    if show_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.08,
            pad=0.08,
        )
        cbar.set_label(r"$p_d$")

    return fig, ax
