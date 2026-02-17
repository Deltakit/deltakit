# (c) Copyright Riverlane 2020-2025.
"""Helpers for drawing surface-code style patches."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from deltakit_circuit import PauliX
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from deltakit_explorer.codes._stabiliser import Stabiliser
from deltakit_explorer.enums._basic_enums import DrawingColours


def _count_nones_at_the_end(stabiliser: Stabiliser) -> int:
    """Count trailing ``None`` values in a stabiliser's Pauli list.

    Args:
        stabiliser: Stabiliser to inspect.

    Returns:
        Number of consecutive ``None`` values at the end of ``stabiliser.paulis``.
    """
    counter = 0
    for pauli in stabiliser.paulis[::-1]:
        if pauli is None:
            counter += 1
        else:
            break
    return counter


def _contains_x(stabiliser: Stabiliser) -> int:
    """Return whether a stabiliser contains any ``PauliX`` operators.

    Args:
        stabiliser: Stabiliser to inspect.

    Returns:
        ``1`` if at least one non-``None`` Pauli is ``PauliX``, otherwise ``0``.
    """
    for pauli in stabiliser.paulis:
        if isinstance(pauli, PauliX):
            return 1
    return 0


def _sort_stabilisers(stabilisers: tuple[Stabiliser, ...]) -> tuple[Stabiliser, ...]:
    """Sort stabilisers matching legacy planar ordering semantics.

    Args:
        stabilisers: Stabilisers to sort.

    Returns:
        Stabilisers sorted by trailing ``None`` count (descending), then by
        whether they contain a ``PauliX`` (descending).
    """
    return tuple(
        sorted(
            sorted(stabilisers, key=_contains_x, reverse=True),
            key=_count_nones_at_the_end,
            reverse=True,
        )
    )


def _wrap_periodic_axis(coords: list[int], period: int) -> list[int]:
    """Wrap coordinates across a periodic boundary for plotting.

    Args:
        coords: Axis coordinates for a single plaquette polygon.
        period: Axis period.

    Returns:
        Coordinates adjusted to avoid plotting across the full periodic span.
    """
    upper = period - 1
    if 0 in coords and upper in coords:
        if coords.count(0) > coords.count(upper):
            return [-1 if coord == upper else coord for coord in coords]
        return [period if coord == 0 else coord for coord in coords]
    return coords


def _order_vertices_by_angle(
    x_coords: list[int], y_coords: list[int]
) -> tuple[list[int], list[int]]:
    """Order polygon vertices by angle around their centroid.

    Args:
        x_coords: Polygon x-coordinates.
        y_coords: Polygon y-coordinates.

    Returns:
        Reordered x- and y-coordinate lists.
    """
    x_mean = sum(x_coords) / len(x_coords)
    y_mean = sum(y_coords) / len(y_coords)
    order = sorted(
        range(len(x_coords)),
        key=lambda index: math.atan2(y_coords[index] - y_mean, x_coords[index] - x_mean),
    )
    return [x_coords[index] for index in order], [y_coords[index] for index in order]


def draw_surface_code_patch(
    code,
    filename: str | None = None,
    margin: int = 2,
    show_legend: bool = False,
    sort_stabilisers: bool = False,
    periodic: tuple[int, int] | None = None,
    order_vertices_by_angle: bool = False,
) -> tuple[Figure, Axes]:
    """Draw a surface-code style patch.

    Args:
        code: Code object exposing ``qubits``, ``stabilisers``, ``data_qubits``,
            ``ancilla_qubits`` and ``use_ancilla_qubits`` attributes.
        filename: Optional output path. If provided, the figure is saved and closed.
        margin: Margin to apply around the qubit-coordinate bounding box.
        show_legend: Whether to add the toric-style legend.
        sort_stabilisers: Whether to sort stabilisers with legacy planar ordering.
        periodic: Optional ``(period_x, period_y)`` for toric boundary wrapping.
        order_vertices_by_angle: Whether to reorder each plaquette polygon by angle
            around its centroid.

    Returns:
        ``(fig, ax)`` containing the resulting patch drawing.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)

    all_qubit_x_coords = [qubit.unique_identifier.x for qubit in code.qubits]
    all_qubit_y_coords = [qubit.unique_identifier.y for qubit in code.qubits]
    ax.set_xlim(min(all_qubit_x_coords) - margin, max(all_qubit_x_coords) + margin)
    ax.set_ylim(min(all_qubit_y_coords) - margin, max(all_qubit_y_coords) + margin)

    stabilisers: tuple[Stabiliser, ...] = tuple(
        itertools.chain.from_iterable(code.stabilisers)
    )
    if sort_stabilisers:
        stabilisers = _sort_stabilisers(stabilisers)

    for stabiliser in stabilisers:
        paulis = [pauli for pauli in stabiliser.paulis if pauli is not None]
        x_coords = [pauli.qubit.unique_identifier[0] for pauli in paulis]
        y_coords = [pauli.qubit.unique_identifier[1] for pauli in paulis]

        if len(paulis) == 2 and stabiliser.ancilla_qubit is not None:
            ancilla_coords = stabiliser.ancilla_qubit.unique_identifier
            x_coords.append(ancilla_coords[0])
            y_coords.append(ancilla_coords[1])

        if periodic is not None:
            period_x, period_y = periodic
            x_coords = _wrap_periodic_axis(x_coords, period_x)
            y_coords = _wrap_periodic_axis(y_coords, period_y)

        if order_vertices_by_angle:
            x_coords, y_coords = _order_vertices_by_angle(x_coords, y_coords)
        elif len(paulis) == 4:
            x_coords[2], x_coords[3] = x_coords[3], x_coords[2]
            y_coords[2], y_coords[3] = y_coords[3], y_coords[2]

        ax.fill(
            x_coords,
            y_coords,
            color=(
                DrawingColours.X_COLOUR.value
                if isinstance(paulis[0], PauliX)
                else DrawingColours.Z_COLOUR.value
            ),
            alpha=1,
        )

    for qubit in code.data_qubits:
        ax.add_artist(
            plt.Circle(
                qubit.unique_identifier,
                0.2,
                color=DrawingColours.DATA_QUBIT_COLOUR.value,
                alpha=1,
            )
        )

    if code.use_ancilla_qubits:
        for qubit in code.ancilla_qubits:
            ax.add_artist(
                plt.Circle(
                    qubit.unique_identifier,
                    0.2,
                    color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                    alpha=1,
                )
            )

    ax.set_aspect(1)

    legend = None
    if show_legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Data Qubit",
                markerfacecolor=DrawingColours.DATA_QUBIT_COLOUR.value,
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Ancilla Qubit",
                markerfacecolor=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                markersize=15,
            ),
            Patch(facecolor=DrawingColours.X_COLOUR.value, label="X Stabiliser"),
            Patch(facecolor=DrawingColours.Z_COLOUR.value, label="Z Stabiliser"),
        ]
        legend = ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
        )

    if filename is not None:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if legend is not None:
            fig.savefig(filename, bbox_extra_artists=(legend,), bbox_inches="tight")
        else:
            fig.savefig(filename)
        plt.close(fig)

    return fig, ax
