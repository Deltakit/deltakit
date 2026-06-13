# (c) Copyright Riverlane 2020-2026.
"""Plot detector probabilities directly on a planar-code patch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from deltakit_circuit import Coord2D, PauliX
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

if TYPE_CHECKING:
    from deltakit_explorer.codes._planar_code import PlanarCode
    from deltakit_explorer.codes._stabiliser import Stabiliser


class DetectionProbabilityAggregation(Enum):
    """Reduction applied to per-round detection probabilities.

    Attributes:
        MEAN: Arithmetic mean across all rounds.
        MEDIAN: Median across all rounds.
        VARIANCE: Variance across steady-state rounds, excluding the first and last.
    """

    MEAN = "mean"
    MEDIAN = "median"
    VARIANCE = "variance"


def _aggregate_detection_probabilities(
    detection_probabilities: Mapping[tuple[float, ...], Sequence[float]],
    *,
    aggregation: DetectionProbabilityAggregation = DetectionProbabilityAggregation.MEAN,
    round_index: int | None = None,
) -> dict[Coord2D, float]:
    """Reduce per-round detector probabilities to one value per coordinate.

    Args:
        detection_probabilities: Detector coordinates mapped to per-round values.
        aggregation: Reduction to apply when ``round_index`` is not provided.
        round_index: Optional round to select instead of aggregating.

    Returns:
        Detection probability or variance keyed by spatial detector coordinate.

    Raises:
        ValueError: If detector coordinates, probabilities, or the round are invalid.
    """
    aggregated: dict[Coord2D, float] = {}
    for coordinate, rates in detection_probabilities.items():
        if len(coordinate) < 2:
            msg = f"Detector coordinate {coordinate!r} must contain x and y values."
            raise ValueError(msg)

        values = np.asarray(rates, dtype=float)
        if values.ndim != 1:
            msg = f"Detector coordinate {coordinate!r} must have one value per round."
            raise ValueError(msg)
        if values.size == 0:
            msg = f"Detector coordinate {coordinate!r} has no probability values."
            raise ValueError(msg)
        if not np.all(np.isfinite(values)):
            msg = f"Detector coordinate {coordinate!r} contains a non-finite value."
            raise ValueError(msg)
        if np.any((values < 0) | (values > 1)):
            msg = (
                f"Detector coordinate {coordinate!r} contains a probability "
                "outside [0, 1]."
            )
            raise ValueError(msg)

        if round_index is not None:
            try:
                value = float(values[round_index])
            except IndexError as error:
                msg = (
                    f"Round index {round_index} is out of range for detector "
                    f"coordinate {coordinate!r}."
                )
                raise ValueError(msg) from error
        elif aggregation is DetectionProbabilityAggregation.MEAN:
            value = float(np.mean(values))
        elif aggregation is DetectionProbabilityAggregation.MEDIAN:
            value = float(np.median(values))
        elif aggregation is DetectionProbabilityAggregation.VARIANCE:
            if values.size < 3:
                msg = (
                    "Variance requires at least three rounds so the first and last "
                    "rounds can be excluded."
                )
                raise ValueError(msg)
            # State preparation and final measurement make the boundary rounds
            # expected outliers, so variance describes only the steady-state rounds.
            value = float(np.var(values[1:-1]))
        else:
            msg = f"Unknown aggregation {aggregation!r}."
            raise ValueError(msg)

        aggregated[Coord2D(float(coordinate[0]), float(coordinate[1]))] = value

    if not aggregated:
        msg = "detection_probabilities is empty; there is nothing to plot."
        raise ValueError(msg)
    return aggregated


def _stabiliser_vertices(stabiliser: Stabiliser) -> list[tuple[float, float]]:
    """Return ordered vertices for the stabiliser's plaquette.

    Args:
        stabiliser: Stabiliser whose data-qubit geometry should be drawn.

    Returns:
        Plaquette vertices ordered anticlockwise around their center.

    Raises:
        ValueError: If the stabiliser lacks the geometry required for a plaquette.
    """
    coordinates = [
        (
            float(pauli.qubit.unique_identifier[0]),
            float(pauli.qubit.unique_identifier[1]),
        )
        for pauli in stabiliser.paulis
        if pauli is not None
    ]
    if stabiliser.ancilla_qubit is None:
        msg = "Detection probabilities require stabilisers with ancilla qubits."
        raise ValueError(msg)

    if len(coordinates) == 2:
        ancilla = cast(
            Sequence[float],
            stabiliser.ancilla_qubit.unique_identifier,
        )
        coordinates.append((float(ancilla[0]), float(ancilla[1])))
    elif len(coordinates) < 3:
        msg = "A stabiliser plaquette needs at least two data qubits."
        raise ValueError(msg)

    center = np.mean(np.asarray(coordinates), axis=0)
    return sorted(
        coordinates,
        key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]),
    )


def _find_detection_probability(
    stabiliser: Stabiliser,
    probabilities: Mapping[Coord2D, float],
    tolerance: float,
) -> float | None:
    """Match a stabiliser ancilla to an aggregated detector probability.

    Args:
        stabiliser: Stabiliser whose ancilla coordinate should be matched.
        probabilities: Aggregated probabilities keyed by spatial coordinate.
        tolerance: Absolute coordinate matching tolerance.

    Returns:
        Matching probability, or ``None`` when no coordinate matches.
    """
    if stabiliser.ancilla_qubit is None:
        return None
    ancilla = cast(Sequence[float], stabiliser.ancilla_qubit.unique_identifier)
    for coordinate, probability in probabilities.items():
        if np.allclose(
            coordinate,
            (float(ancilla[0]), float(ancilla[1])),
            atol=tolerance,
            rtol=0,
        ):
            return probability
    return None


def plot_detection_probability_on_patch(
    code: PlanarCode,
    detection_probabilities: Mapping[tuple[float, ...], Sequence[float]],
    *,
    aggregation: DetectionProbabilityAggregation = DetectionProbabilityAggregation.MEAN,
    round_index: int | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Axes | None = None,
    show_colorbar: bool = True,
    show_data_qubits: bool = True,
    coordinate_tolerance: float = 1e-6,
) -> tuple[Figure, Axes]:
    """Plot detection probabilities as filled stabiliser plaquettes.

    The plot preserves the code geometry, including triangular boundary
    plaquettes, and can be drawn into an existing axes for use as an inset.

    Args:
        code: Planar code whose stabiliser patch should be drawn.
        detection_probabilities: Detector coordinates mapped to per-round
            detection probabilities.
        aggregation: Reduction used when ``round_index`` is not provided.
            ``VARIANCE`` excludes the first and last rounds because state
            preparation and final measurement make them expected outliers.
        round_index: Optional round to plot instead of aggregating all rounds.
        cmap: Matplotlib colour map used for detection probabilities.
        vmin: Optional lower limit for the shared colour scale.
        vmax: Optional upper limit for the shared colour scale.
        ax: Existing axes to draw into. A new figure is created when omitted.
        show_colorbar: Whether to add a detection-probability colour bar.
        show_data_qubits: Whether to draw data qubits over the plaquettes.
        coordinate_tolerance: Absolute tolerance when matching detector and
            ancilla coordinates.

    Returns:
        Figure and axes containing the patch plot.

    Raises:
        ValueError: If the probabilities are invalid or do not match the code.
    """
    probabilities = _aggregate_detection_probabilities(
        detection_probabilities,
        aggregation=aggregation,
        round_index=round_index,
    )
    stabilisers = tuple(chain.from_iterable(code.stabilisers))
    matched = [
        (stabiliser, probability)
        for stabiliser in stabilisers
        if (
            probability := _find_detection_probability(
                stabiliser,
                probabilities,
                coordinate_tolerance,
            )
        )
        is not None
    ]
    if not matched:
        msg = "Detector coordinates do not match any stabiliser ancilla coordinates."
        raise ValueError(msg)

    values = np.asarray([probability for _, probability in matched])
    lower = float(np.min(values)) if vmin is None else vmin
    upper = float(np.max(values)) if vmax is None else vmax
    if lower > upper:
        msg = f"vmin ({lower}) cannot be greater than vmax ({upper})."
        raise ValueError(msg)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.01)
        lower -= padding
        upper += padding

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.get_figure())

    norm = Normalize(vmin=lower, vmax=upper)
    colour_map = plt.get_cmap(cmap)
    all_vertices: list[tuple[float, float]] = []
    for stabiliser, probability in matched:
        vertices = _stabiliser_vertices(stabiliser)
        all_vertices.extend(vertices)
        paulis = [pauli for pauli in stabiliser.paulis if pauli is not None]
        edge_colour = "#b3261e" if isinstance(paulis[0], PauliX) else "#1557a0"
        axes.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=colour_map(norm(probability)),
                edgecolor=edge_colour,
                linewidth=1.2,
                zorder=1,
            )
        )

    if show_data_qubits:
        for qubit in code.data_qubits:
            coordinate = qubit.unique_identifier
            axes.add_patch(
                Circle(
                    (float(coordinate[0]), float(coordinate[1])),
                    radius=0.16,
                    facecolor="white",
                    edgecolor="#202020",
                    linewidth=0.8,
                    zorder=2,
                )
            )

    x_coordinates, y_coordinates = zip(*all_vertices, strict=True)
    axes.set_xlim(min(x_coordinates) - 0.6, max(x_coordinates) + 0.6)
    axes.set_ylim(min(y_coordinates) - 0.6, max(y_coordinates) + 0.6)
    axes.set_aspect("equal")
    axes.axis("off")

    if show_colorbar:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=colour_map)
        mappable.set_array([])
        colour_bar = figure.colorbar(mappable, ax=axes, fraction=0.05, pad=0.04)
        label = (
            f"Detection probability (round {round_index})"
            if round_index is not None
            else f"Detection probability ({aggregation.value})"
        )
        colour_bar.set_label(label)

    return figure, axes
