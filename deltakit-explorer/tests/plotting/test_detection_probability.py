from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pytest
from deltakit_circuit import Coord2D
from matplotlib.patches import Polygon

from deltakit_explorer.codes import RotatedPlanarCode
from deltakit_explorer.codes._stabiliser import Stabiliser
from deltakit_explorer.plotting import (
    DetectionProbabilityAggregation,
    plot_detection_probability_on_patch,
)
from deltakit_explorer.plotting._detection_probability import (
    _aggregate_detection_probabilities,
    _find_detection_probability,
    _stabiliser_vertices,
)


def _rates_for_code(
    code: RotatedPlanarCode,
    values: list[float] | None = None,
) -> dict[tuple[float, float], list[float]]:
    rates = values or [0.1, 0.2, 0.3, 0.4]
    return {
        (
            float(stabiliser.ancilla_qubit.unique_identifier.x),
            float(stabiliser.ancilla_qubit.unique_identifier.y),
        ): rates
        for stabiliser in chain.from_iterable(code.stabilisers)
        if stabiliser.ancilla_qubit is not None
    }


@pytest.mark.parametrize(
    ("aggregation", "round_index", "expected"),
    [
        (DetectionProbabilityAggregation.MEAN, None, 0.25),
        (DetectionProbabilityAggregation.MEDIAN, None, 0.25),
        (DetectionProbabilityAggregation.VARIANCE, None, 0.0025),
        (DetectionProbabilityAggregation.MEAN, 1, 0.2),
        (DetectionProbabilityAggregation.MEAN, -1, 0.4),
    ],
)
def test_aggregate_detection_probabilities(
    aggregation: DetectionProbabilityAggregation,
    round_index: int | None,
    expected: float,
) -> None:
    values = _aggregate_detection_probabilities(
        {(1.0, 2.0): [0.1, 0.2, 0.3, 0.4]},
        aggregation=aggregation,
        round_index=round_index,
    )

    assert values == {Coord2D(1.0, 2.0): pytest.approx(expected)}


def test_variance_requires_an_interior_round() -> None:
    with pytest.raises(ValueError, match="at least three rounds"):
        _aggregate_detection_probabilities(
            {(1.0, 2.0): [0.1, 0.2]},
            aggregation=DetectionProbabilityAggregation.VARIANCE,
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({}, "empty"),
        ({(1.0,): [0.1]}, "x and y"),
        ({(1.0, 2.0): []}, "no probability values"),
        ({(1.0, 2.0): [[0.1]]}, "one value per round"),
        ({(1.0, 2.0): [np.nan]}, "non-finite"),
        ({(1.0, 2.0): [-0.1]}, r"outside \[0, 1\]"),
        ({(1.0, 2.0): [1.1]}, r"outside \[0, 1\]"),
    ],
)
def test_aggregate_detection_probabilities_rejects_invalid_data(
    probabilities: dict[tuple[float, ...], list[float] | list[list[float]]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _aggregate_detection_probabilities(probabilities)


def test_aggregate_detection_probabilities_rejects_missing_round() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _aggregate_detection_probabilities(
            {(1.0, 2.0): [0.1]},
            round_index=2,
        )


def test_stabiliser_vertices_include_boundary_ancilla() -> None:
    code = RotatedPlanarCode(3, 3)
    stabilisers = tuple(chain.from_iterable(code.stabilisers))
    boundary = next(
        stabiliser for stabiliser in stabilisers if len(stabiliser.data_qubits) == 2
    )

    vertices = _stabiliser_vertices(boundary)
    ancilla = boundary.ancilla_qubit

    assert ancilla is not None
    assert len(vertices) == 3
    assert (
        float(ancilla.unique_identifier.x),
        float(ancilla.unique_identifier.y),
    ) in vertices


def test_stabiliser_vertices_order_interior_plaquette() -> None:
    code = RotatedPlanarCode(3, 3)
    stabilisers = tuple(chain.from_iterable(code.stabilisers))
    interior = next(
        stabiliser for stabiliser in stabilisers if len(stabiliser.data_qubits) == 4
    )

    vertices = _stabiliser_vertices(interior)
    center = np.mean(np.asarray(vertices), axis=0)
    angles = [
        np.arctan2(vertex[1] - center[1], vertex[0] - center[0]) for vertex in vertices
    ]

    assert len(vertices) == 4
    assert angles == sorted(angles)


def test_find_detection_probability_uses_coordinate_tolerance() -> None:
    code = RotatedPlanarCode(3, 3)
    stabiliser = next(chain.from_iterable(code.stabilisers))
    ancilla = stabiliser.ancilla_qubit

    assert ancilla is not None
    probabilities = {
        Coord2D(
            ancilla.unique_identifier.x + 5e-7,
            ancilla.unique_identifier.y - 5e-7,
        ): 0.25
    }

    assert _find_detection_probability(stabiliser, probabilities, 1e-6) == 0.25
    assert _find_detection_probability(stabiliser, probabilities, 1e-8) is None


def test_find_detection_probability_without_ancilla_returns_none() -> None:
    code = RotatedPlanarCode(3, 3)
    stabiliser = Stabiliser(next(chain.from_iterable(code.stabilisers)).paulis)

    assert stabiliser.ancilla_qubit is None
    assert (
        _find_detection_probability(
            stabiliser,
            {Coord2D(0.0, 0.0): 0.25},
            1e-6,
        )
        is None
    )


def test_stabiliser_vertices_requires_ancilla() -> None:
    code = RotatedPlanarCode(3, 3)
    stabiliser = Stabiliser(next(chain.from_iterable(code.stabilisers)).paulis)

    with pytest.raises(ValueError, match="ancilla qubits"):
        _stabiliser_vertices(stabiliser)


def test_plot_detection_probability_fills_stabiliser_plaquettes() -> None:
    code = RotatedPlanarCode(3, 3)

    figure, axes = plot_detection_probability_on_patch(
        code,
        _rates_for_code(code),
        show_colorbar=False,
    )

    stabilisers = tuple(chain.from_iterable(code.stabilisers))
    polygons = [patch for patch in axes.patches if isinstance(patch, Polygon)]
    polygon_vertex_counts = sorted(len(polygon.get_xy()) - 1 for polygon in polygons)
    expected_vertex_counts = sorted(
        len([pauli for pauli in stabiliser.paulis if pauli is not None])
        + (1 if len(stabiliser.data_qubits) == 2 else 0)
        for stabiliser in stabilisers
    )

    assert len(polygons) == len(stabilisers)
    assert polygon_vertex_counts == expected_vertex_counts
    assert axes.get_aspect() == 1.0
    plt.close(figure)


def test_plot_detection_probability_reuses_axes_for_inset() -> None:
    code = RotatedPlanarCode(3, 3)
    figure, axes = plt.subplots()
    inset = axes.inset_axes([0.1, 0.1, 0.4, 0.4])

    returned_figure, returned_axes = plot_detection_probability_on_patch(
        code,
        _rates_for_code(code),
        ax=inset,
        show_colorbar=False,
    )

    assert returned_figure is figure
    assert returned_axes is inset
    assert inset in axes.child_axes
    plt.close(figure)


def test_plot_detection_probability_rejects_unmatched_coordinates() -> None:
    code = RotatedPlanarCode(3, 3)

    with pytest.raises(ValueError, match="do not match"):
        plot_detection_probability_on_patch(
            code,
            {(100.0, 100.0): [0.1]},
        )


def test_plot_detection_probability_uses_shared_colour_scale() -> None:
    code = RotatedPlanarCode(3, 3)

    figure, axes = plot_detection_probability_on_patch(
        code,
        _rates_for_code(code, [0.2]),
        vmin=0.0,
        vmax=1.0,
    )

    polygons = [patch for patch in axes.patches if isinstance(patch, Polygon)]
    assert polygons
    assert all(
        np.allclose(polygon.get_facecolor(), polygons[0].get_facecolor())
        for polygon in polygons
    )
    assert len(figure.axes) == 2
    plt.close(figure)
