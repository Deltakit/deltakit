from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Polygon

from deltakit_explorer.codes import RotatedPlanarCode
from deltakit_explorer.plotting import plot_detection_probability_on_patch
from deltakit_explorer.plotting._detection_probability import (
    _aggregate_detection_probabilities,
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
        ("mean", None, 2.5),
        ("median", None, 2.5),
        ("variance", None, 0.25),
        ("mean", 1, 2.0),
        ("mean", -1, 4.0),
    ],
)
def test_aggregate_detection_probabilities(
    aggregation: str,
    round_index: int | None,
    expected: float,
) -> None:
    values = _aggregate_detection_probabilities(
        {(1.0, 2.0): [1.0, 2.0, 3.0, 4.0]},
        aggregation=aggregation,
        round_index=round_index,
    )

    assert values == {(1.0, 2.0): expected}


def test_variance_requires_an_interior_round() -> None:
    with pytest.raises(ValueError, match="at least three rounds"):
        _aggregate_detection_probabilities(
            {(1.0, 2.0): [1.0, 2.0]},
            aggregation="variance",
        )


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
