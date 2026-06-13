from __future__ import annotations

import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle, Polygon, Rectangle

from deltakit_explorer.codes import RotatedPlanarCode
from deltakit_explorer.plotting import (
    DetectionProbabilityAggregation,
    plot_detection_probability_on_patch,
)
from deltakit_explorer.plotting._detection_on_patch import (
    _aggregate_probabilities,
    _match_ancilla_coords,
    _stabiliser_vertices,
    _validate_detection_probabilities,
)

mpl.use("Agg")


@pytest.fixture(autouse=True)
def _close_all_figures():
    plt.close("all")


RNG = np.random.default_rng(42)

# Ancilla coordinates for a d=3 RotatedPlanarCode (verified at runtime)
ANCILLA_COORDS = [
    (0.0, 2.0),
    (2.0, 2.0),
    (2.0, 4.0),
    (2.0, 6.0),
    (4.0, 0.0),
    (4.0, 2.0),
    (4.0, 4.0),
    (6.0, 4.0),
]
NUM_ROUNDS = 8


@pytest.fixture
def code() -> RotatedPlanarCode:
    return RotatedPlanarCode(3, 3)


@pytest.fixture
def detection_probabilities() -> dict[tuple[float, float], list[float]]:
    return {
        coord: list(RNG.uniform(0.02, 0.25, size=NUM_ROUNDS))
        for coord in ANCILLA_COORDS
    }


class TestDetectionOnPatch:
    def test_output_type(self, code, detection_probabilities) -> None:
        fig, ax = plot_detection_probability_on_patch(code, detection_probabilities)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_custom_fig_ax(self, code, detection_probabilities) -> None:
        fig, ax = plt.subplots()
        fig_out, ax_out = plot_detection_probability_on_patch(
            code, detection_probabilities, fig=fig, ax=ax
        )
        assert fig_out is fig
        assert ax_out is ax

    def test_average_mode(self, code, detection_probabilities) -> None:
        _, ax = plot_detection_probability_on_patch(
            code, detection_probabilities, mode=DetectionProbabilityAggregation.MEAN
        )
        assert len(ax.patches) == 8

    def test_plaquette_patch_shapes(self, code, detection_probabilities) -> None:
        _, ax = plot_detection_probability_on_patch(code, detection_probabilities)
        assert sum(isinstance(patch, Polygon) for patch in ax.patches) == 8
        assert not ax.axison

    def test_heatmap_patch_shapes(self, code, detection_probabilities) -> None:
        _, ax = plot_detection_probability_on_patch(
            code, detection_probabilities, style="heatmap"
        )
        assert sum(isinstance(patch, Rectangle) for patch in ax.patches) == 4
        assert sum(isinstance(patch, Circle) for patch in ax.patches) == 4
        assert not ax.axison

    def test_median_mode(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, mode=DetectionProbabilityAggregation.MEDIAN
        )
        assert isinstance(fig, plt.Figure)

    def test_variance_mode(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, mode=DetectionProbabilityAggregation.VARIANCE
        )
        assert isinstance(fig, plt.Figure)

    def test_no_colorbar(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, show_colorbar=False
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

    def test_custom_cmap(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, cmap="plasma"
        )
        assert isinstance(fig, plt.Figure)

    def test_fixed_vmin_vmax(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, vmin=0.0, vmax=0.5
        )
        assert isinstance(fig, plt.Figure)

    def test_missing_ancilla_coords(self, code) -> None:
        partial_data = {(0.0, 2.0): [0.05, 0.08, 0.07, 0.09]}
        fig, _ = plot_detection_probability_on_patch(code, partial_data)
        assert isinstance(fig, plt.Figure)

    def test_empty_raises(self, code) -> None:
        with pytest.raises(ValueError, match="empty"):
            plot_detection_probability_on_patch(code, {})

    def test_ax_alone_infers_fig(self, code, detection_probabilities) -> None:
        _, ax = plt.subplots()
        fig_out, ax_out = plot_detection_probability_on_patch(
            code, detection_probabilities, ax=ax
        )
        assert ax_out is ax
        assert fig_out is ax.get_figure()

    def test_fig_alone_infers_ax(self, code, detection_probabilities) -> None:
        fig = plt.figure()
        fig_out, ax_out = plot_detection_probability_on_patch(
            code, detection_probabilities, fig=fig
        )
        assert fig_out is fig
        assert ax_out is not None
        assert ax_out.get_figure() is fig

    def test_plot_matches_reference(
        self, code, detection_probabilities, tmp_path
    ) -> None:
        fig, _ = plot_detection_probability_on_patch(code, detection_probabilities)
        path = tmp_path / "detection_on_patch.png"
        fig.savefig(path)
        assert path.exists()
        loaded = img.imread(path)
        assert loaded.ndim == 3
        assert loaded.shape[-1] in (3, 4)

    def test_include_outlier_rounds_flag(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, include_outlier_rounds=False
        )
        assert isinstance(fig, plt.Figure)

    def test_variance_includes_outliers_by_default(
        self, code, detection_probabilities
    ) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, mode=DetectionProbabilityAggregation.VARIANCE
        )
        assert isinstance(fig, plt.Figure)

    def test_vmin_greater_than_vmax_raises(self, code, detection_probabilities) -> None:
        with pytest.raises(ValueError, match="vmin"):
            plot_detection_probability_on_patch(
                code, detection_probabilities, vmin=0.5, vmax=0.1
            )

    def test_no_matching_ancillas_raises(self, code) -> None:
        with pytest.raises(ValueError, match="do not match"):
            plot_detection_probability_on_patch(code, {(100.0, 100.0): [0.1, 0.2, 0.3]})

    def test_round_index_integration(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, round_index=0
        )
        assert isinstance(fig, plt.Figure)


class TestValidateDetectionProbabilities:
    def test_non_finite_values_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _validate_detection_probabilities({(0.0, 2.0): [np.nan, 0.2]})
        with pytest.raises(ValueError, match="non-finite"):
            _validate_detection_probabilities({(0.0, 2.0): [np.inf, 0.2]})

    def test_out_of_range_values_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            _validate_detection_probabilities({(0.0, 2.0): [-0.1, 0.2]})
        with pytest.raises(ValueError, match="outside"):
            _validate_detection_probabilities({(0.0, 2.0): [1.1, 0.2]})

    def test_non_1d_values_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            _validate_detection_probabilities({(0.0, 2.0): [[0.1, 0.2], [0.3, 0.4]]})

    def test_empty_values_per_coord_raises(self) -> None:
        with pytest.raises(ValueError, match="empty probability"):
            _validate_detection_probabilities({(0.0, 2.0): []})

    def test_coord_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _validate_detection_probabilities({(0.0,): [0.1, 0.2]})

    def test_valid_data_passes(self) -> None:
        _validate_detection_probabilities({(0.0, 2.0): [0.1, 0.2, 0.3]})


class TestAggregateProbabilities:
    def test_average(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.2, 0.3]}
        result = _aggregate_probabilities(
            data, mode=DetectionProbabilityAggregation.MEAN
        )
        assert result[(0.0, 2.0)] == pytest.approx(0.2)

    def test_median(self) -> None:
        data = {(0.0, 2.0): [0.3, 0.1, 0.2]}
        result = _aggregate_probabilities(
            data, mode=DetectionProbabilityAggregation.MEDIAN
        )
        assert result[(0.0, 2.0)] == pytest.approx(0.2)

    def test_variance_excludes_outliers_with_flag(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.5, 0.5, 0.1]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.VARIANCE,
            include_outlier_rounds=False,
        )
        inner = [0.5, 0.5]
        assert result[(0.0, 2.0)] == pytest.approx(float(np.var(inner)))

    def test_variance_short_no_removal(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.2]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.VARIANCE,
            include_outlier_rounds=False,
        )
        assert result[(0.0, 2.0)] == pytest.approx(float(np.var([0.1, 0.2])))

    def test_multiple_coords(self) -> None:
        data = {
            (0.0, 2.0): [0.1, 0.2],
            (2.0, 2.0): [0.3, 0.4],
        }
        result = _aggregate_probabilities(
            data, mode=DetectionProbabilityAggregation.MEAN
        )
        assert len(result) == 2
        assert result[(0.0, 2.0)] == pytest.approx(0.15)
        assert result[(2.0, 2.0)] == pytest.approx(0.35)

    def test_variance_includes_all_rounds_by_default(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.5, 0.5, 0.1]}
        result = _aggregate_probabilities(
            data, mode=DetectionProbabilityAggregation.VARIANCE
        )
        assert result[(0.0, 2.0)] == pytest.approx(float(np.var([0.1, 0.5, 0.5, 0.1])))

    def test_unknown_mode_raises(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.2]}
        with pytest.raises(ValueError, match="Unknown"):
            _aggregate_probabilities(data, mode="invalid")  # type: ignore[arg-type]

    def test_include_outlier_rounds_true_average(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.5, 0.5, 0.1]}
        result = _aggregate_probabilities(
            data, mode=DetectionProbabilityAggregation.MEAN, include_outlier_rounds=True
        )
        assert result[(0.0, 2.0)] == pytest.approx(0.3)

    def test_include_outlier_rounds_false_average(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.5, 0.5, 0.1]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.MEAN,
            include_outlier_rounds=False,
        )
        inner = [0.5, 0.5]
        assert result[(0.0, 2.0)] == pytest.approx(float(np.mean(inner)))

    def test_include_outlier_rounds_true_variance(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.5, 0.5, 0.1]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.VARIANCE,
            include_outlier_rounds=True,
        )
        assert result[(0.0, 2.0)] == pytest.approx(float(np.var([0.1, 0.5, 0.5, 0.1])))

    def test_include_outlier_rounds_false_variance(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.5, 0.5, 0.1]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.VARIANCE,
            include_outlier_rounds=False,
        )
        inner = [0.5, 0.5]
        assert result[(0.0, 2.0)] == pytest.approx(float(np.var(inner)))

    def test_include_outlier_rounds_false_median(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.3, 0.5, 0.9]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.MEDIAN,
            include_outlier_rounds=False,
        )
        inner = [0.3, 0.5]
        assert result[(0.0, 2.0)] == pytest.approx(float(np.median(inner)))

    def test_include_outlier_rounds_false_short_list(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.2]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.MEAN,
            include_outlier_rounds=False,
        )
        assert result[(0.0, 2.0)] == pytest.approx(0.15)

    def test_round_index_selects_specific_round(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.2, 0.3]}
        result = _aggregate_probabilities(data, round_index=1)
        assert result[(0.0, 2.0)] == pytest.approx(0.2)

    def test_round_index_negative(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.2, 0.3]}
        result = _aggregate_probabilities(data, round_index=-1)
        assert result[(0.0, 2.0)] == pytest.approx(0.3)

    def test_round_index_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _aggregate_probabilities({(0.0, 2.0): [0.1, 0.2]}, round_index=5)

    def test_round_index_takes_precedence_over_mode(self) -> None:
        data = {(0.0, 2.0): [0.1, 0.5, 0.3]}
        result = _aggregate_probabilities(
            data,
            mode=DetectionProbabilityAggregation.MEAN,
            round_index=2,
        )
        assert result[(0.0, 2.0)] == pytest.approx(0.3)


class TestMatchAncillaCoords:
    def test_basic_match(self, code) -> None:
        aggregated = {
            (0.0, 2.0): 0.15,
            (2.0, 2.0): 0.25,
        }
        result = _match_ancilla_coords(aggregated, code)
        assert (0.0, 2.0) in result
        assert (2.0, 2.0) in result
        assert result[(0.0, 2.0)] == 0.15
        assert result[(2.0, 2.0)] == 0.25

    def test_no_match(self, code) -> None:
        aggregated = {(99.0, 99.0): 0.5}
        result = _match_ancilla_coords(aggregated, code)
        assert result == {}

    def test_partial_match(self, code) -> None:
        aggregated = {
            (0.0, 2.0): 0.15,
            (99.0, 99.0): 0.5,
        }
        result = _match_ancilla_coords(aggregated, code)
        assert (0.0, 2.0) in result
        assert (99.0, 99.0) not in result

    def test_tolerance(self, code) -> None:
        aggregated = {
            (0.0005, 2.0005): 0.15,
        }
        result = _match_ancilla_coords(aggregated, code)
        assert (0.0, 2.0) in result
        assert result[(0.0, 2.0)] == 0.15


class TestDetectionOnPatchPlaquette:
    def test_plaquette_default_style(self, code, detection_probabilities) -> None:
        fig, ax = plot_detection_probability_on_patch(code, detection_probabilities)
        assert isinstance(fig, plt.Figure)
        assert all(isinstance(p, Polygon) for p in ax.patches)

    def test_plaquette_explicit(self, code, detection_probabilities) -> None:
        fig, ax = plot_detection_probability_on_patch(
            code, detection_probabilities, style="plaquette"
        )
        assert isinstance(fig, plt.Figure)
        assert all(isinstance(p, Polygon) for p in ax.patches)

    def test_plaquette_data_qubits(self, code, detection_probabilities) -> None:
        _, ax = plot_detection_probability_on_patch(
            code, detection_probabilities, show_data_qubits=True
        )
        circles = [p for p in ax.patches if isinstance(p, Circle)]
        assert len(circles) > 0
        assert any(p.radius == 0.12 for p in circles)

    def test_plaquette_no_data_qubits_default(
        self, code, detection_probabilities
    ) -> None:
        _, ax = plot_detection_probability_on_patch(
            code, detection_probabilities, style="plaquette"
        )
        circles = [p for p in ax.patches if isinstance(p, Circle)]
        assert len(circles) == 0

    def test_plaquette_no_colorbar(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, show_colorbar=False, style="plaquette"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

    def test_plaquette_variance_mode(self, code, detection_probabilities) -> None:
        fig, _ = plot_detection_probability_on_patch(
            code,
            detection_probabilities,
            mode=DetectionProbabilityAggregation.VARIANCE,
            style="plaquette",
        )
        assert isinstance(fig, plt.Figure)

    def test_plaquette_custom_fig_ax(self, code, detection_probabilities) -> None:
        fig, ax = plt.subplots()
        fig_out, ax_out = plot_detection_probability_on_patch(
            code, detection_probabilities, fig=fig, ax=ax, style="plaquette"
        )
        assert fig_out is fig
        assert ax_out is ax

    def test_unknown_style_raises(self, code, detection_probabilities) -> None:
        with pytest.raises(ValueError, match="Unknown style"):
            plot_detection_probability_on_patch(
                code,
                detection_probabilities,
                style="invalid",  # type: ignore[arg-type]
            )

    def test_plaquette_missing_ancilla_coords(self, code) -> None:
        partial_data = {(0.0, 2.0): [0.05, 0.08, 0.07, 0.09]}
        fig, _ = plot_detection_probability_on_patch(
            code, partial_data, style="plaquette"
        )
        assert isinstance(fig, plt.Figure)


class TestStabiliserVertices:
    def test_weight_4_has_4_vertices(self, code) -> None:
        for stabilisers_group in code.stabilisers:
            for stabiliser in stabilisers_group:
                if stabiliser.ancilla_qubit is None:
                    continue
                weight = len([p for p in stabiliser.paulis if p is not None])
                if weight == 4:
                    vertices = _stabiliser_vertices(stabiliser)
                    assert len(vertices) == 4

    def test_weight_2_has_3_vertices(self, code) -> None:
        for stabilisers_group in code.stabilisers:
            for stabiliser in stabilisers_group:
                if stabiliser.ancilla_qubit is None:
                    continue
                weight = len([p for p in stabiliser.paulis if p is not None])
                if weight == 2:
                    vertices = _stabiliser_vertices(stabiliser)
                    assert len(vertices) == 3

    def test_polygon_is_counter_clockwise(self, code) -> None:
        for stabilisers_group in code.stabilisers:
            for stabiliser in stabilisers_group:
                if stabiliser.ancilla_qubit is None:
                    continue
                vertices = _stabiliser_vertices(stabiliser)
                if len(vertices) < 3:
                    continue
                area = 0.0
                n = len(vertices)
                for i in range(n):
                    x1, y1 = vertices[i]
                    x2, y2 = vertices[(i + 1) % n]
                    area += x1 * y2 - x2 * y1
                area /= 2.0
                assert area > 0, f"Polygon area should be positive, got {area}"

    def test_vertices_close_to_data_qubits(self, code) -> None:
        data_coords = {
            (float(q.unique_identifier.x), float(q.unique_identifier.y))
            for q in code.data_qubits
        }
        anc_coords = {
            (float(q.unique_identifier.x), float(q.unique_identifier.y))
            for q in code.ancilla_qubits
        }
        expected = data_coords | anc_coords
        for stabilisers_group in code.stabilisers:
            for stabiliser in stabilisers_group:
                if stabiliser.ancilla_qubit is None:
                    continue
                vertices = _stabiliser_vertices(stabiliser)
                for v in vertices:
                    matches = any(
                        abs(v[0] - e[0]) < 1e-6 and abs(v[1] - e[1]) < 1e-6
                        for e in expected
                    )
                    assert matches, f"Vertex {v} not found in data/ancilla qubits"

    def test_no_ancilla_raises(self, code) -> None:
        stabiliser_no_anc = None
        for stabilisers_group in code.stabilisers:
            for stabiliser in stabilisers_group:
                weight = len([p for p in stabiliser.paulis if p is not None])
                if weight == 2 and stabiliser.ancilla_qubit is None:
                    stabiliser_no_anc = stabiliser
                    break

        if stabiliser_no_anc is not None:
            with pytest.raises(ValueError, match="ancilla"):
                _stabiliser_vertices(stabiliser_no_anc)
