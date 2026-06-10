from __future__ import annotations

import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle, Rectangle

from deltakit_explorer.codes import RotatedPlanarCode
from deltakit_explorer.plotting import plot_detection_probability_on_patch
from deltakit_explorer.plotting._detection_on_patch import (
    _aggregate_probabilities,
    _match_ancilla_coords,
)

mpl.use("Agg")

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
    def test_output_type(self, code, detection_probabilities):
        fig, ax = plot_detection_probability_on_patch(code, detection_probabilities)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_custom_fig_ax(self, code, detection_probabilities):
        fig, ax = plt.subplots()
        fig_out, ax_out = plot_detection_probability_on_patch(
            code, detection_probabilities, fig=fig, ax=ax
        )
        assert fig_out is fig
        assert ax_out is ax

    def test_average_mode(self, code, detection_probabilities):
        _, ax = plot_detection_probability_on_patch(
            code, detection_probabilities, mode="average"
        )
        assert len(ax.patches) == 8

    def test_heatmap_patch_shapes(self, code, detection_probabilities):
        _, ax = plot_detection_probability_on_patch(code, detection_probabilities)
        assert sum(isinstance(patch, Rectangle) for patch in ax.patches) == 4
        assert sum(isinstance(patch, Circle) for patch in ax.patches) == 4
        assert not ax.axison

    def test_median_mode(self, code, detection_probabilities):
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, mode="median"
        )
        assert isinstance(fig, plt.Figure)

    def test_variance_mode(self, code, detection_probabilities):
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, mode="variance"
        )
        assert isinstance(fig, plt.Figure)

    def test_no_colorbar(self, code, detection_probabilities):
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, show_colorbar=False
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

    def test_custom_cmap(self, code, detection_probabilities):
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, cmap="plasma"
        )
        assert isinstance(fig, plt.Figure)

    def test_fixed_vmin_vmax(self, code, detection_probabilities):
        fig, _ = plot_detection_probability_on_patch(
            code, detection_probabilities, vmin=0.0, vmax=0.5
        )
        assert isinstance(fig, plt.Figure)

    def test_missing_ancilla_coords(self, code):
        partial_data = {(0.0, 2.0): [0.05, 0.08, 0.07, 0.09]}
        fig, _ = plot_detection_probability_on_patch(code, partial_data)
        assert isinstance(fig, plt.Figure)

    def test_empty_raises(self, code):
        with pytest.raises(ValueError, match="empty"):
            plot_detection_probability_on_patch(code, {})

    def test_fig_ax_mismatch_raises(self, code, detection_probabilities):
        fig = plt.figure()
        with pytest.raises(ValueError, match="both None or both set"):
            plot_detection_probability_on_patch(
                code, detection_probabilities, fig=fig, ax=None
            )

    def test_plot_matches_reference(self, code, detection_probabilities, tmp_path):
        fig, _ = plot_detection_probability_on_patch(code, detection_probabilities)
        path = tmp_path / "detection_on_patch.png"
        fig.savefig(path)
        assert path.exists()
        loaded = img.imread(path)
        assert loaded.ndim == 3
        assert loaded.shape[-1] in (3, 4)


class TestAggregateProbabilities:
    def test_average(self):
        data = {(0.0, 2.0): [0.1, 0.2, 0.3]}
        result = _aggregate_probabilities(data, mode="average")
        assert result[(0.0, 2.0)] == pytest.approx(0.2)

    def test_median(self):
        data = {(0.0, 2.0): [0.3, 0.1, 0.2]}
        result = _aggregate_probabilities(data, mode="median")
        assert result[(0.0, 2.0)] == pytest.approx(0.2)

    def test_variance_removes_outliers(self):
        data = {(0.0, 2.0): [0.1, 0.5, 0.5, 0.1]}
        result = _aggregate_probabilities(data, mode="variance")
        inner = [0.5, 0.5]
        assert result[(0.0, 2.0)] == pytest.approx(float(np.var(inner)))

    def test_variance_short_no_removal(self):
        data = {(0.0, 2.0): [0.1, 0.2]}
        result = _aggregate_probabilities(data, mode="variance")
        assert result[(0.0, 2.0)] == pytest.approx(float(np.var([0.1, 0.2])))

    def test_multiple_coords(self):
        data = {
            (0.0, 2.0): [0.1, 0.2],
            (2.0, 2.0): [0.3, 0.4],
        }
        result = _aggregate_probabilities(data, mode="average")
        assert len(result) == 2
        assert result[(0.0, 2.0)] == pytest.approx(0.15)
        assert result[(2.0, 2.0)] == pytest.approx(0.35)

    def test_unknown_mode_raises(self):
        data = {(0.0, 2.0): [0.1, 0.2]}
        with pytest.raises(ValueError, match="Unknown mode"):
            _aggregate_probabilities(data, mode="mean")


class TestMatchAncillaCoords:
    def test_basic_match(self, code):
        aggregated = {
            (0.0, 2.0): 0.15,
            (2.0, 2.0): 0.25,
        }
        result = _match_ancilla_coords(aggregated, code)
        assert (0.0, 2.0) in result
        assert (2.0, 2.0) in result
        assert result[(0.0, 2.0)] == 0.15
        assert result[(2.0, 2.0)] == 0.25

    def test_no_match(self, code):
        aggregated = {(99.0, 99.0): 0.5}
        result = _match_ancilla_coords(aggregated, code)
        assert result == {}

    def test_partial_match(self, code):
        aggregated = {
            (0.0, 2.0): 0.15,
            (99.0, 99.0): 0.5,
        }
        result = _match_ancilla_coords(aggregated, code)
        assert (0.0, 2.0) in result
        assert (99.0, 99.0) not in result

    def test_tolerance(self, code):
        aggregated = {
            (0.0005, 2.0005): 0.15,
        }
        result = _match_ancilla_coords(aggregated, code)
        assert (0.0, 2.0) in result
        assert result[(0.0, 2.0)] == 0.15
