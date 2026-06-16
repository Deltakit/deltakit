# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import plot_leppr
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import (
    LambdaResult,
    interpolate_lambda,
    interpolate_leppr,
)
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)

# Use non-interactive backend for CI
mpl.use("Agg")


class TestComputeLambdaPlot:
    def test_output_type(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        assert isinstance(result, LambdaResult)

    def test_default_num_points(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        assert len(result.distances) == 200
        assert len(result.interpolated) == 200
        assert len(result.lower_boundary) == 200
        assert len(result.upper_boundary) == 200

    def test_custom_num_points(self, lambda_results):
        result = interpolate_lambda(lambda_results, num_points=50)
        assert len(result.distances) == 50

    def test_distance_range(self, lambda_results, distances):
        result = interpolate_lambda(lambda_results)
        assert result.distances[0] == pytest.approx(distances[0])
        assert result.distances[-1] == pytest.approx(distances[-1])

    def test_interpolated_values_positive(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        assert np.all(result.interpolated >= 0)

    def test_frozen_dataclass(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        with pytest.raises(AttributeError):
            result.distances = np.array([1, 2, 3])


class TestComputeLepprPlot:
    def test_output_type(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        assert isinstance(result, LEPPRResult)

    def test_default_num_points(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        assert len(result.rounds) == 200
        assert len(result.interpolated) == 200
        assert len(result.lower_boundary) == 200
        assert len(result.upper_boundary) == 200

    def test_custom_num_points(self, leppr_results):
        result = interpolate_leppr(leppr_results, num_points=100)
        assert len(result.rounds) == 100

    def test_rounds_range(self, leppr_results, num_rounds):
        result = interpolate_leppr(leppr_results)
        assert result.rounds[0] == pytest.approx(num_rounds[0])
        assert result.rounds[-1] == pytest.approx(num_rounds[-1])

    def test_boundaries_clipped(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        assert np.all(result.lower_boundary >= 0)
        assert np.all(result.lower_boundary <= 1)
        assert np.all(result.upper_boundary >= 0)
        assert np.all(result.upper_boundary <= 1)

    def test_frozen_dataclass(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        with pytest.raises(AttributeError):
            result.rounds = np.array([1, 2, 3])


class TestPlot:
    def test_plot_interpolates_lambda_data(self, lambda_results):
        fig, ax = plot(lambda_results, num_points=50)
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Error Suppression Factor Λ"
        assert ax.get_xlabel() == "Code distance"
        assert len(ax.lines[0].get_xdata()) == 50
        plt.close(fig)

    def test_plot_interpolates_lambda_data_before_dispatch(
        self, lambda_results, monkeypatch
    ):
        lambda_result = interpolate_lambda(lambda_results)
        fig, ax = plt.subplots()

        def fake_interpolate_lambda(result, *, num_sigmas=3, num_points=200):
            assert result is lambda_results
            assert num_sigmas == 2
            assert num_points == 50
            return lambda_result

        def fake_plot_lambda(result, *, fig=None, ax=None, title=None):
            assert result is lambda_result
            assert title == "Custom title"
            return fig, ax

        monkeypatch.setattr(
            "deltakit_explorer.plotting.plotting.interpolate_lambda",
            fake_interpolate_lambda,
        )
        monkeypatch.setattr(
            "deltakit_explorer.plotting.plotting.plot_lambda", fake_plot_lambda
        )

        returned_fig, returned_ax = plot(
            lambda_results,
            num_sigmas=2,
            num_points=50,
            fig=fig,
            ax=ax,
            title="Custom title",
        )
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_interpolates_leppr_data(self, leppr_results):
        fig, ax = plot(leppr_results, num_points=50)
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Logical Error Probability per Round"
        assert ax.get_xlabel() == "Rounds"
        assert len(ax.lines[0].get_xdata()) == 50
        plt.close(fig)

    def test_plot_interpolates_leppr_data_before_dispatch(
        self, leppr_results, monkeypatch
    ):
        leppr_result = interpolate_leppr(leppr_results)
        fig, ax = plt.subplots()

        def fake_interpolate_leppr(result, *, num_sigmas=3, num_points=200):
            assert result is leppr_results
            assert num_sigmas == 2
            assert num_points == 50
            return leppr_result

        def fake_plot_leppr(result, *, fig=None, ax=None, title=None):
            assert result is leppr_result
            assert title == "Custom title"
            return fig, ax

        monkeypatch.setattr(
            "deltakit_explorer.plotting.plotting.interpolate_leppr",
            fake_interpolate_leppr,
        )
        monkeypatch.setattr(
            "deltakit_explorer.plotting.plotting.plot_leppr", fake_plot_leppr
        )

        returned_fig, returned_ax = plot(
            leppr_results,
            num_sigmas=2,
            num_points=50,
            fig=fig,
            ax=ax,
            title="Custom title",
        )
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_with_existing_fig_ax(self, lambda_results):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot(lambda_results, fig=fig, ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_raises_on_mismatched_fig_ax(self, lambda_results):
        fig, _ = plt.subplots()
        with pytest.raises(ValueError, match="both `None` or both set"):
            plot(lambda_results, fig=fig, ax=None)
        plt.close(fig)

    def test_plot_raises_on_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported result type"):
            plot("invalid")


def _assert_axes_match(ax_actual, ax_expected):
    assert ax_actual.get_title() == ax_expected.get_title()
    assert ax_actual.get_xlabel() == ax_expected.get_xlabel()
    assert ax_actual.get_ylabel() == ax_expected.get_ylabel()

    actual_legend = ax_actual.get_legend()
    expected_legend = ax_expected.get_legend()
    assert actual_legend is not None
    assert expected_legend is not None
    assert [text.get_text() for text in actual_legend.get_texts()] == [
        text.get_text() for text in expected_legend.get_texts()
    ]

    assert len(ax_actual.lines) == len(ax_expected.lines)
    for actual_line, expected_line in zip(ax_actual.lines, ax_expected.lines):
        np.testing.assert_allclose(actual_line.get_xdata(), expected_line.get_xdata())
        np.testing.assert_allclose(actual_line.get_ydata(), expected_line.get_ydata())
        assert actual_line.get_label() == expected_line.get_label()
        assert actual_line.get_color() == expected_line.get_color()

    assert len(ax_actual.collections) == len(ax_expected.collections)
    for actual_collection, expected_collection in zip(
        ax_actual.collections, ax_expected.collections
    ):
        actual_paths = actual_collection.get_paths()
        expected_paths = expected_collection.get_paths()
        assert len(actual_paths) == len(expected_paths)
        for actual_path, expected_path in zip(actual_paths, expected_paths):
            np.testing.assert_allclose(actual_path.vertices, expected_path.vertices)


class TestPlotVisualEquivalence:
    def test_lambda_plot_matches_explicit_interpolation_and_specialised_plotter(
        self, lambda_results
    ):
        lambda_result = interpolate_lambda(lambda_results, num_sigmas=2, num_points=50)

        fig_actual, ax_actual = plot(
            lambda_results, num_sigmas=2, num_points=50, title="Custom title"
        )
        fig_expected, ax_expected = plot_lambda(lambda_result, title="Custom title")

        _assert_axes_match(ax_actual, ax_expected)

        plt.close(fig_actual)
        plt.close(fig_expected)

    def test_leppr_plot_matches_explicit_interpolation_and_specialised_plotter(
        self, leppr_results
    ):
        leppr_result = interpolate_leppr(leppr_results, num_sigmas=2, num_points=50)

        fig_actual, ax_actual = plot(
            leppr_results, num_sigmas=2, num_points=50, title="Custom title"
        )
        fig_expected, ax_expected = plot_leppr(leppr_result, title="Custom title")

        _assert_axes_match(ax_actual, ax_expected)

        plt.close(fig_actual)
        plt.close(fig_expected)


class TestSpecialisedPlotters:
    def test_plot_lambda_accepts_interpolated_result(self, lambda_results):
        lambda_result = interpolate_lambda(lambda_results)
        fig, ax = plot_lambda(lambda_result)
        assert fig is not None
        assert ax.get_title() == "Error Suppression Factor Λ"
        assert ax.get_xlabel() == "Code distance"
        plt.close(fig)

    def test_plot_leppr_accepts_interpolated_result(self, leppr_results):
        leppr_result = interpolate_leppr(leppr_results)
        fig, ax = plot_leppr(leppr_result)
        assert fig is not None
        assert ax.get_title() == "Logical Error Probability per Round"
        assert ax.get_xlabel() == "Rounds"
        plt.close(fig)
