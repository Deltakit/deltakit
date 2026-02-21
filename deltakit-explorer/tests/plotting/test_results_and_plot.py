# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from deltakit_explorer.analysis._lambda import LambdaResults
from deltakit_explorer.analysis._leppr import LogicalErrorProbabilityPerRoundResults
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import (
    LEPPRPlot,
    LambdaPlot,
    compute_lambda_plot,
    compute_leppr_plot,
)


# Use non-interactive backend for CI
mpl.use("Agg")


# ──────────────────── fixtures ────────────────────


@pytest.fixture()
def lambda_results() -> LambdaResults:
    """Example LambdaResults for testing."""
    return LambdaResults(
        lambda_=3.0,
        lambda_stddev=0.1,
        lambda0=1.5,
        lambda0_stddev=0.05,
    )


@pytest.fixture()
def leppr_results() -> LogicalErrorProbabilityPerRoundResults:
    """Example LEPPR results for testing."""
    return LogicalErrorProbabilityPerRoundResults(
        leppr=0.001,
        leppr_stddev=0.0001,
        spam_error=0.01,
        spam_error_stddev=0.001,
    )


@pytest.fixture()
def distances() -> np.ndarray:
    return np.array([5, 7, 9])


@pytest.fixture()
def num_rounds() -> np.ndarray:
    return np.array([2, 4, 6])


# ──────────────── compute_lambda_plot ─────────────────


class TestComputeLambdaPlot:
    def test_output_type(self, lambda_results, distances):
        result = compute_lambda_plot(lambda_results, distances)
        assert isinstance(result, LambdaPlot)

    def test_default_num_points(self, lambda_results, distances):
        result = compute_lambda_plot(lambda_results, distances)
        assert len(result.distances) == 200
        assert len(result.interpolated) == 200
        assert len(result.lower_boundary) == 200
        assert len(result.upper_boundary) == 200

    def test_custom_num_points(self, lambda_results, distances):
        result = compute_lambda_plot(lambda_results, distances, num_points=50)
        assert len(result.distances) == 50

    def test_distance_range(self, lambda_results, distances):
        result = compute_lambda_plot(lambda_results, distances)
        assert result.distances[0] == pytest.approx(distances[0])
        assert result.distances[-1] == pytest.approx(distances[-1])

    def test_interpolated_values_positive(self, lambda_results, distances):
        result = compute_lambda_plot(lambda_results, distances)
        assert np.all(result.interpolated > 0)

    def test_frozen_dataclass(self, lambda_results, distances):
        result = compute_lambda_plot(lambda_results, distances)
        with pytest.raises(AttributeError):
            result.distances = np.array([1, 2, 3])


# ──────────────── compute_leppr_plot ─────────────────


class TestComputeLepprPlot:
    def test_output_type(self, leppr_results, num_rounds):
        result = compute_leppr_plot(leppr_results, num_rounds)
        assert isinstance(result, LEPPRPlot)

    def test_default_num_points(self, leppr_results, num_rounds):
        result = compute_leppr_plot(leppr_results, num_rounds)
        assert len(result.rounds) == 200
        assert len(result.interpolated) == 200
        assert len(result.lower_boundary) == 200
        assert len(result.upper_boundary) == 200

    def test_custom_num_points(self, leppr_results, num_rounds):
        result = compute_leppr_plot(leppr_results, num_rounds, num_points=100)
        assert len(result.rounds) == 100

    def test_rounds_range(self, leppr_results, num_rounds):
        result = compute_leppr_plot(leppr_results, num_rounds)
        assert result.rounds[0] == pytest.approx(num_rounds[0])
        assert result.rounds[-1] == pytest.approx(num_rounds[-1])

    def test_boundaries_clipped(self, leppr_results, num_rounds):
        result = compute_leppr_plot(leppr_results, num_rounds)
        assert np.all(result.lower_boundary >= 0)
        assert np.all(result.lower_boundary <= 1)
        assert np.all(result.upper_boundary >= 0)
        assert np.all(result.upper_boundary <= 1)

    def test_frozen_dataclass(self, leppr_results, num_rounds):
        result = compute_leppr_plot(leppr_results, num_rounds)
        with pytest.raises(AttributeError):
            result.rounds = np.array([1, 2, 3])


# ──────────────── generic plot function ─────────────────


class TestPlot:
    def test_plot_lambda(self, lambda_results, distances):
        lambda_plot = compute_lambda_plot(lambda_results, distances)
        fig, ax = plot(lambda_plot)
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Logical Error Probability Per Round Fit"
        assert ax.get_xlabel() == "Code distance"
        plt.close(fig)

    def test_plot_leppr(self, leppr_results, num_rounds):
        leppr_plot = compute_leppr_plot(leppr_results, num_rounds)
        fig, ax = plot(leppr_plot)
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Logical Error Probability Per Round Fit"
        assert ax.get_xlabel() == "Rounds"
        plt.close(fig)

    def test_plot_with_existing_fig_ax(self, lambda_results, distances):
        lambda_plot = compute_lambda_plot(lambda_results, distances)
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot(lambda_plot, fig=fig, ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_raises_on_mismatched_fig_ax(self, lambda_results, distances):
        lambda_plot = compute_lambda_plot(lambda_results, distances)
        fig, _ = plt.subplots()
        with pytest.raises(ValueError, match="both None or both set"):
            plot(lambda_plot, fig=fig, ax=None)
        plt.close(fig)

    def test_plot_raises_on_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported result type"):
            plot("invalid")
