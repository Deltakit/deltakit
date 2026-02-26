# (c) Copyright Riverlane 2020-2025.
"""Tests for LEPPR and Lambda plotting functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from deltakit_explorer.analysis import (
    calculate_lambda_and_lambda_stddev,
    compute_logical_error_per_round,
)
from deltakit_explorer.plotting import plot_lambda, plot_leppr


class TestPlotLeppr:
    """Tests for plot_leppr."""

    # Synthetic data following LEP(r) = (1 - (1-2*spam)*(1-2*leppr)^r) / 2
    # with leppr=0.05, spam=0.001 so the fit has high R2 and no warning.
    _num_rounds = [2, 4, 6, 8]
    _lep = np.array(
        [(1 - (1 - 0.002) * (0.9**r)) / 2 for r in _num_rounds], dtype=np.float64
    )
    _lep_std = _lep * 0.05  # 5% relative uncertainty

    def test_plot_leppr_creates_figure(self, tmp_path: Path) -> None:
        matplotlib.use("Agg")
        res = compute_logical_error_per_round(
            self._num_rounds, self._lep, self._lep_std, force_include_single_round=True
        )
        ax = plot_leppr(
            self._num_rounds,
            self._lep,
            self._lep_std,
            leppr=res.leppr,
            leppr_stddev=res.leppr_stddev,
            spam_error=res.spam_error,
        )
        assert ax is not None
        assert ax.figure is not None
        out = tmp_path / "leppr.png"
        ax.figure.savefig(out)
        assert out.exists()

    def test_plot_leppr_with_ax(self) -> None:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax_in = plt.subplots()
        res = compute_logical_error_per_round(
            self._num_rounds[:3],
            self._lep[:3],
            self._lep_std[:3],
            force_include_single_round=True,
        )
        ax_out = plot_leppr(
            self._num_rounds[:3],
            self._lep[:3],
            self._lep_std[:3],
            leppr=res.leppr,
            leppr_stddev=res.leppr_stddev,
            spam_error=res.spam_error,
            ax=ax_in,
        )
        assert ax_out is ax_in

    def test_plot_leppr_with_label(self) -> None:
        matplotlib.use("Agg")
        res = compute_logical_error_per_round(
            self._num_rounds[:3],
            self._lep[:3],
            self._lep_std[:3],
            force_include_single_round=True,
        )
        ax = plot_leppr(
            self._num_rounds[:3],
            self._lep[:3],
            self._lep_std[:3],
            leppr=res.leppr,
            leppr_stddev=res.leppr_stddev,
            spam_error=res.spam_error,
            label="Experiment 1",
        )
        assert ax.get_legend() is not None


class TestPlotLambda:
    """Tests for plot_lambda."""

    def test_plot_lambda_creates_figure(self, tmp_path: Path) -> None:
        matplotlib.use("Agg")
        distances = [5, 7, 9]
        lep_per_round = np.array([1.992e-04, 4.314e-05, 7.556e-06])
        lep_stddev = np.array([1.2e-05, 9.3e-06, 3.9e-06])
        res = calculate_lambda_and_lambda_stddev(distances, lep_per_round, lep_stddev)
        ax = plot_lambda(
            distances,
            lep_per_round,
            lep_stddev,
            lambda_value=res.lambda_,
            lambda_stddev=res.lambda_stddev,
            lambda0=res.lambda0,
        )
        assert ax is not None
        assert ax.figure is not None
        out = tmp_path / "lambda.png"
        ax.figure.savefig(out)
        assert out.exists()

    def test_plot_lambda_with_ax(self) -> None:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax_in = plt.subplots()
        distances = [5, 7, 9]
        lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
        lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
        res = calculate_lambda_and_lambda_stddev(distances, lep_per_round, lep_stddev)
        ax_out = plot_lambda(
            distances,
            lep_per_round,
            lep_stddev,
            lambda_value=res.lambda_,
            lambda_stddev=res.lambda_stddev,
            lambda0=res.lambda0,
            ax=ax_in,
        )
        assert ax_out is ax_in

    def test_plot_lambda_with_label(self) -> None:
        matplotlib.use("Agg")
        distances = [5, 7, 9]
        lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
        lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
        res = calculate_lambda_and_lambda_stddev(distances, lep_per_round, lep_stddev)
        ax = plot_lambda(
            distances,
            lep_per_round,
            lep_stddev,
            lambda_value=res.lambda_,
            lambda_stddev=res.lambda_stddev,
            lambda0=res.lambda0,
            label="Decoder A",
        )
        assert ax.get_legend() is not None

    @pytest.mark.parametrize("method", ["d", "(d+1)/2", "direct"])
    def test_plot_lambda_methods(self, method: str, tmp_path: Path) -> None:
        matplotlib.use("Agg")
        distances = [5, 7, 9]
        lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
        lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
        res = calculate_lambda_and_lambda_stddev(
            distances, lep_per_round, lep_stddev, method=method
        )
        ax = plot_lambda(
            distances,
            lep_per_round,
            lep_stddev,
            lambda_value=res.lambda_,
            lambda_stddev=res.lambda_stddev,
            lambda0=res.lambda0,
        )
        assert ax is not None
        # Sanitize method for filename: "/" is path separator on Unix
        safe_name = method.replace("/", "_")
        out = tmp_path / f"lambda_{safe_name}.png"
        ax.figure.savefig(out)
        assert out.exists()
