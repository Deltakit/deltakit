# Copyright Riverlane 2020-2025.
"""Tests for plotting functions and defect visualisations."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pytest

from deltakit_explorer import plotting
from deltakit_explorer.analysis import (
    calculate_lambda_and_lambda_stddev,
    compute_logical_error_per_round,
)
from deltakit_explorer.types import QubitCoordinateToDetectorMapping


class TestPlotLeppr:
    """Tests for plot_leppr."""

    # Synthetic data following LEP(r) = (1 - (1-2*spam)*(1-2*leppr)^r) / 2
    # with leppr=0.05, spam=0.001 so the fit has high R2 and no warning.
    _num_rounds = [2, 4, 6, 8]
    _lep = np.array(
        [(1 - (1 - 0.002) * (0.9**r)) / 2 for r in _num_rounds],
        dtype=np.float64,
    )
    _lep_std = _lep * 0.05  # 5% relative uncertainty

    def test_plot_leppr_creates_figure(self, tmp_path: Path) -> None:
        mpl.use("Agg")
        res = compute_logical_error_per_round(
            self._num_rounds,
            self._lep,
            self._lep_std,
            force_include_single_round=True,
        )
        ax = plotting.plot_leppr(
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
        mpl.use("Agg")

        fig, ax_in = plt.subplots()
        res = compute_logical_error_per_round(
            self._num_rounds[:3],
            self._lep[:3],
            self._lep_std[:3],
            force_include_single_round=True,
        )
        ax_out = plotting.plot_leppr(
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
        mpl.use("Agg")
        res = compute_logical_error_per_round(
            self._num_rounds[:3],
            self._lep[:3],
            self._lep_std[:3],
            force_include_single_round=True,
        )
        ax = plotting.plot_leppr(
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
        mpl.use("Agg")
        distances = [5, 7, 9]
        lep_per_round = np.array([1.992e-04, 4.314e-05, 7.556e-06])
        lep_stddev = np.array([1.2e-05, 9.3e-06, 3.9e-06])
        res = calculate_lambda_and_lambda_stddev(distances, lep_per_round, lep_stddev)
        ax = plotting.plot_lambda(
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
        mpl.use("Agg")

        fig, ax_in = plt.subplots()
        distances = [5, 7, 9]
        lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
        lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
        res = calculate_lambda_and_lambda_stddev(distances, lep_per_round, lep_stddev)
        ax_out = plotting.plot_lambda(
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
        mpl.use("Agg")
        distances = [5, 7, 9]
        lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
        lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
        res = calculate_lambda_and_lambda_stddev(distances, lep_per_round, lep_stddev)
        ax = plotting.plot_lambda(
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
        mpl.use("Agg")
        distances = [5, 7, 9]
        lep_per_round = [1.992e-04, 4.314e-05, 7.556e-06]
        lep_stddev = [1.2e-05, 9.3e-06, 3.9e-06]
        res = calculate_lambda_and_lambda_stddev(
            distances,
            lep_per_round,
            lep_stddev,
            method=method,
        )
        ax = plotting.plot_lambda(
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


class TestVisualisation:
    """Tests for defect visualisation helpers moved under plotting."""

    defect_rates: ClassVar[list[dict[tuple[float, ...], list[float]]]] = [
        {
            (4.0, 5.0): [0.06904, 0.18388, 0.18308, 0.18564, 0.18602, 0.18416, 0.18526, 0.12188],
            (2.0, 5.0): [0.08488, 0.19028, 0.18948, 0.19204, 0.19622, 0.20552, 0.20192, 0.14066],
            (2.0, 7.0): [0.04926, 0.10428, 0.10848, 0.10914, 0.10932, 0.11754, 0.11748, 0.04888],
            (4.0, 3.0): [0.0383, 0.10928, 0.10908, 0.1134, 0.1143, 0.11124, 0.11408, 0.07362],
        },
        {
            (5.0, 6.0): [0.04326, 0.11596, 0.11936, 0.12142, 0.12186, 0.11838, 0.12, 0.07874],
            (3.0, 4.0): [0.05906, 0.14218, 0.1584, 0.15778, 0.16064, 0.16834, 0.16924, 0.10692],
            (1.0, 4.0): [0.05468, 0.1092, 0.11364, 0.1169, 0.11988, 0.12116, 0.12476, 0.09074],
            (3.0, 6.0): [0.07178, 0.18952, 0.191, 0.1929, 0.19794, 0.20064, 0.20042, 0.11874],
        },
    ]

    detector_map: ClassVar[dict[tuple[float, ...], list[int]]] = {
        (2.0, 5.0): [0, 4, 8, 12, 16, 20, 24, 28],
        (2.0, 7.0): [2, 5, 9, 13, 17, 21, 25, 30],
        (4.0, 5.0): [3, 7, 11, 15, 19, 23, 27, 31],
        (4.0, 3.0): [1, 6, 10, 14, 18, 22, 26, 29],
    }

    defect_coords: ClassVar[dict[int, tuple[float, ...]]] = {
        0: (1.0, 4.0, 0.0),
        1: (3.0, 4.0, 0.0),
        2: (3.0, 6.0, 0.0),
        3: (5.0, 6.0, 0.0),
        4: (1.0, 4.0, 1.0),
        5: (3.0, 4.0, 1.0),
        6: (3.0, 6.0, 1.0),
        7: (5.0, 6.0, 1.0),
        8: (1.0, 4.0, 2.0),
        9: (3.0, 4.0, 2.0),
        10: (3.0, 6.0, 2.0),
        11: (5.0, 6.0, 2.0),
        12: (1.0, 4.0, 3.0),
        13: (3.0, 4.0, 3.0),
        14: (3.0, 6.0, 3.0),
        15: (5.0, 6.0, 3.0),
        16: (1.0, 4.0, 4.0),
        17: (3.0, 4.0, 4.0),
        18: (3.0, 6.0, 4.0),
        19: (5.0, 6.0, 4.0),
        20: (1.0, 4.0, 5.0),
        21: (3.0, 4.0, 5.0),
        22: (3.0, 6.0, 5.0),
        23: (5.0, 6.0, 5.0),
        24: (1.0, 4.0, 6.0),
        25: (3.0, 4.0, 6.0),
        26: (3.0, 6.0, 6.0),
        27: (5.0, 6.0, 6.0),
        28: (1.0, 4.0, 7.0),
        29: (3.0, 4.0, 7.0),
        30: (3.0, 6.0, 7.0),
        31: (5.0, 6.0, 7.0),
    }

    resource_folder = Path(__file__).parent / "../resources"

    def assert_the_same(self, path1: Path, path2: Path) -> None:
        self.assert_same_size(path1, path2)
        img1 = img.imread(path1)
        img2 = img.imread(path2)
        # relative variation is small
        assert np.allclose(img1, img2, rtol=0.01)

    def assert_same_size(self, path1: Path, path2: Path) -> None:
        img1 = img.imread(path1)
        img2 = img.imread(path2)
        assert img2.shape == img1.shape

    def test_detect_rate_plot(self, tmp_path: Path) -> None:
        coord_w2 = {(5, 6), (1, 4), (4, 3), (2, 7)}
        plt.figure(figsize=(6.4, 4.8))
        plotting.defect_rates(self.defect_rates, coord_w2)
        path = tmp_path / "defects_plot.png"
        path_ref = self.resource_folder / "defects_plot.png"
        plt.savefig(path)
        plt.clf()
        self.assert_the_same(path, path_ref)

    def test_plot_defect_diagram(self, tmp_path: Path) -> None:
        # force matplotlib to use different, windows-compatible backend
        mpl.use("Agg")
        plt.figure(figsize=(5, 5))
        dr = self.defect_rates[0].copy()
        dr.update(self.defect_rates[1])
        plotting.defect_diagram(self.defect_coords, dr)
        path = tmp_path / "defects_diagram.png"
        path_ref = self.resource_folder / "defects_diagram.png"
        plt.savefig(path)
        plt.clf()
        self.assert_same_size(path, path_ref)

    def test_plot_correlation_matrix(self, tmp_path: Path) -> None:
        stick = np.linspace(-0.5, 0.5, 32).reshape(-1, 1)
        mx = stick @ stick.T
        plt.figure(figsize=(5, 5))
        plotting.correlation_matrix(
            mx,
            QubitCoordinateToDetectorMapping(self.detector_map),
        )
        path = tmp_path / "corr.png"
        path_ref = self.resource_folder / "corr.png"
        plt.savefig(path)
        plt.clf()
        self.assert_the_same(path, path_ref)

