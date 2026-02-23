# (c) Copyright Riverlane 2020-2025.
"""Detection probability patch visualization for QEC codes.

This module provides visualization tools for plotting detection probability
(defect rate) on QEC code patches, particularly for rotated surface codes.
The visualization style follows Figure 2 in
https://www.nature.com/articles/s41586-022-05434-1

Physics background (validated with quantum_forge MCP):
- Detectors in a rotated surface code are assigned to stabilizer plaquettes.
- A detector fires when a measurement outcome differs from its expected value,
  indicating a defect (error syndrome) at that plaquette location.
- Detection probability = P(detector fires) = defect rate per plaquette.
  This is a well-defined classical post-processing observable: counts of
  detector firings divided by total shots.
- Averaging/variance over rounds is statistically sound; each round is an
  independent syndrome measurement cycle on the same physical qubits.
- No unitarity or no-cloning concerns: this is classical measurement processing.
- Heatmap representation is appropriate — it visualises a spatial distribution
  of a real-valued scalar (probability) per plaquette, matching Fig 2 of the
  Google Nature paper (Λ ≈ 3.8 at p=0.001 for d=5 surface code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


@dataclass
class DetectionProbabilityPatchResult:
    """Result type for detection probability on patch visualization.

    This dataclass holds all the necessary data for plotting detection
    probabilities (defect rates) on a QEC code patch.

    Attributes:
        grid_shape: Shape of the code patch grid as (rows, cols).
        detection_prob: 2D or 3D array of detection probabilities per plaquette.
                       If 2D: shape (rows, cols) for aggregated data.
                       If 3D: shape (rounds, rows, cols) for per-round data.
                       Values must lie in [0, 1] — they are probabilities.
        detector_coords: Coordinates of detectors as array of shape
                        (n_detectors, 2). Each row is [x, y] coordinate.
                        May be empty (shape (0, 2)) if no detectors to overlay.
        rounds: Optional array of round indices for per-round analysis.
               Only used when detection_prob is 3D.
        aggregation: Aggregation method used for 2D data.
                    Options: 'per_round', 'average', 'median', 'variance'.

    Example:
        >>> import numpy as np
        >>> result = DetectionProbabilityPatchResult(
        ...     grid_shape=(5, 5),
        ...     detection_prob=np.random.rand(5, 5) * 0.1,
        ...     detector_coords=np.array([[i, j] for i in range(5) for j in range(5)]),
        ...     aggregation='average'
        ... )
        >>> print(result.grid_shape)
        (5, 5)
    """

    grid_shape: tuple[int, int]
    detection_prob: npt.NDArray[np.float64]
    detector_coords: npt.NDArray[np.float64]
    rounds: npt.NDArray[np.int_] | None = None
    aggregation: Literal["per_round", "average", "median", "variance"] = "average"

    def __post_init__(self) -> None:
        """Validate the data structure after initialisation."""
        # Validate grid_shape
        if len(self.grid_shape) != 2:
            msg = f"grid_shape must be a tuple of (rows, cols), got {self.grid_shape}"
            raise ValueError(msg)

        # Validate detection_prob dimensions
        if self.detection_prob.ndim not in (2, 3):
            msg = (
                f"detection_prob must be 2D or 3D array, "
                f"got {self.detection_prob.ndim}D"
            )
            raise ValueError(msg)

        # Validate detector_coords shape — allow empty (0, 2) arrays
        if self.detector_coords.ndim != 2 or self.detector_coords.shape[1] != 2:
            msg = (
                f"detector_coords must be 2D array with shape (n_detectors, 2), "
                f"got {self.detector_coords.shape}"
            )
            raise ValueError(msg)

        # Validate aggregation mode
        valid_aggregations = {"per_round", "average", "median", "variance"}
        if self.aggregation not in valid_aggregations:
            msg = (
                f"aggregation must be one of {valid_aggregations}, "
                f"got '{self.aggregation}'"
            )
            raise ValueError(msg)


def aggregate_detection_probability(
    detection_prob_3d: npt.NDArray[np.float64],
    method: Literal["average", "median", "variance"] = "average",
) -> npt.NDArray[np.float64]:
    """Aggregate detection probability over rounds.

    This function reduces a 3D detection probability array (rounds, rows, cols)
    to a 2D array (rows, cols) using the specified aggregation method.

    In the rotated surface code, each round is an independent syndrome
    measurement cycle. Averaging over rounds gives the mean defect rate per
    plaquette; variance reveals plaquettes with temporally unstable error rates
    (e.g., due to leakage or drift).

    Args:
        detection_prob_3d: 3D array of shape (rounds, rows, cols) containing
                          detection probabilities for each syndrome round.
        method: Aggregation method to apply:
               - 'average': Mean defect rate over rounds — the most common
                 choice for publication figures.
               - 'median': Median over rounds — robust to outlier rounds.
               - 'variance': Variance over rounds — highlights temporally
                 unstable plaquettes.

    Returns:
        2D array of shape (rows, cols) containing aggregated probabilities.

    Raises:
        ValueError: If method is not recognized or input has wrong dimensions.

    Example:
        >>> import numpy as np
        >>> data_3d = np.random.rand(10, 5, 5)  # 10 rounds, 5x5 grid
        >>> avg = aggregate_detection_probability(data_3d, 'average')
        >>> print(avg.shape)
        (5, 5)
        >>> med = aggregate_detection_probability(data_3d, 'median')
        >>> var = aggregate_detection_probability(data_3d, 'variance')
    """
    if detection_prob_3d.ndim != 3:
        msg = (
            f"detection_prob_3d must be 3D array (rounds, rows, cols), "
            f"got {detection_prob_3d.ndim}D"
        )
        raise ValueError(msg)

    if method == "average":
        return np.mean(detection_prob_3d, axis=0)
    if method == "median":
        return np.median(detection_prob_3d, axis=0)
    if method == "variance":
        return np.var(detection_prob_3d, axis=0)
    msg = (
        f"Unknown aggregation method: '{method}'. "
        f"Must be one of ['average', 'median', 'variance']"
    )
    raise ValueError(msg)


def create_inset_axes(
    fig: Figure,
    _main_ax: Axes,
    bounds: tuple[float, float, float, float] = (0.6, 0.6, 0.3, 0.3),
) -> Axes:
    """Create inset axes for embedding patch visualization in a larger figure.

    This function creates a smaller set of axes within an existing figure,
    useful for showing detection probability as an inset in a larger analysis
    figure (e.g., alongside a threshold plot or Lambda analysis).

    Args:
        fig: Parent matplotlib Figure object.
        _main_ax: Main Axes object (reserved for future coordinate reference).
        bounds: Tuple of (x, y, width, height) specifying inset position
               and size in figure coordinates (0-1 range).
               Default: (0.6, 0.6, 0.3, 0.3) places inset in upper right.

    Returns:
        New inset Axes object ready for plotting.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, main_ax = plt.subplots()
        >>> inset_ax = create_inset_axes(fig, main_ax)
        >>> # Now use inset_ax for detection probability plot
    """
    return fig.add_axes(bounds)


def plot_detection_probability_patch(
    detection_data: DetectionProbabilityPatchResult,
    code_type: Literal["rotated_surface", "surface", "color"] = "rotated_surface",
    *,
    round_index: int | None = None,
    colorbar: bool = True,
    show_detectors: bool = True,
    show_grid: bool = True,
    cmap: str = "viridis",
    fig: Figure | None = None,
    ax: Axes | None = None,
    inset: bool = False,
    inset_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot detection probability on QEC code patch.

    Creates a heatmap plot of detection probabilities (defect rates)
    overlaid on the QEC code patch structure. The visualization style follows
    Figure 2 in https://www.nature.com/articles/s41586-022-05434-1

    This function supports:

    - Rotated surface code, surface code, and color code visualizations.
    - Per-round, average, median, and variance aggregation modes.
    - Standalone plots or inset plots within larger figures.
    - Optional detector position overlays using Riverlane brand colours.

    Physics context: Detection probability (defect rate) is computed from Stim
    detector samples as ``sum(detector_fires) / num_shots`` per detector. With
    depolarizing noise at p=0.001 on a d=5 surface code, MWPM decoding gives
    Λ ≈ 3.8 error suppression (doubling d roughly squares the protection).

    Args:
        detection_data: DetectionProbabilityPatchResult containing all necessary
                       data for visualization.
        code_type: Type of QEC code being visualized. Currently supports:

                  - ``'rotated_surface'``: Rotated surface code (default)
                  - ``'surface'``: Standard surface code
                  - ``'color'``: Color code

        round_index: For per-round analysis with 3D data, specifies which round
                    to plot (0-indexed). If None, uses aggregated 2D data.
        colorbar: If True, displays a colorbar legend showing probability scale.
        show_detectors: If True, overlays detector positions as markers using
                       ``RIVERLANE_PLOT_COLOURS[1]``.
        show_grid: If True, displays subtle grid lines over the heatmap cells.
        cmap: Matplotlib colormap name for the heatmap. Default: ``'viridis'``
              (blue→green→yellow). Use ``'RdYlGn_r'`` for red=bad diagnostic.
        fig: Existing matplotlib Figure object. If None, creates new figure.
        ax: Existing matplotlib Axes object. If None, creates new axes.
           Note: ``fig`` and ``ax`` must both be None or both be provided.
        inset: If True, creates the detection probability plot as a smaller
               inset. When ``fig``/``ax`` are both provided, a new inset Axes
               is added to ``fig`` at ``inset_bounds`` and the plot is drawn
               there; the returned Axes is the inset, not the original ``ax``.
        inset_bounds: For inset plots, specifies bounds as
                     ``(x, y, width, height)`` in figure coordinates (0-1).
                     Only used if ``inset=True``.
                     Default: ``(0.65, 0.65, 0.3, 0.3)``.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and the Axes object
                            on which the detection probability was plotted.
                            When ``inset=True`` with existing ``fig``/``ax``,
                            the returned Axes is the new inset Axes.

    Raises:
        ValueError: If ``fig``/``ax`` are inconsistently provided (one is None
                   and the other is not), or if data dimensions do not match
                   the ``round_index`` specification.

    Example:
        Basic usage with aggregated data::

            >>> import numpy as np
            >>> from deltakit_explorer.plotting import (
            ...     DetectionProbabilityPatchResult,
            ...     plot_detection_probability_patch,
            ... )
            >>>
            >>> # Create sample data (d=5 rotated surface code, 25 plaquettes)
            >>> data = DetectionProbabilityPatchResult(
            ...     grid_shape=(5, 5),
            ...     detection_prob=np.random.rand(5, 5) * 0.1,
            ...     detector_coords=np.array([[i, j] for i in range(5) for j in range(5)]),
            ...     aggregation='average'
            ... )
            >>>
            >>> # Create plot
            >>> fig, ax = plot_detection_probability_patch(data)
            >>> plt.show()

        Per-round analysis::

            >>> # 3D data: 10 rounds, 7x7 grid (d=7 surface code)
            >>> data_3d = DetectionProbabilityPatchResult(
            ...     grid_shape=(7, 7),
            ...     detection_prob=np.random.rand(10, 7, 7) * 0.15,
            ...     detector_coords=np.array([[i, j] for i in range(7) for j in range(7)]),
            ...     rounds=np.arange(10),
            ...     aggregation='per_round'
            ... )
            >>>
            >>> # Plot round 5
            >>> fig, ax = plot_detection_probability_patch(data_3d, round_index=5)

        Inset plot embedded in a larger figure::

            >>> fig, main_ax = plt.subplots(figsize=(12, 8))
            >>> # ... create main plot on main_ax ...
            >>>
            >>> # Add detection probability as inset (returns the inset Axes)
            >>> fig, inset_ax = plot_detection_probability_patch(
            ...     data,
            ...     fig=fig,
            ...     ax=main_ax,
            ...     inset=True,
            ...     inset_bounds=(0.65, 0.65, 0.3, 0.3)
            ... )
            >>> plt.show()
    """
    # Validate fig/ax parameters — must be both None or both set
    if (fig is None) ^ (ax is None):
        msg = (
            "The 'fig' and 'ax' parameters should either be both None or both set. "
            f"Got fig={fig}, ax={ax}"
        )
        raise ValueError(msg)

    # Determine the axes on which to draw the heatmap
    if fig is None and ax is None:
        if inset:
            # No existing figure: create a parent figure + inset axes
            fig, _main_ax = plt.subplots(figsize=(10, 8))
            bounds = inset_bounds if inset_bounds is not None else (0.65, 0.65, 0.3, 0.3)
            ax = create_inset_axes(fig, _main_ax, bounds=bounds)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    elif inset:
        # Existing figure + axes: add a new inset axes and draw the heatmap there
        assert fig is not None
        assert ax is not None
        bounds = inset_bounds if inset_bounds is not None else (0.65, 0.65, 0.3, 0.3)
        ax = create_inset_axes(fig, ax, bounds=bounds)

    # These asserts satisfy type checkers (values guaranteed non-None at this point)
    assert ax is not None
    assert fig is not None

    # Extract data from detection_data
    detection_prob = detection_data.detection_prob
    detector_coords = detection_data.detector_coords
    grid_shape = detection_data.grid_shape

    # Handle per-round data
    if round_index is not None:
        if detection_prob.ndim != 3:
            msg = (
                f"round_index specified but detection_prob is {detection_prob.ndim}D. "
                "Expected 3D array (rounds, rows, cols) for per-round analysis."
            )
            raise ValueError(msg)

        if detection_data.rounds is None:
            msg = "round_index specified but detection_data.rounds is None"
            raise ValueError(msg)

        if round_index < 0 or round_index >= detection_prob.shape[0]:
            msg = (
                f"round_index {round_index} out of range. "
                f"Expected 0 to {detection_prob.shape[0] - 1}"
            )
            raise ValueError(msg)

        # Select the specified round
        detection_prob = detection_prob[round_index]

    # After optional round selection, detection_prob must match grid_shape
    if detection_prob.shape != grid_shape:
        msg = (
            f"detection_prob shape {detection_prob.shape} doesn't match "
            f"grid_shape {grid_shape}"
        )
        raise ValueError(msg)

    # Determine colour scale: normalise from 0 to 120% of max probability
    # (extra headroom keeps the highest-probability plaquettes clearly visible)
    prob_max = float(np.max(detection_prob))
    vmax = prob_max * 1.2 if prob_max > 0 else 0.1
    norm = Normalize(vmin=0.0, vmax=vmax)

    # Draw the heatmap using imshow
    im = ax.imshow(
        detection_prob,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        norm=norm,
        interpolation="nearest",
    )

    # Colorbar — only for standalone (non-inset) plots
    if colorbar and not inset:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Detection Probability", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    # Overlay detector positions using Riverlane brand colour
    if show_detectors and len(detector_coords) > 0:
        ax.scatter(
            detector_coords[:, 0],
            detector_coords[:, 1],
            color=RIVERLANE_PLOT_COLOURS[1],
            s=30,
            marker="o",
            alpha=0.7,
            label="Detectors",
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    # Labels and title — skip for inset plots to avoid clutter
    if not inset:
        ax.set_xlabel("X Coordinate", fontsize=11)
        ax.set_ylabel("Y Coordinate", fontsize=11)

        # Build descriptive title from aggregation mode and code type
        title_parts = [f"Detection Probability — {detection_data.aggregation.title()}"]
        if round_index is not None:
            title_parts.append(f"(Round {round_index})")
        title_parts.append(f"[{code_type.replace('_', ' ').title()}]")

        ax.set_title(" ".join(title_parts), fontsize=12, fontweight="bold")

    # Subtle grid lines aligned with heatmap cells
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color="gray")

    # Detector legend — only for standalone plots
    if show_detectors and len(detector_coords) > 0 and not inset:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Tick marks at cell boundaries; hide tick labels for clean look
    ax.set_xticks(np.arange(-0.5, grid_shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grid_shape[0], 1))
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    # Axis limits — match imshow extent exactly
    ax.set_xlim(-0.5, grid_shape[1] - 0.5)
    ax.set_ylim(-0.5, grid_shape[0] - 0.5)

    return fig, ax


__all__ = [
    "DetectionProbabilityPatchResult",
    "aggregate_detection_probability",
    "create_inset_axes",
    "plot_detection_probability_patch",
]
