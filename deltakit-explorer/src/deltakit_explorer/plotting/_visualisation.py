# (c) Copyright Riverlane 2020-2025.
"""`visualisation` module aggregates data plotting methods.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from deltakit_explorer.types._types import QubitCoordinateToDetectorMapping


def correlation_matrix(
    matrix: npt.NDArray,
    qubit_to_detector_mapping: QubitCoordinateToDetectorMapping,
    labels: Sequence[str] = (),
):
    """Plot a given correlation matrix as a heatmap.

    Args:
        matrix (npt.NDArray): correlation matrix.
        qubit_to_detector_mapping (QubitCoordinateToDetectorMapping):
            {qubit_coordinate_tuple_1: [det_1, det_2, det_2, ...], }
        labels (Sequence[str]): labels to the qubits.

    Returns:
        matplotlib.plt:
            The plt object containing the drawn heatmap.

    Examples:

        Collect the data for a correlation matrix plot::

            matrix, mapping = client.get_correlation_matrix(
                detectors, stim_circuit,
                use_default_noise_model_edges=True,
            )
            plt = correlation_matrix(matrix, mapping)
            plt.show()

    """
    # create a list of indices of the minor ticks for which to label with
    # the qubit labels such that the labels are in the middle of the major
    # ticks. Sort the labels as they are not guaranteed to be in order.
    minor_ticks_in_major = len(
        next(iter(qubit_to_detector_mapping.detector_map.values())))
    num_major_ticks = len(qubit_to_detector_mapping.detector_map.keys())
    num_ticks = minor_ticks_in_major * num_major_ticks
    num_minor_ticks = num_ticks - num_major_ticks
    ticks_per_major = num_minor_ticks // num_major_ticks
    mid_im = ticks_per_major // 2
    label_indices = [mid_im + (ticks_per_major * i) for i in range(num_major_ticks)]
    sorted_labels = (
        sorted(qubit_to_detector_mapping.detector_map.keys())
        if len(labels) == 0 else labels
    )

    def format_func(_, tick_number):
        if tick_number in label_indices:
            return sorted_labels[label_indices.index(tick_number)]
        return None

    col1 = sns.cubehelix_palette(start=2.43, rot=-0.1, light=1.0, as_cmap=True)
    axes = sns.heatmap(matrix, cmap=col1)
    axes.invert_yaxis()
    axes.set(xlabel="Qubit coordinate", ylabel="Qubit coordinate")
    major_ticks = np.arange(0, len(matrix[0]), minor_ticks_in_major)
    minor_ticks = np.arange(0, len(matrix[0]), 1)
    axes.set_xticks(major_ticks)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(major_ticks)
    axes.set_yticks(minor_ticks, minor=True)
    axes.xaxis.set_minor_formatter(FuncFormatter(format_func))
    axes.yaxis.set_minor_formatter(FuncFormatter(format_func))
    axes.tick_params(axis="x", rotation=0)
    axes.grid(which="major", color="#333333", linestyle="-", alpha=0.6)
    axes.grid(which="minor", color="#AAAAAA", linestyle="--", alpha=0.2)
    return plt

def _rotate_defect_rate_points(
    detector_coords: dict,
    defect_rates: dict,
    transform_mat=np.array([[1, 1], [-1, 1]]) * (0.5**0.5),
) -> dict:
    # transform det coordinates to get upright plot
    for d_id, coord in detector_coords.items():
        detector_coords[d_id] = [*list(transform_mat @ coord[:-1]), coord[-1]]

    # transform defect rate coords
    x_offset = 0
    y_offset = 0
    rotated_coord_defect_rates = {}
    for coord, rates in defect_rates.items():
        new_coord = tuple((transform_mat @ coord).round())
        x_offset = min(x_offset, new_coord[0])
        y_offset = min(y_offset, new_coord[1])
        rotated_coord_defect_rates[new_coord] = np.mean(rates[1:-1])

    offset_defect_rates = {}
    for coord, defect_rate in rotated_coord_defect_rates.items():
        offset_defect_rates[
            (coord[0] - x_offset, coord[1] - y_offset)
        ] = defect_rate

    return offset_defect_rates

def defect_diagram(all_detector_coords: dict, all_defect_rates: dict):
    """Plots defect rates patch diagram given detector coordinates and
    their error rates.

    Args:
        all_detector_coords (Dict): Mapping from coordinates to detector numbers.
        all_defect_rates (Dict): Defect rates of detectors.

    Returns:
        matplotlib.pyplot: matplotlib module
    """
    # rotate coords
    defect_rates = _rotate_defect_rate_points(
        all_detector_coords, all_defect_rates
    )
    all_dr_means = defect_rates.values()
    cmap_min, cmap_max = min(all_dr_means), max(all_dr_means)

    all_coords = list(chain.from_iterable(defect_rates.keys()))

    # plot numbers in matrix
    matrix_width = max(all_coords)
    tpl: tuple[int, ...] = (int(matrix_width) + 1, int(matrix_width) + 1)
    defect_rate_mat = np.zeros(tpl, dtype=np.float64)
    for coord, val in defect_rates.items():
        defect_rate_mat[round(coord[0]), round(coord[1])] = val
    defect_rate_mat = np.array(defect_rate_mat).T

    # remove all-0 columns and rows
    defect_rate_mat = defect_rate_mat[~np.all(defect_rate_mat == 0, axis=1)]
    zero_column_idx = np.argwhere(np.all(defect_rate_mat[..., :] == 0, axis=0))
    defect_rate_mat = np.delete(defect_rate_mat, zero_column_idx, axis=1)

    # get coords to plot semicircles:
    # to plot these we need to find the weight-2 stabiliser values in the heatmap.
    # these will exist in the first and last rows and columns.
    # find their values + coordinates, and then shift them to accommodate
    # for deleting the first and last rows and columns, as we instead want to plot
    # semicircles in place of the squares.
    num_rows, num_cols = defect_rate_mat.shape
    last_col, first_col = defect_rate_mat[:, 0], defect_rate_mat[:, num_cols - 1]
    last_row, first_row = defect_rate_mat[0, :], defect_rate_mat[num_rows - 1, :]

    # create coordinates where the circles will be drawn, and keep a record
    # of the value so that the circles can be plotted with the correct colour
    top_sc_indices = [
        (num_rows - 1 - 1 - 0.5, x - 1, first_row[x])
        for x in np.where(first_row != 0)[0]
    ]
    bottom_sc_indices = [
        (-0.5, x - 1, last_row[x]) for x in np.where(last_row != 0)[0]
    ]
    left_sc_indices = [
        (num_cols - x - 1 - 1, -0.5, first_col[x])
        for x in np.where(first_col != 0)[0]
    ]
    right_sc_indices = [
        (num_cols - x - 1 - 1, num_cols - 1 - 1 - 0.5, last_col[x])
        for x in np.where(last_col != 0)[0]
    ]

    # delete rows/columns containing weight-2 stabilisers
    # to make room for the semicircles
    defect_rate_mat = np.delete(defect_rate_mat, (0, num_cols - 1), axis=1)
    defect_rate_mat = np.delete(defect_rate_mat, (0, num_rows - 1), axis=0)

    # begin plot
    fig, axes = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("viridis")

    # plot heatmap+colorbar, hide axis
    image = axes.imshow(defect_rate_mat, cmap=cmap, vmin=0.1, vmax=0.25, zorder=1)
    axes.set_yticks([])
    axes.set_xticks([])
    image.axes.invert_yaxis()
    cbar = axes.figure.colorbar(
        image, ax=axes, shrink=1, orientation="vertical",
        pad=0.2, label="Defect rate"
    )
    cbar.set_ticks([0.1, 0.25])

    # plot semicircles
    for circles in (
        top_sc_indices,
        bottom_sc_indices,
        left_sc_indices,
        right_sc_indices,
    ):
        for circle_coord in circles:
            rgba = cmap((circle_coord[2] - cmap_min) / cmap_max)
            circle = plt.Circle(
                (circle_coord[1], circle_coord[0]),
                0.5,
                color=rgba,
                clip_on=False,
                zorder=0,
            )
            axes.add_patch(circle)

    # plot
    fig.tight_layout()
    return plt


def defect_rates(
    defect_rates_series: Iterable[dict[tuple[float, ...], list[float]]],
    w2_det_coords: Collection[tuple[float, ...]],
):
    """
    Convenience function to plot the average defect rate plot fig 2b
    in Google paper https://www.nature.com/articles/s41586-022-05434-1.

    Args:
        defect_rates_series (Iterable[Dict[Tuple[float, ...], List[float]]]):
            List of defect rates dictionaries.
            E.g, this can be for the X and Z experiments
            for the Google data set.
        w2_det_coords (Container[Tuple[float, ...]]):
            Coordinates for the weight 2 detectors, so
            that these may be plotted with a separate colour
            and have their average separate from the higher-weight
            stabilisers.

    Returns:
        matplotlib.pyplot: pyplot module

    Examples:
        Plotting the Google example::

            z_data_folder = data_folder / "surface_code_bZ_d3_r07_center_3_5/"
            x_data_folder = data_folder / "surface_code_bX_d3_r07_center_3_5/"
            z_and_x_experiment_folders = [z_data_folder, x_data_folder]

            experiments = [
                QECExperiment.from_circuit_and_measurements(
                    folder / "circuit_noisy.stim",
                    folder / "measurements.b8", DataFormat.B8,
                    folder / "sweep.b8", DataFormat.B8,
                )
                for folder
                in z_and_x_experiment_folders
            ]
            all_rates = []
            for experiment in experiments:
                _, rates = client.get_experiment_detectors_and_defect_rates(
                    experiment
                )
                all_rates.append(rates)
            defect_rates(
                all_rates,
                w2_det_coords=set({(5., 6.), (1., 4.), (4., 3.), (2., 7.)})
            )
    """
    # ensure these are floats!
    w2_det_coords = {tuple(map(float, item)) for item in w2_det_coords}
    w2_avg = []
    w4_avg = []
    for all_qubit_defect_rates in defect_rates_series:
        for coord, defect_rate in all_qubit_defect_rates.items():
            if coord in w2_det_coords:
                w2_avg.append(defect_rate)
                plt.plot(
                    range(1, len(defect_rate) + 1),
                    defect_rate, color="#ff7500", alpha=0.3
                )
            else:
                w4_avg.append(defect_rate)
                plt.plot(
                    range(1, len(defect_rate) + 1),
                    defect_rate, color="#006f62", alpha=0.3
                )
    w2_detectors = np.mean(w2_avg if w2_avg else [[]], axis=0)
    w4_detectors = np.mean(w4_avg if w4_avg else [[]], axis=0)
    plt.plot(
        range(1, len(w4_detectors) + 1),
        w4_detectors,
        color="#006f62", label="Weight-4")
    plt.plot(
        range(1, len(w2_detectors) + 1),
        w2_detectors,
        color="#ff7500", label="Weight-2")
    plt.xlabel("Round")
    plt.xticks(range(1, len(w4_detectors) + 1))
    plt.ylabel("Defect rate")
    plt.legend(loc="lower right", frameon=False)
    return plt
