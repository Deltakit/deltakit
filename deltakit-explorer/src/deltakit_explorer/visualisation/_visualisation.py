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
from deltakit_explorer.types._types import QubitCoordinateToDetectorMapping
from matplotlib.ticker import FuncFormatter
import warnings


from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_core.constants import RIVERLANE_PLOT_COLOURS
from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundResults, calculate_lambda_and_lambda_stddev, get_lambda_fit

def plot_logical_error_probability_per_round(
    leppr_data: LogicalErrorProbabilityPerRoundResults,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: npt.NDArray[np.float64] | Sequence[float] | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_data (LogicalErrorProbabilityPerRoundResults):
            Data class containing logical error probability per round fit results.
        num_rounds (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``.
        logical_error_probability (npt.NDArray[np.float64] | Sequence[float]):
            a sequence of floats representing the logical error probabilities
            corresponding to the number of rounds in ``num_rounds``.
        logical_error_probability_stddev (npt.NDArray[np.float64] | Sequence[float] | None, optional):
            a sequence of floats representing the standard deviation of the logical
            error probabilities corresponding to the number of rounds in ``num_rounds``.
            If None, no error bars will be plotted. Default is None.
        fig (Figure | None, optional):
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax (Axes | None, optional):
            a matplotlib Axes object to plot on. If None, a new axes will be created.
            Default is None.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example
        from deltakit_explorer.analysis import calculate_lep_and_lep_stddev, compute_logical_error_per_round
        num_failed_shots=[34, 151, 356]
        num_shots=[500000] * 3
        num_rounds=[2, 4, 6]

        res = compute_logical_error_per_round(
                    num_failed_shots=num_failed_shots,
                    num_shots=num_shots,
                    num_rounds=num_rounds,
                )

        # plot logical error probabilities
        lep, lep_stddev = calculate_lep_and_lep_stddev(fails=num_failed_shots, shots=num_shots)
        fig, ax = plot_logical_error_probability_per_round(
            res,
            num_rounds=num_rounds,
            logical_error_probability=lep,
            logical_error_probability_stddev=lep_stddev,
            )
        plt.show()
    """
    if (fig is None) ^ (ax is None):
        raise ValueError("The 'fig' and 'ax' parameters should either both be set or unset. Got only one set, which is not handled.")
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if not len(num_rounds) == len(logical_error_probability) == len(logical_error_probability_stddev):
        raise ValueError(
            "The lengths of 'num_rounds', 'logical_error_probability' and "
            "'logical_error_probability_stddev' must be the same. "
            f"Got lengths {len(num_rounds)}, {len(logical_error_probability)}, "
            f"and {len(logical_error_probability_stddev)} respectively."
        )

    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    logical_error_probability = np.asarray(logical_error_probability)[isort]
    logical_error_probability_stddev = np.asarray(logical_error_probability_stddev)[isort]
    while num_rounds[0] <= 0:
        warnings.warn(
            f"Found an invalid number of rounds: {num_rounds[0]}. Number of rounds "
            "should be >= 1."
        )
        num_rounds = num_rounds[1:]
        logical_error_probability = logical_error_probability[1:]
        logical_error_probability_stddev = logical_error_probability_stddev[1:]

    if np.any(logical_error_probability <= 0) or np.any(logical_error_probability >= 1):
        raise RuntimeError(
            "Found an invalid logical error probability. Probabilities must be between 0 and 1"
            "Logical error probabilities: "
            f"{logical_error_probability}."
        )

    leppr, leppr_stddev = leppr_data.leppr, leppr_data.leppr_stddev
    spam, spam_stddev = leppr_data.spam_error, leppr_data.spam_error_stddev

    if leppr < 0 or spam < 0 or leppr >= 0.5 or spam >= 0.5:
        warnings.warn(
            "LEPPR or SPAM error is not within [0, 0.5)."
            f"LEPPR: {leppr}, SPAM error: {spam}."
        )
    if leppr_stddev < 0 or spam_stddev < 0:
        raise RuntimeError(
            "LEPPR or SPAM error standard deviation is negative. Standard deviations must be non-negative."
            f"LEPPR stddev: {leppr_stddev}, SPAM error stddev: {spam_stddev}."
        )

    # plot logical error probabilities
    ax.errorbar(num_rounds, logical_error_probability, yerr=logical_error_probability_stddev, fmt="o", color=RIVERLANE_PLOT_COLOURS[0], label = "Logical error probabilities")
    # plot fitted LEPPR curve
    interpolation_points = 200
    rounds_interpolated = np.linspace(
        num_rounds[0], num_rounds[-1], interpolation_points,
        dtype=np.float64,
    )
    y_interpolated = (1 - 2 * spam) * (1 - 2 * leppr) ** rounds_interpolated

    lep_interpolated = (1 - y_interpolated) /2
    ax.plot(rounds_interpolated, lep_interpolated,label=f"Fit, ε={leppr:.4f}" + r"$\pm$" + f"{leppr_stddev:.4f}", color=RIVERLANE_PLOT_COLOURS[0])

    # add error band to LEPPR curve
    lep_interpolated_err = np.array([np.sqrt(((2*spam_stddev)/(1-2*spam_stddev))**2 + ((2*r*leppr_stddev)/(1-2*leppr_stddev))**2) for r in rounds_interpolated])
    upper_error = lep_interpolated + lep_interpolated_err
    lower_error = lep_interpolated - lep_interpolated_err
    lower_error = np.clip(lower_error, 0, 1)  # ensure no negative values for log scale
    ax.fill_between(rounds_interpolated, lower_error, upper_error, color=RIVERLANE_PLOT_COLOURS[0], alpha=0.2)

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()

    return fig, ax

# next step: plot lambda
def plot_lambda(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float],
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Args:
        distances (npt.NDArray[np.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_stddev_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The standard deviation of the logical error probabilities per round.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example:
        fig, ax = plot_lambda(
            distances = [5, 7, 9],
            lep_per_round = [0.15, 0.1, 0.05],
            lep_stddev_per_round = [0.01, 0.008, 0.005],
        )
        ax.set_yscale("log")
        plt.show()
    """
    if (fig is None) ^ (ax is None):
        raise ValueError("The 'fig' and 'ax' parameters should either both be set or unset. Got only one set, which is not handled.")
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    res = calculate_lambda_and_lambda_stddev(
        distances=distances,
        lep_per_round=lep_per_round,
        lep_stddev_per_round=lep_stddev_per_round,
    )
    lambda_val, lambda_val_stddev = res.lambda_, res.lambda_stddev
    ax.errorbar(distances, lep_per_round, yerr=lep_stddev_per_round, fmt="o", color=RIVERLANE_PLOT_COLOURS[0])

    y_vals = get_lambda_fit(distances, lep_per_round, lep_stddev_per_round)
    ax.plot(distances,y_vals,label=f"Fit, λ={lambda_val:.4f}" + r"$\pm$" + f"{lambda_val_stddev:.4f}", color=RIVERLANE_PLOT_COLOURS[0])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Logical error probability per round")
    ax.legend()
    return fig, ax

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
    w2_detectors = np.mean(w2_avg, axis=0)
    w4_detectors = np.mean(w4_avg, axis=0)
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
    plt.ylim(0, 0.25)
    plt.ylabel("Defect rate")
    plt.legend(loc="lower right", frameon=False)
    return plt
