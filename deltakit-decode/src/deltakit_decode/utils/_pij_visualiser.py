# (c) Copyright Riverlane 2020-2025.
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def plot_correlation_matrix(
    matrix: list[list[float]],
    major_minor_mapping: dict[tuple[float, ...], list[int]],
    labels: Sequence[str] = (),
):
    """Plot a given correlation matrix as a heatmap.

    Parameters
    ----------
    matrix : List[List[float]]
        A correlation matrix generated from Pij data.
    major_minor_mapping : Dict[Tuple[int, ...], List[int]]
        The accompanying coordinate mapping for a correlation matrix.
    labels : Sequence[str]
        Optional list of labels to assign to the qubits in-order. If unspecified,
        will use the qubit's coordinates instead.
        By default, ().

    Returns
    -------
    matplotlib.plt
        The plt object containing the drawn heatmap.
    """
    try:
        import seaborn as sns  # noqa: PLC0415
    except ImportError as ie:
        msg = "Seaborn is not installed - please install Visualisation extras"
        raise ImportError(msg) from ie

    # create a list of indices of the minor ticks for which to label with
    # the qubit labels such that the labels are in the middle of the major
    # ticks. Sort the labels as they are not guaranteed to be in order.
    minor_ticks_in_major = len(next(iter(major_minor_mapping.values())))
    num_major_ticks = len(major_minor_mapping.keys())
    num_ticks = minor_ticks_in_major * num_major_ticks
    num_minor_ticks = num_ticks - num_major_ticks
    im = num_minor_ticks // num_major_ticks
    mid_im = im // 2
    label_indices = [mid_im + (im * i) for i in range(num_major_ticks)]
    sorted_labels = sorted(major_minor_mapping.keys()) if len(labels) == 0 else labels

    def format_func(_, tick_number):
        if tick_number in label_indices:
            return sorted_labels[label_indices.index(tick_number)]
        return None

    col1 = sns.cubehelix_palette(start=.2, rot=-.3, light=1.0, as_cmap=True)
    ax = sns.heatmap(matrix, cmap=col1)
    ax.invert_yaxis()
    ax.set(xlabel="Qubits", ylabel="Qubits")
    major_ticks = np.arange(0, len(matrix[0]), minor_ticks_in_major)
    minor_ticks = np.arange(0, len(matrix[0]), 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.xaxis.set_minor_formatter(FuncFormatter(format_func))
    ax.yaxis.set_minor_formatter(FuncFormatter(format_func))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(which="major", color="#333333", linestyle="-", alpha=0.6)
    ax.grid(which="minor", color="#AAAAAA", linestyle="--", alpha=0.2)
    return plt
