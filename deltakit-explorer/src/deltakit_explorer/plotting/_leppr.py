from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundResults
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import interpolate_leppr


def plot_logical_error_probability_per_round(
    leppr_data: LogicalErrorProbabilityPerRoundResults,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: (
        npt.NDArray[np.float64] | Sequence[float] | None
    ) = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_data: Data class containing logical error probability per round
            fit results.
        num_rounds: a sequence of integers representing the number of rounds
            used to get the corresponding results in ``num_failed_shots`` and
            ``num_shots``.
        logical_error_probability: a sequence of floats representing the logical
            error probabilities corresponding to the number of rounds in
            ``num_rounds``.
        logical_error_probability_stddev: a sequence of floats representing the
            standard deviation of the logical error probabilities corresponding
            to the number of rounds in ``num_rounds``. If None, no error bars
            will be plotted. Default is None.
        num_sigmas: number of sigmas to consider when plotting error bars.
        fig: a matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: a matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:

        >>> from deltakit_explorer.analysis import (
        ...     calculate_lep_and_lep_stddev,
        ...     compute_logical_error_per_round,
        ... )
        >>> num_failed_shots = [34, 151, 356]
        >>> num_shots = [500000] * 3
        >>> num_rounds = [2, 4, 6]
        >>> res = compute_logical_error_per_round(
        ...     num_failed_shots=num_failed_shots,
        ...     num_shots=num_shots,
        ...     num_rounds=num_rounds,
        ... )
        >>> lep, lep_stddev = calculate_lep_and_lep_stddev(
        ...     fails=num_failed_shots, shots=num_shots
        ... )
        >>> fig, ax = plot_logical_error_probability_per_round(
        ...     res,
        ...     num_rounds=num_rounds,
        ...     logical_error_probability=lep,
        ...     logical_error_probability_stddev=lep_stddev,
        ... )
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    lens = {len(num_rounds), len(logical_error_probability)}
    if logical_error_probability_stddev is not None:
        lens.add(len(logical_error_probability_stddev))
    if len(lens) > 1:
        msg = (
            "The lengths of 'num_rounds', 'logical_error_probability' and "
            "'logical_error_probability_stddev' must be the same. Got the following "
            f"lengths: {lens}."
        )
        raise ValueError(msg)

    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    logical_error_probability = np.asarray(logical_error_probability)[isort]
    if logical_error_probability_stddev is not None:
        logical_error_probability_stddev = (
            num_sigmas * np.asarray(logical_error_probability_stddev)[isort]
        )

    # Plot the logical error probabilities
    ax.errorbar(
        num_rounds,
        logical_error_probability,
        yerr=logical_error_probability_stddev,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=f"Logical error probabilities (±{num_sigmas}σ)",  # noqa: RUF001
    )

    leppr_result = interpolate_leppr(leppr_data, num_rounds, num_sigmas=num_sigmas)

    plot(leppr_result, fig=fig, ax=ax)

    return fig, ax
