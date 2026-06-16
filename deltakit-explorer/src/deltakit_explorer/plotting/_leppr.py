from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundResults


def _lep_interpolated(
    spam: float, leppr: float, rounds_interpolated: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Computes logical error that would be obtained with the provided values.

    Uses the formula ``F = Fs * Fε**r`` where:

    - ``F`` is the expected fidelity of the computation,
    - ``Fs`` is the fidelity of SPAM-related operations,
    - ``Fε`` is the fidelity of one quantum error-correction round,
    - ``r`` is the number of quantum error-correction rounds performed.

    Each fidelity is obtained from the respective error probability with the formula
    ``f = (1 - 2 * e)`` where ``f`` is any of ``F``, ``Fs`` or ``Fε`` and ``e`` is any
    of logical error probability, logical error probability of a SPAM or logical error
    probability per round.
    """
    expected_fidelity = (1 - 2 * spam) * (1 - 2 * leppr) ** rounds_interpolated
    return (1 - expected_fidelity) / 2


def plot_logical_error_probability_per_round(
    leppr_data: LogicalErrorProbabilityPerRoundResults,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: npt.NDArray[np.float64] | Sequence[float] | None = None,
    logical_error_probability_high: npt.NDArray[np.float64] | Sequence[float] | None = None,
    logical_error_probability_low: npt.NDArray[np.float64] | Sequence[float] | None = None,
    num_sigmas: int = 1,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_data (LogicalErrorProbabilityPerRoundResults):
            Data class containing logical error probability per round fit results.
        num_rounds (npt.NDArray[numpy.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``.
        logical_error_probability (npt.NDArray[numpy.float64] | Sequence[float]):
            a sequence of floats representing the logical error probabilities
            corresponding to the number of rounds in ``num_rounds``.
        logical_error_probability_stddev (npt.NDArray[numpy.float64] | Sequence[float] | None, optional):
            a sequence of floats representing the standard deviation of the logical
            error probabilities corresponding to the number of rounds in ``num_rounds``.
            If None, no error bars will be plotted on data points (unless asymmetric
            errors are provided). Default is None.
        logical_error_probability_high (npt.NDArray[numpy.float64] | Sequence[float] | None, optional):
            upper asymmetric 1-sigma error *magnitude* on each logical error probability
            data point (the upper bound is ``logical_error_probability +
            logical_error_probability_high``). When provided together with
            ``logical_error_probability_low``, these are used instead of
            ``logical_error_probability_stddev`` for the error bars. Default is None.
        logical_error_probability_low (npt.NDArray[numpy.float64] | Sequence[float] | None, optional):
            lower asymmetric 1-sigma error *magnitude* on each logical error probability
            data point (the lower bound is ``logical_error_probability -
            logical_error_probability_low``). When provided together with
            ``logical_error_probability_high``, these are used instead of
            ``logical_error_probability_stddev`` for the error bars. Default is None.
        num_sigmas (int): number of sigmas to consider when plotting error bars.
        fig (Figure | None, optional):
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax (Axes | None, optional):
            a matplotlib Axes object to plot on. If None, a new axes will be created.
            Default is None.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example:

        >>> from deltakit_explorer.analysis import (
        ...     calculate_lep_and_lep_stddev, compute_logical_error_per_round,
        ... )
        >>> num_failed_shots=[34, 151, 356]
        >>> num_shots=[500000] * 3
        >>> num_rounds=[2, 4, 6]
        >>> lep, lep_error_low, lep_error_high = calculate_lep_and_lep_stddev(
        ...     fails=num_failed_shots, shots=num_shots
        ... )
        >>> res = compute_logical_error_per_round(
        ...     num_rounds=num_rounds,
        ...     logical_error_probabilities=lep,
        ...     logical_error_probabilities_low=lep_error_low,
        ...     logical_error_probabilities_high=lep_error_high,
        ... )
        >>> fig, ax = plot_logical_error_probability_per_round(
        ...     res,
        ...     num_rounds=num_rounds,
        ...     logical_error_probability=lep,
        ...     logical_error_probability_low=lep_error_low,
        ...     logical_error_probability_high=lep_error_high,
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
    # Prefer the asymmetric high/low errors when provided; otherwise fall back to the
    # symmetric standard deviation, and to no error bars if neither is given.
    # ``logical_error_probability_low``/``_high`` are error *magnitudes* (offsets relative
    # to the point estimate), matching the convention of ``calculate_lep_and_lep_stddev``
    # and what matplotlib's ``errorbar`` expects for ``yerr=[lower, upper]``.
    if (
        logical_error_probability_high is not None
        and logical_error_probability_low is not None
    ):
        yerr: npt.NDArray[np.floating] | None = np.array([
            num_sigmas*np.asarray(logical_error_probability_low)[isort],
            num_sigmas*np.asarray(logical_error_probability_high)[isort],
        ])
    elif logical_error_probability_stddev is not None:
        yerr =   np.asarray(logical_error_probability_stddev)[isort]
    else:
        yerr = None
    # Plot the logical error probabilities
    ax.errorbar(
        num_rounds,
        logical_error_probability,
        yerr=yerr,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=f"Logical error probabilities (±{num_sigmas}σ)",  # noqa: RUF001
    )

    # Plot the fitted logical error probability per round curve
    leppr = leppr_data.leppr
    leppr_err_high = leppr_data.leppr_error_high
    leppr_err_low = leppr_data.leppr_error_low
    spam = leppr_data.spam_error
    spam_err_high = leppr_data.spam_error_high
    spam_err_low = leppr_data.spam_error_low

    rounds_interpolated = np.linspace(num_rounds[0], num_rounds[-1], 200)
    lep_interpolated = _lep_interpolated(spam, leppr, rounds_interpolated)
    ax.plot(
        rounds_interpolated,
        lep_interpolated,
        label=(  # noqa: RUF001
            f"Fit, ε={leppr:.4f} "
            f"+{leppr_err_high:.4f}/-{leppr_err_low:.4f} "
            f"({num_sigmas}σ)"
        ),
        color=RIVERLANE_PLOT_COLOURS[1],
    )

    # Add the asymmetric error band to the fitted curve. The high/low errors carried by
    # ``leppr_data`` are already the num_sigmas-level confidence interval, so they are
    # used directly (no extra num_sigmas scaling here).
    # Higher leppr + higher spam → upper band; lower leppr + lower spam → lower band.
    lep_interpolated_high = _lep_interpolated(
        spam + spam_err_high,
        leppr + leppr_err_high,
        rounds_interpolated,
    )
    lep_interpolated_low = _lep_interpolated(
        spam - spam_err_low,
        leppr - leppr_err_low,
        rounds_interpolated,
    )
    ax.fill_between(
        rounds_interpolated,
        np.clip(lep_interpolated_low, 0, 1),
        np.clip(lep_interpolated_high, 0, 1),
        color=RIVERLANE_PLOT_COLOURS[1],
        alpha=0.2,
        label=f"±{num_sigmas}σ band",  # noqa: RUF001
    )

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()

    return fig, ax
