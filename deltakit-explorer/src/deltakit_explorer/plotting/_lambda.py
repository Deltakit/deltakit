from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis._lambda import LambdaResults


def _lambda_interpolated(
    lambda0: float, lambda_: float, distances: npt.NDArray[np.int_]
) -> npt.NDArray[np.floating]:
    """Computes logical error probability per round that would be obtained with the
    provided values.

    Uses the formula ``ε = 1 / Λ_0 * Λ**(-(d + 1) / 2)`` where:

    - ``ε`` is the logical error probability per round,
    - ``Λ_0`` is a multiplicative constant,
    - ``Λ`` is the error suppression factor,
    - ``d`` is the distance of the code,

    to estimate the logical error probability per round from the provided ``lambda_``
    and ``lambda0`` on the provided list of ``distances``.
    """
    return lambda_ ** (-(distances + 1) / 2) / lambda0


def plot_lambda(
    lambda_data: LambdaResults,
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_stddev: npt.NDArray[np.float64] | Sequence[float] | None = None,
    lep_per_round_high: npt.NDArray[np.float64] | Sequence[float] | None = None,
    lep_per_round_low: npt.NDArray[np.float64] | Sequence[float] | None = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot Λ-fitting data.

    This function plots both the logical error-probability per round that has been used
    to compute Λ, the associated error-rates if provided, and the resulting fit, showing
    how close the fit is from actual data.

    Args:
        distances (npt.NDArray[numpy.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[numpy.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_per_round_stddev (npt.NDArray[numpy.float64] | Sequence[float] | None):
            The standard deviation of the logical error probabilities per round. When
            provided, the data-point error bars span ``num_sigmas`` standard deviations.
            Ignored when ``lep_per_round_high``/``lep_per_round_low`` are provided.
        lep_per_round_high (npt.NDArray[numpy.float64] | Sequence[float] | None):
            Upper asymmetric error *magnitude* on each data point (the upper bound is
            ``lep_per_round + lep_per_round_high``). When provided together with
            ``lep_per_round_low``, these are used for the error bars instead of
            ``lep_per_round_stddev``. This matches the convention of the
            ``leppr_error_low``/``leppr_error_high`` returned by
            :func:`compute_logical_error_per_round`.
        lep_per_round_low (npt.NDArray[numpy.float64] | Sequence[float] | None):
            Lower asymmetric error *magnitude* on each data point (the lower bound is
            ``lep_per_round - lep_per_round_low``). See ``lep_per_round_high``.
        num_sigmas (int): number of sigmas used for the data-point error bars (when based
            on ``lep_per_round_stddev``) and the Λ_0 contribution to the fit band. The Λ
            contribution to the band uses the asymmetric errors stored in ``lambda_data``,
            which should have been computed with the same ``num_sigmas``.

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
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # These should be already checked by the above code, but type checkers are not able
    # to infer that information, so including the asserts explicitly for type checkers
    # to understand.
    assert ax is not None
    assert fig is not None

    lengths = {len(distances), len(lep_per_round)}
    if lep_per_round_stddev is not None:
        lengths.add(len(lep_per_round_stddev))
    if len(lengths) > 1:
        msg = (
            "The lengths of 'distances', 'lep_per_round' and 'lep_per_round_stddev' "
            f"must be the same. Got the following lengths: {lengths}."
        )
        raise ValueError(msg)

    isort = np.argsort(distances)
    distances = np.asarray(distances)[isort]
    lep_per_round = np.asarray(lep_per_round)[isort]
    # Prefer the asymmetric high/low errors when provided; otherwise fall back to the
    # symmetric standard deviation, and to no error bars if neither is given.
    # ``lep_per_round_low``/``_high`` are error *magnitudes* (offsets relative to the
    # point estimate), matching the convention of the ``leppr_error_low``/
    # ``leppr_error_high`` returned by :func:`compute_logical_error_per_round` and what
    # matplotlib's ``errorbar`` expects for ``yerr=[lower, upper]``.
    if lep_per_round_high is not None and lep_per_round_low is not None:
        yerr: npt.NDArray[np.floating] | None = num_sigmas * np.array([
            np.asarray(lep_per_round_low)[isort],
            np.asarray(lep_per_round_high)[isort],
        ])
    elif lep_per_round_stddev is not None:
        yerr = num_sigmas * np.asarray(lep_per_round_stddev)[isort]
    else:
        yerr = None
    # Plot the logical error probabilities per round
    ax.errorbar(
        distances,
        lep_per_round,
        yerr=yerr,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[1],
        label=f"Logical error probabilities per round (±{num_sigmas}σ)",  # noqa: RUF001
    )

    # Plot the fitted lambda curve
    lambda_ = lambda_data.lambda_
    lambda0, lambda0_stddev = lambda_data.lambda0, lambda_data.lambda0_stddev
    lambda_err_high = lambda_data.lambda_error_high
    lambda_err_low = lambda_data.lambda_error_low
    distances_interpolated = np.linspace(distances[0], distances[-1], 200)
    lambda_interpolated = _lambda_interpolated(lambda0, lambda_, distances_interpolated)
    ax.plot(
        distances_interpolated,
        lambda_interpolated,
        label=(  # noqa: RUF001
            f"Fit, Λ={lambda_:.4f} "
            f"+{lambda_err_high:.4f}/-{lambda_err_low:.4f} "
            f"({num_sigmas}σ)"
        ),
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    # Add the error band using the asymmetric CI from lambda_data. The lambda high/low
    # errors are already the num_sigmas-level interval, so they are used directly; Λ_0
    # only has a symmetric standard deviation, so that one is scaled by num_sigmas here.
    # Higher Λ and higher Λ_0 both reduce lep → lower band; lower values → upper band.
    lambda_interpolated_low = _lambda_interpolated(
        lambda0 + num_sigmas * lambda0_stddev,
        lambda_ + lambda_err_high,
        distances_interpolated,
    )
    lambda_interpolated_high = _lambda_interpolated(
        lambda0 - num_sigmas * lambda0_stddev,
        lambda_ - lambda_err_low,
        distances_interpolated,
    )
    ax.fill_between(
        distances_interpolated,
        lambda_interpolated_low,
        lambda_interpolated_high,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )

    ax.set_title("Logical Error Probability Per Round Fit")
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Error suppression factor Λ")
    ax.legend()
    return fig, ax
