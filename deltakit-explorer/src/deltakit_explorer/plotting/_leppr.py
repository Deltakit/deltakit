# Copyright Riverlane 2020-2025.
"""Plot logical error probabilities vs rounds with optional LEPPR fit.

This module provides a thin, styling-focused wrapper around matplotlib for
visualising logical error probabilities as a function of the number of
rounds. It intentionally does **not** compute LEPPR itself; callers are
expected to pass any pre-computed LEPPR-related quantities explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy
import numpy.typing as npt
from matplotlib.axes import Axes

from deltakit_explorer.plotting._plot_style import (
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
    apply_publication_style,
)


def plot_leppr(
    num_rounds: npt.NDArray[numpy.int_] | Sequence[int],
    logical_error_probabilities: npt.NDArray[numpy.floating] | Sequence[float],
    logical_error_probabilities_stddev: (
        npt.NDArray[numpy.floating] | Sequence[float] | None
    ) = None,
    *,
    leppr: float | None = None,
    leppr_stddev: float | None = None,
    spam_error: float | None = None,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot logical error probability vs number of rounds.

    The function expects **pre-computed** logical error probabilities and,
    optionally, a pre-computed logical error probability per round (LEPPR)
    and SPAM error. It does **not** perform any analysis itself, which allows
    callers to use any workflow for computing LEPPR and just rely on this
    helper for styling and layout.

    If ``leppr`` and ``spam_error`` are provided, an exponential fit of the
    form

    .. math::

        p_\\mathrm{L}(r) = \\frac{1 - (1 - 2 p_\\mathrm{SPAM}) (1 - 2 \\epsilon)^r}{2}

    is overlaid using the given parameters :math:`\\epsilon` (``leppr``) and
    :math:`p_\\mathrm{SPAM}` (``spam_error``).

    Args:
        num_rounds: Number of rounds corresponding to each data point.
        logical_error_probabilities: Logical error probability at each
            number of rounds in ``num_rounds``.
        logical_error_probabilities_stddev: Standard deviation of each
            logical error probability value. If ``None``, no error bars are
            drawn.
        leppr: Pre-computed logical error probability per round
            :math:`\\epsilon`. This can be obtained for example with
            :func:`deltakit_explorer.analysis.compute_logical_error_per_round`,
            or with any other analysis code.
        leppr_stddev: Standard deviation of ``leppr``. Used only for
            labelling the fit in the legend.
        spam_error: Pre-computed SPAM error probability used in the fit
            formula above. If ``None``, no fitted curve is drawn.
        ax: Axes to draw on. If ``None``, a new figure and axes are created.
        label: Optional label for the data series (used in the legend).

    Returns:
        The Axes instance used for plotting (for further customization).

    Examples:
        Plot LEP vs rounds with externally-computed LEPPR::

            import numpy as np
            from deltakit_explorer.analysis import (
                calculate_lep_and_lep_stddev,
                compute_logical_error_per_round,
            )
            from deltakit_explorer.plotting import plot_leppr

            num_rounds = [2, 4, 6, 8]
            fails, shots = [34, 151, 356, 512], [500000] * 4
            lep, lep_std = calculate_lep_and_lep_stddev(fails, shots)
            res = compute_logical_error_per_round(
                num_rounds, lep, lep_std, force_include_single_round=True
            )
            ax = plot_leppr(
                num_rounds,
                lep,
                lep_std,
                leppr=res.leppr,
                leppr_stddev=res.leppr_stddev,
                spam_error=res.spam_error,
            )
            ax.figure.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    num_rounds_arr = numpy.asarray(num_rounds)
    lep_arr = numpy.asarray(logical_error_probabilities)
    lep_std_arr = (
        None
        if logical_error_probabilities_stddev is None
        else numpy.asarray(logical_error_probabilities_stddev)
    )

    data_label = label if label is not None else "Data"
    if lep_std_arr is not None:
        ax.errorbar(
            num_rounds_arr,
            lep_arr,
            yerr=lep_std_arr,
            fmt="o",
            linestyle="-",
            capsize=3,
            label=data_label,
        )
    else:
        ax.plot(
            num_rounds_arr,
            lep_arr,
            marker="o",
            linestyle="-",
            label=data_label,
        )

    # Optional fitted curve if LEPPR and SPAM error are provided.
    if leppr is not None and spam_error is not None:
        r_min, r_max = int(num_rounds_arr.min()), int(num_rounds_arr.max())
        r_smooth = numpy.linspace(r_min, r_max, max(50, (r_max - r_min) * 5))
        fidelity = (1 - 2 * spam_error) * (1 - 2 * leppr) ** r_smooth
        lep_fit = (1 - fidelity) / 2

        if leppr_stddev is not None:
            fit_label = f"Fit, ε={leppr:.4f} ± {leppr_stddev:.4f}"
        else:
            fit_label = f"Fit, ε={leppr:.4f}"

        ax.plot(r_smooth, lep_fit, linestyle="--", label=fit_label)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Rounds", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Logical Error Probability", fontsize=FONTSIZE_LABEL)
    ax.legend(loc="best", fontsize=FONTSIZE_LEGEND, framealpha=0.9)

    apply_publication_style(ax)
    return ax

