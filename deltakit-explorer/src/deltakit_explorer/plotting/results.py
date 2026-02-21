# (c) Copyright Riverlane 2020-2025.
"""Result types for plotting LEPPR and Lambda data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from deltakit_explorer.analysis._lambda import LambdaResults
from deltakit_explorer.analysis._leppr import LogicalErrorProbabilityPerRoundResults


@dataclass(frozen=True)
class LambdaPlot:
    """Result type holding the data needed to plot a Lambda fit.

    Attributes:
        distances: Interpolated distance grid for the fit curve.
        interpolated: Interpolated logical error probability per round values
            along the distance grid.
        lower_boundary: Lower boundary of the error band (computed from
            ``lambda - num_sigmas * lambda_stddev`` and
            ``lambda0 - num_sigmas * lambda0_stddev``).
        upper_boundary: Upper boundary of the error band (computed from
            ``lambda + num_sigmas * lambda_stddev`` and
            ``lambda0 + num_sigmas * lambda0_stddev``).
    """

    distances: npt.NDArray[np.floating]
    interpolated: npt.NDArray[np.floating]
    lower_boundary: npt.NDArray[np.floating]
    upper_boundary: npt.NDArray[np.floating]


@dataclass(frozen=True)
class LEPPRPlot:
    """Result type holding the data needed to plot a LEPPR fit.

    Attributes:
        rounds: Interpolated rounds grid for the fit curve.
        interpolated: Interpolated logical error probability values along the
            rounds grid.
        lower_boundary: Lower boundary of the error band (computed from
            ``leppr - num_sigmas * leppr_stddev`` and
            ``spam - num_sigmas * spam_stddev``).
        upper_boundary: Upper boundary of the error band (computed from
            ``leppr + num_sigmas * leppr_stddev`` and
            ``spam + num_sigmas * spam_stddev``).
    """

    rounds: npt.NDArray[np.floating]
    interpolated: npt.NDArray[np.floating]
    lower_boundary: npt.NDArray[np.floating]
    upper_boundary: npt.NDArray[np.floating]


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
    return lambda_**(-(distances + 1) / 2) / lambda0


def compute_lambda_plot(
    lambda_data: LambdaResults,
    distances: npt.NDArray[np.int_],
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
) -> LambdaPlot:
    """Compute the interpolated Lambda fit curve and its error band.

    Args:
        lambda_data (LambdaResults): Results from
            :func:`~deltakit_explorer.analysis.calculate_lambda_and_lambda_stddev`.
        distances (npt.NDArray[numpy.int\\_]): The code distances used for
            interpolation. The interpolated grid will span from ``min(distances)``
            to ``max(distances)``.
        num_sigmas (int): Number of standard deviations for the error band.
            Defaults to 3.
        num_points (int): Number of interpolation points. Defaults to 200.

    Returns:
        LambdaPlot: The interpolated fit data with error boundaries.
    """
    lambda_, lambda_stddev = lambda_data.lambda_, lambda_data.lambda_stddev
    lambda0, lambda0_stddev = lambda_data.lambda0, lambda_data.lambda0_stddev

    distances_interpolated = np.linspace(distances[0], distances[-1], num_points)
    interpolated = _lambda_interpolated(lambda0, lambda_, distances_interpolated)
    lower_boundary = _lambda_interpolated(
        lambda0 - num_sigmas * lambda0_stddev,
        lambda_ - num_sigmas * lambda_stddev,
        distances_interpolated,
    )
    upper_boundary = _lambda_interpolated(
        lambda0 + num_sigmas * lambda0_stddev,
        lambda_ + num_sigmas * lambda_stddev,
        distances_interpolated,
    )
    return LambdaPlot(
        distances=distances_interpolated,
        interpolated=interpolated,
        lower_boundary=lower_boundary,
        upper_boundary=upper_boundary,
    )


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


def compute_leppr_plot(
    leppr_data: LogicalErrorProbabilityPerRoundResults,
    num_rounds: npt.NDArray[np.int_],
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
) -> LEPPRPlot:
    """Compute the interpolated LEPPR fit curve and its error band.

    Args:
        leppr_data (LogicalErrorProbabilityPerRoundResults): Results from
            :func:`~deltakit_explorer.analysis.compute_logical_error_per_round`.
        num_rounds (npt.NDArray[numpy.int\\_]): The number of rounds used for
            interpolation. The interpolated grid will span from ``min(num_rounds)``
            to ``max(num_rounds)``.
        num_sigmas (int): Number of standard deviations for the error band.
            Defaults to 3.
        num_points (int): Number of interpolation points. Defaults to 200.

    Returns:
        LEPPRPlot: The interpolated fit data with error boundaries.
    """
    leppr, leppr_stddev = leppr_data.leppr, leppr_data.leppr_stddev
    spam, spam_stddev = leppr_data.spam_error, leppr_data.spam_error_stddev

    rounds_interpolated = np.linspace(num_rounds[0], num_rounds[-1], num_points)
    interpolated = _lep_interpolated(spam, leppr, rounds_interpolated)
    lower_boundary = _lep_interpolated(
        spam - num_sigmas * spam_stddev,
        leppr - num_sigmas * leppr_stddev,
        rounds_interpolated,
    )
    upper_boundary = _lep_interpolated(
        spam + num_sigmas * spam_stddev,
        leppr + num_sigmas * leppr_stddev,
        rounds_interpolated,
    )
    return LEPPRPlot(
        rounds=rounds_interpolated,
        interpolated=interpolated,
        lower_boundary=np.clip(lower_boundary, 0, 1),
        upper_boundary=np.clip(upper_boundary, 0, 1),
    )
