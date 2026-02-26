# (c) Copyright Riverlane 2020-2025.
"""Result types for plotting LEPPR and Lambda data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from deltakit_explorer.analysis._lambda import LambdaResults as LambdaData
from deltakit_explorer.analysis._leppr import LogicalErrorProbabilityPerRoundResults


def _lambda_interpolated(
    lambda0: float, lambda_: float, distances: npt.NDArray[np.int_ | np.floating]  # lambda_ avoids shadowing the built-in `lambda` keyword
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


@dataclass(frozen=True)
class Interpolated:
    """Base class for interpolated plotting data."""

    interpolated: npt.NDArray[np.floating]
    lower_boundary: npt.NDArray[np.floating]
    upper_boundary: npt.NDArray[np.floating]
    fit_label: str

    def __post_init__(self) -> None:
        """Validate that all arrays have the same shape and data ranges."""
        if not (self.interpolated.shape == self.lower_boundary.shape == self.upper_boundary.shape):
            msg = "All arrays must have the same shape."
            raise ValueError(msg)

        # Check that provided interpolated is within [0, 1]
        # boundaries are also within [0, 1]
        # Since the fit could technically exceed it slightly or we just want to warn/clip.
        # Provided `interpolated` is within `[0, 1)`, boundaries are also within `[0, 1)`.
        if not np.all((self.interpolated >= 0) & (self.interpolated <= 1)):
            msg = "Interpolated values must be within [0, 1]"
            raise ValueError(msg)
        if not np.all((self.lower_boundary >= 0) & (self.lower_boundary <= 1)):
            msg = "Lower boundary values must be within [0, 1]"
            raise ValueError(msg)
        if not np.all((self.upper_boundary >= 0) & (self.upper_boundary <= 1)):
            msg = "Upper boundary values must be within [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True)
class LambdaResult(Interpolated):
    """Result type holding the data needed to plot a Lambda fit.

    Attributes:
        distances: Interpolated distance grid for the fit curve.
        interpolated: Interpolated logical error probability per round values
            along the distance grid.
        lower_boundary: Lower boundary of the error band.
        upper_boundary: Upper boundary of the error band.
        fit_label: The label to use for the fit curve.
    """

    distances: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.all(self.distances > 0):
            msg = "Distances must be positive."
            raise ValueError(msg)


def interpolate_lambda(
    lambda_data: LambdaData,
    distances: npt.NDArray[np.int_],
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
) -> LambdaResult:
    """Compute the interpolated Lambda fit curve and its error band.

    Args:
        lambda_data: Results from calculate_lambda_and_lambda_stddev (a :class:`LambdaData` instance).
        distances: The code distances used for interpolation.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.

    Returns:
        The interpolated fit data with error boundaries.
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

    fit_label = f"Fit, Λ={lambda_:.4f} ± {num_sigmas * lambda_stddev:.4f} ({num_sigmas}σ)"  # noqa: RUF001

    return LambdaResult(
        distances=distances_interpolated,
        interpolated=np.clip(interpolated, 0, 1),
        lower_boundary=np.clip(lower_boundary, 0, 1),
        upper_boundary=np.clip(upper_boundary, 0, 1),
        fit_label=fit_label,
    )


@dataclass(frozen=True)
class LEPPRResult(Interpolated):
    """Result type holding the data needed to plot a LEPPR fit.

    Attributes:
        rounds: Interpolated rounds grid for the fit curve.
        interpolated: Interpolated logical error probability values along the
            rounds grid.
        lower_boundary: Lower boundary of the error band.
        upper_boundary: Upper boundary of the error band.
        fit_label: The label to use for the fit curve.
    """

    rounds: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.all(self.rounds > 0):
            msg = "Rounds must be positive."
            raise ValueError(msg)


def interpolate_leppr(
    leppr_data: LogicalErrorProbabilityPerRoundResults,
    num_rounds: npt.NDArray[np.int_],
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
) -> LEPPRResult:
    """Compute the interpolated LEPPR fit curve and its error band.

    Args:
        leppr_data: Results from compute_logical_error_per_round.
        num_rounds: The number of rounds used for interpolation.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.

    Returns:
        The interpolated fit data with error boundaries.
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

    fit_label = f"Fit, ε={leppr:.4f} ± {num_sigmas * leppr_stddev:.4f} ({num_sigmas}σ)"  # noqa: RUF001

    return LEPPRResult(
        rounds=rounds_interpolated,
        interpolated=np.clip(interpolated, 0, 1),
        lower_boundary=np.clip(lower_boundary, 0, 1),
        upper_boundary=np.clip(upper_boundary, 0, 1),
        fit_label=fit_label,
    )
