# (c) Copyright Riverlane 2020-2025.
"""Result types for plotting LEPPR and Lambda data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.analysis import (
    LogicalErrorProbabilityPerRoundData as LEPPRData,
)


def _lambda_interpolated(
    lambda0: float,
    lambda_: float,
    distances: npt.NDArray[np.int_ | np.floating],
) -> npt.NDArray[np.floating]:
    """Estimate the logical error probability per round for given parameters.

    The estimate is based on the formula

        ε = (1 / Λ₀) * Λ**(-(d + 1) / 2)

    where:
      - ε is the logical error probability per round,
      - Λ₀ is a normalisation constant,
      - Λ is the error suppression factor,
      - d is the code distance.

    For each distance in ``distances``, this function computes the corresponding
    logical error probability using the supplied ``lambda_`` (Λ) and ``lambda0`` (Λ₀).

    Args:
        lambda0: Normalisation constant Λ₀.
        lambda_: Error suppression factor Λ.
        distances: Iterable of code distances d.

    Returns:
        An array containing the estimated logical error probability per round
        for each provided distance.
    """
    return lambda_ ** (-(distances + 1) / 2) / lambda0


def _lep_interpolated(
    spam: float, leppr: float, rounds_interpolated: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute the logical error probability corresponding to the given parameters.

    The expected computation fidelity is modelled as

        F = Fs * Fε**r

    where:
      - F is the overall fidelity of the computation,
      - Fs is the fidelity of SPAM-related operations,
      - Fε is the fidelity of a single quantum error-correction round,
      - r is the number of error-correction rounds performed.

    Each fidelity value is derived from its associated error probability using

        f = 1 - 2e

    where:
      - f represents F, Fs, or Fε,
      - e represents the corresponding logical error probability
        (overall, SPAM-related, or per-round).

    Args:
        spam: Error probability associated with SPAM operations.
        leppr: Error probability per error-correction round.
        rounds_interpolated: Number of error-correction rounds performed.

    Returns:
        Logical error probability of the full computation.
    """
    expected_fidelity = (1 - 2 * spam) * (1 - 2 * leppr) ** rounds_interpolated
    return (1 - expected_fidelity) / 2


@dataclass(frozen=True)
class Interpolated:
    """Container for interpolated plotting data and associated confidence bounds.

    Stores the interpolated central values together with their lower and upper
    confidence interval boundaries, along with labels describing the fit and
    its confidence interval for visualisation purposes.

    Attributes:
        interpolated: Array of interpolated central values (e.g., fitted curve values).
        lower_boundary: Array representing the lower bound of the confidence interval
            corresponding to ``interpolated``.
        upper_boundary: Array representing the upper bound of the confidence interval
            corresponding to ``interpolated``.
        fit_label: Label describing the fitted/interpolated data (e.g., legend entry).
        confidence_interval_label: Label describing the confidence interval region (e.g., legend entry).
    """

    interpolated: npt.NDArray[np.floating]
    lower_boundary: npt.NDArray[np.floating]
    upper_boundary: npt.NDArray[np.floating]
    fit_label: str
    confidence_interval_label: str

    def __post_init__(self) -> None:
        """Validate consistency of interpolated data and confidence bounds.

        Ensures that ``interpolated``, ``lower_boundary``, and ``upper_boundary``
        arrays all have identical shapes so that each interpolated value has
        corresponding lower and upper confidence bounds.

        Raises:
            ValueError: If the shapes of ``interpolated``,
                ``lower_boundary``, and ``upper_boundary`` do not match.
        """
        if not (
            self.interpolated.shape
            == self.lower_boundary.shape
            == self.upper_boundary.shape
        ):
            msg = (
                "The 'interpolated', 'lower_boundary', and 'upper_boundary' arrays "
                f"must have the same shape. Got {self.interpolated.shape}, "
                f"{self.lower_boundary.shape}, and {self.upper_boundary.shape} respectively."
            )
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
    """

    distances: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.all(self.distances > 0):
            msg = "Distances must be positive."
            raise ValueError(msg)
        if self.distances.shape != self.interpolated.shape:
            msg = (
                f"The 'distances' array shape {self.distances.shape} must match the "
                f"'interpolated' array shape {self.interpolated.shape}."
            )
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

    fit_label = (
        f"Fit, Λ={lambda_:.4f} ± {num_sigmas * lambda_stddev:.4f} ({num_sigmas}σ)"  # noqa: RUF001
    )

    return LambdaResult(
        distances=distances_interpolated,
        interpolated=np.clip(interpolated, 0, 1),
        lower_boundary=np.clip(lower_boundary, 0, 1),
        upper_boundary=np.clip(upper_boundary, 0, 1),
        fit_label=fit_label,
        confidence_interval_label=f"Confidence interval ({num_sigmas}σ) on Λ fit",  # noqa: RUF001
    )


@dataclass(frozen=True)
class LogicalErrorProbabilityPerRoundResult(Interpolated):
    """Result type holding the data needed to plot a LogicalErrorProbabilityPerRound (LEPPR) fit.

    Attributes:
        rounds: Interpolated rounds grid for the fit curve.
    """

    rounds: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.all(self.rounds > 0):
            msg = "Rounds must be positive."
            raise ValueError(msg)
        if self.rounds.shape != self.interpolated.shape:
            msg = (
                f"The 'rounds' array shape {self.rounds.shape} must match the "
                f"'interpolated' array shape {self.interpolated.shape}."
            )
            raise ValueError(msg)


def interpolate_leppr(
    leppr_data: LEPPRData,
    num_rounds: npt.NDArray[np.int_],
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
) -> LogicalErrorProbabilityPerRoundResult:
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

    return LogicalErrorProbabilityPerRoundResult(
        rounds=rounds_interpolated,
        interpolated=np.clip(interpolated, 0, 1),
        lower_boundary=np.clip(lower_boundary, 0, 1),
        upper_boundary=np.clip(upper_boundary, 0, 1),
        fit_label=fit_label,
        confidence_interval_label=f"Confidence interval ({num_sigmas}σ) on ε fit",  # noqa: RUF001
    )
