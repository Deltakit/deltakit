from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.optimize


@dataclass(frozen=True)
class LambdaData:
    """Container for error suppression parameters and associated data.

    This dataclass stores the fitted error suppression factor (Λ) and
    prefactor (Λ₀), along with their standard deviations and the underlying
    data used for the fit.

    The error model assumes an exponential decay of the logical error
    probability per round (leepr) ε_d with the error suppression factors and
    code distance:

        ε_d ≈ 1 / (Λ₀ · Λ^((d+1)/2))

    where:
        - Λ (lambda) is the error suppression factor
        - Λ₀ (lambda_0) is a multiplicative offset
        - d is the code distance

    Attributes:
        lambda_: Error suppression factor. The trailing underscore
            avoids shadowing the Python keyword ``lambda``.
        lambda_std: Error suppression factor standard deviation.
        lambda0: Error suppression prefactor.
        lambda0_std: Error suppression prefactor standard deviation.
        distances: Code distances.
        leppr: Logical error probability per round.
        leppr_std: Logical error probability per round standard deviation.
    """

    lambda_: float
    lambda_std: float
    lambda0: float
    lambda0_std: float
    distances: npt.NDArray[np.int_]
    leppr: npt.NDArray[np.float64]
    leppr_std: npt.NDArray[np.float64]


_LambdaFitCallable = Callable[
    [
        npt.NDArray[np.int_],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    LambdaData,
]


class LambdaFitMethod(StrEnum):
    SHIFTED = "shifted"
    """Linear fit with shifted distances over logarithmic values."""
    LIN = "lin"
    """Linear fit over logarithmic values."""
    CURVE = "curve"
    """Non-linear fit."""


def _lambda_shifted_fit(
    distances: npt.NDArray[np.int_],
    leppr: npt.NDArray[np.float64],
    leppr_std: npt.NDArray[np.float64],
) -> LambdaData:
    """Estimate error suppression factors Λ and Λ₀ via linear fit and shifted distances.

    From the logical error probability per round (leppr) ε_d relationship
    with error suppression factors and code distance:

        ε_d ≈ 1 / (Λ₀ · Λ^((d+1)/2))

    This function fits a linear model to the logarithm of the leppr
    as a function of the distance:

        ln(ε_d) = -ln(Λ₀) - (d+1)/2 · ln(Λ)

    A linear fit of ln(ε_d) versus shifted distance d gives:

        slope  = -ln(Λ) / 2
        offset = -ln(Λ₀) - ln(Λ) / 2

    Recovering the original parameters:

         Λ  = exp(-2 · slope)
         Λ₀ = exp(-offset - ln(Λ)/2)

    Standard deviations for both fitted parameters are also computed using
    standard formulae found in:
    https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae

        σ(ln(Λ)/2) = σ(Λ) / (2 · Λ)

        σ(-offset - ln(Λ)/2)
            = sqrt( σ(offset)² + σ(Λ)² / (4 · Λ²)
                    - 2 · cov(offset, ln(Λ)/2) )

        σ(Λ₀)
            = Λ₀ · sqrt( σ(offset)² + σ(Λ)² / (4 · Λ²)
                         - 2 · cov(offset, ln(Λ)/2) )

    Attributes:
        distances: Code distances.
        leppr: Logical error probability per round.
        leppr_std: Logical error probability per round standard deviation.

    Returns:
        LambdaData: A container for error suppression parameters.
    """
    # Prepare log data for linear fit.
    log_leppr = np.log(leppr)
    log_leppr_std = leppr_std / leppr
    # Fitting with numpy.polyfit provides
    # standard deviations and a covariance matrix.
    (slope, offset), cov = np.polyfit(
        distances,  # Shifted distances.
        log_leppr,
        1,
        w=1 / log_leppr_std,
        full=False,
        cov="unscaled",
    )
    slope_std, offset_std = np.sqrt(np.diagonal(cov))
    # Estimate error suppression factors.
    estimated_lambda = float(np.exp(-2 * slope))
    estimated_lambda_std = float(estimated_lambda * 2 * slope_std)
    estimated_lambda0 = float(np.exp(-offset - np.log(estimated_lambda) / 2))
    # Uncertainty propagation.
    estimated_lambda0_std = float(
        estimated_lambda0
        * np.sqrt(
            offset_std**2
            + estimated_lambda_std**2 / (4 * estimated_lambda**2)
            - 2 * cov[0, 1]
        )
    )
    return LambdaData(
        lambda_=estimated_lambda,
        lambda_std=estimated_lambda_std,
        lambda0=estimated_lambda0,
        lambda0_std=estimated_lambda0_std,
        distances=distances,
        leppr=leppr,
        leppr_std=leppr_std,
    )


def _lambda_lin_fit(
    distances: npt.NDArray[np.int_],
    leppr: npt.NDArray[np.float64],
    leppr_std: npt.NDArray[np.float64],
) -> LambdaData:
    """Estimate error suppression factors Λ and Λ₀ via linear fit.

    From the logical error probability per round (leppr) ε_d relationship
    with error suppression factors and code distance:

        ε_d ≈ 1 / (Λ₀ · Λ^((d+1)/2))

    This function fits a linear model to the logarithm of the leppr
    as a function of the distance:

        ln(ε_d) = -ln(Λ₀) - (d+1)/2 · ln(Λ)

    A linear fit of ln(ε_d) versus distance (d+1)/2 gives:

        slope  = -ln(Λ)
        offset = -ln(Λ₀)

    Recovering the original parameters:

         Λ  = exp(-slope)
         Λ₀ = exp(-offset)

    Standard deviations for both fitted parameters are also computed using
    standard formulae found in:
    https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae

        σ(Λ)  = Λ · σ(slope)
        σ(Λ₀) = Λ₀ · σ(offset)

    Attributes:
        distances: Code distances.
        leppr: Logical error probability per round.
        leppr_std: Logical error probability per round standard deviation.

    Returns:
        LambdaData: A container for error suppression parameters.
    """
    # Prepare log data for linear fit.
    log_leppr = np.log(leppr)
    log_leppr_std = leppr_std / leppr
    # Fitting with numpy.polyfit provides
    # standard deviations and a covariance matrix.
    (slope, offset), cov = np.polyfit(
        (distances + 1) / 2,
        log_leppr,
        1,
        w=1 / log_leppr_std,
        full=False,
        cov="unscaled",
    )
    slope_std, offset_std = np.sqrt(np.diagonal(cov))
    # Estimate error suppression factors.
    estimated_lambda = float(np.exp(-slope))
    estimated_lambda_std = float(estimated_lambda * slope_std)
    estimated_lambda0 = float(np.exp(-offset))
    estimated_lambda0_std = float(estimated_lambda0 * offset_std)
    return LambdaData(
        lambda_=estimated_lambda,
        lambda_std=estimated_lambda_std,
        lambda0=estimated_lambda0,
        lambda0_std=estimated_lambda0_std,
        distances=distances,
        leppr=leppr,
        leppr_std=leppr_std,
    )


) -> LambdaData:
    """Compute Λ, Λ_0 and their associated standard deviations by fitting
    ``lep_per_round`` to ``1 / Λ_0 * Λ**(-(distance + 1) / 2)`` directly.

    This method does not rely on least-square polynomial fitting but rather on a more
    generic method. As such, it requires more time to converge.
    """
    # Prepare data for the fit.
    distances = np.asarray(distances, dtype=np.int_)
    # Here we are not fitting a polynomial anymore but directly the formula:
    #   Ɛ_d = 1 / [ Λ_0 * Λ**((d+1)/2) ]
    # with ``x`` that is ``(d+1)/2``.
    (lamb0, lamb), cov = scipy.optimize.curve_fit(
        lambda x, lamb0, lamb: 1 / lamb0 * lamb ** (-x),
        (distances + 1) / 2,
        lep_per_round,
        sigma=lep_stddev_per_round,
        absolute_sigma=True,
        jac=lambda x, lamb0, lamb: np.transpose(
            [
                -1 / lamb0**2 * lamb ** (-x),
                -1 / lamb0 * x * lamb ** (-x - 1),
            ]
        ),
        bounds=(0, np.inf),  # Ensure convergence in pathological cases.
        maxfev=10000,
    )
    lamb0_std, lamb_std = np.sqrt(np.diagonal(cov))
    return LambdaData(
        lambda_=float(lamb),
        lambda_std=float(lamb_std),
        lambda0=float(lamb0),
        lambda0_std=float(lamb0_std),
        distances=distances,
        leppr=leppr,
        leppr_std=leppr_std,
    )


_LAMBDA_FIT_METHODS: dict[
    LambdaFitMethod, _LambdaFitCallable
] = {
    LambdaFitMethod.SHIFTED: _lambda_shifted_fit,
    LambdaFitMethod.LIN: _lambda_lin_fit,
    LambdaFitMethod.CURVE: _lambda_curve_fit,
}


def calculate_lambda_and_lambda_std(
    distances: npt.NDArray[np.int_] | Sequence[int],
    leppr: npt.NDArray[np.float64] | Sequence[float],
    leppr_std: npt.NDArray[np.float64] | Sequence[float],
    method: LambdaFitMethod | str = LambdaFitMethod.LIN,
) -> LambdaData:
    """Calculate the error suppression factor (Λ) and its standard deviation.

    Requires the logical error probability (LEP) per round (which may be approximated
    as LEP / num_rounds for small LEP or computed with
    :func:`compute_logical_error_per_round` for a more precise approximation), and its
    standard deviation (also returned by :func:`compute_logical_error_per_round`).

    By providing the logical error probability for increasing code distances,
    one can obtain an estimate for how error suppression scales with distances.
    Note that lambda is a "rule of thumb". This approximation is unreliable near
    threshold and for low code distances. If such a regime is detected, a warning will
    be emitted by this function.

    Args:
        distances (npt.NDArray[numpy.int\\_] | Sequence[int]): Distances at which
            ``lep_per_round`` and ``lep_stddev_per_round`` are provided. Should only
            contain odd distances. Estimations of Λ may be unreliable when data from
            distance 3 is used and the value of Λ is low (see Fig. S15 of Supplementary
            information of "Quantum error correction below the surface code threshold"
            at https://www.nature.com/articles/s41586-024-08449-y#Sec8). If such a
            situation is encountered, a warning will be emitted.
        lep_per_round (npt.NDArray[numpy.float64] | Sequence[float]):
            logical error probabilities per round computed for each code distance in
            ``distances``. Should be the same size as ``distances``.
        lep_stddev_per_round (npt.NDArray[numpy.float64] | Sequence[float]):
            standard deviation of the logical error probabilities per round computed for
            each code distance in ``distances``. Should be the same size as
            ``distances``.
        method (Literal["d", "(d+1)/2", "direct"]): mathematical method used to fit the
            data. Defaults to "(d+1)/2". All 3 methods show remarkable numerical
            agreement, but "direct" is slower than both "d" and "(d+1)/2", so these last
            2 should be preferred in general.

    Returns:
        LambdaData: detailed results of the computation.

    Note:
        For values of Λ very close to 1 (``abs(Λ - 1) < 1e-7``) and
        ``method == "direct"``, this function might emit a
        ``scipy.optimize._optimize.OptimizeWarning`` with the message ``"Covariance of
        the parameters could not be estimated"``.

        Realistically, that condition is not expected to occur in practice due to
        sampling noise and sampling overhead, but it might be checked by synthetic
        data (e.g., in unit-tests).

    Examples:
        Fitting the Λ value given information for 5, 7, and 9 round of a QEC
        experiment::

            res = calculate_lambda_and_lambda_stddev(
                distances=[5, 7, 9],
                lep_per_round=[1.992e-04, 4.314e-05, 7.556e-06],
                lep_stddev_per_round=[1.2e-05, 9.3e-06, 3.9e-06],
            )
            lambda_, lambda_stddev = res.lambda_, res.lambda_stddev

    """
    # Make sure that the inputs are numpy arrays sorted by distance
    isort = np.argsort(distances)
    distances = np.asarray(distances)[isort]
    lep_per_round = np.asarray(lep_per_round)[isort]
    lep_stddev_per_round = np.asarray(lep_stddev_per_round)[isort]

    # Check that we do not have duplicate data for the same distance as that will
    # confuse the numerical methods used in this function.
    unique_counts = np.unique_counts(distances)
    non_unique_entries_mask = unique_counts.counts > 1
    if np.any(non_unique_entries_mask):
        non_unique_values = unique_counts.values[non_unique_entries_mask].tolist()
        msg = (
            "Multiple entries were provided for the following distances: "
            f"{non_unique_values}. This is not supported."
        )
        raise ValueError(msg)

    if method not in _LAMBDA_FITTING_METHODS:
        warnings.warn(
            "Got a fitting method that is not supported by this function "
            f"('{method}'). Valid methods are {list(_LAMBDA_FITTING_METHODS)}."
        )
    lambda_fit_func: _LambdaFittingCallable = _LAMBDA_FITTING_METHODS.get(
        method, _lambda_fit_with_d
    )
    res = lambda_fit_func(distances, lep_per_round, lep_stddev_per_round)
    if res.lambda_ < 1.5 and min(distances) < 5:
        warnings.warn(
            "Lambda estimation is unreliable at low code distances and low values of "
            "lambda. Please use distance 5 as a minimum.",
        )
    return res
