from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.optimize


@dataclass(frozen=True)
class LambdaResults:
    """Named-tuple-like class containing computation results from
    :func:`calculate_lambda_and_lambda_stddev`.

    Attributes:
        lambda_ (float): computed error suppression factor.
        lambda_stddev (float): lambda standard deviation.
        lambda0 (float): computed error suppression multiplicative offset (value of Λ_0
            in the expression ``Ɛ_d = 1 / [ Λ_0 * Λ**((d+1)/2) ]``).
        lambda0_stddev (float): Λ_0 standard deviation.
    """

    lambda_: float
    lambda_stddev: float
    lambda0: float
    lambda0_stddev: float


_LambdaFittingCallable = Callable[
    [
        npt.NDArray[np.int_] | Sequence[int],
        npt.NDArray[np.float64] | Sequence[float],
        npt.NDArray[np.float64] | Sequence[float],
    ],
    LambdaResults,
]


def _lambda_fit_with_d(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float],
) -> LambdaResults:
    """Compute Λ, Λ_0 and their associated standard deviations by fitting the logarithm
    of ``lep_per_round`` with ``distance``.
    """
    # Prepare data for the fit.
    lep_per_round = np.asarray(lep_per_round, dtype=np.float64)
    logleppr = np.log(lep_per_round)
    logleppr_stddev = lep_stddev_per_round / lep_per_round
    # Fitting with numpy.polyfit to be able to provide standard deviations and recover a
    # covariance matrix as numpy.polynomial.Polyfit is not able to do that yet.
    (slope, offset), cov = np.polyfit(
        distances, logleppr, 1, w=1 / logleppr_stddev, full=False, cov="unscaled"
    )
    slope_stddev, offset_stddev = np.sqrt(np.diagonal(cov))
    # Recovering the numbers of interest. Maths representing what has been performed:
    # We start from Ɛ_d = 1 / [ Λ_0 * Λ**((d+1)/2) ]
    # Applying ln:  ln(Ɛ_d) = - ln(Λ_0) - (d+1)/2 * ln(Λ)
    #                       = - ln(Λ_0) - ln(Λ)/2 - d * ln(Λ)/2
    # The linear fit performed above gave us slope  = -ln(Λ)/2
    #                                        offset = -ln(Λ_0) - ln(Λ)/2
    lambda_value = float(np.exp(-2 * slope))
    lambda_value_stddev = float(lambda_value * 2 * slope_stddev)
    lambda0 = float(np.exp(-offset - np.log(lambda_value) / 2))
    # Λ_0 = exp(-offset - ln(Λ)/2)
    # Error analysis (to compute the standard deviation of Λ_0) done with the formulas
    # in https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae:
    # σ(ln(Λ)/2) = σ(Λ) / (2 * Λ)
    # σ(offset) is obtained from the covariance matrix
    # σ(-offset - ln(Λ)/2) = √(σ(offset)² + σ(ln(Λ) / 2)²
    #                          - 2 * covariance(offset, ln(Λ) / 2))
    #                      = √(σ(offset)² + σ(Λ)² / (4 * Λ²)
    #                          - 2 * covariance(offset, ln(Λ) / 2))
    # σ(exp(-offset - ln(Λ)/2)) = exp(-offset - ln(Λ)/2) * σ(-offset - ln(Λ)/2)
    #                           = Λ_0 * √(σ(offset)² + σ(Λ)² / (4 * Λ²)
    #                                     - 2 * covariance(offset, ln(Λ) / 2))
    lambda0_stddev = float(
        lambda0
        * np.sqrt(
            offset_stddev**2
            + lambda_value_stddev**2 / (4 * lambda_value**2)
            - 2 * cov[0, 1]
        )
    )
    return LambdaResults(lambda_value, lambda_value_stddev, lambda0, lambda0_stddev)


def _lambda_fit_with_d_plus_1_over_2(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float],
) -> LambdaResults:
    """Compute Λ, Λ_0 and their associated standard deviations by fitting the logarithm
    of ``lep_per_round`` with ``(distance + 1) / 2``.
    """
    # Prepare data for the fit.
    distances = np.asarray(distances, dtype=np.int_)
    lep_per_round = np.asarray(lep_per_round, dtype=np.float64)
    logleppr = np.log(lep_per_round)
    logleppr_stddev = lep_stddev_per_round / lep_per_round
    # Fitting with numpy.polyfit to be able to provide standard deviations and recover a
    # covariance matrix as numpy.polynomial.Polyfit is not able to do that yet.
    (slope, offset), cov = np.polyfit(
        (distances + 1) / 2,
        logleppr,
        1,
        w=1 / logleppr_stddev,
        full=False,
        cov="unscaled",
    )
    slope_stddev, offset_stddev = np.sqrt(np.diagonal(cov))
    # Recovering the numbers of interest. Maths representing what has been performed:
    # We start from Ɛ_d = 1 / [ Λ_0 * Λ**((d+1)/2) ]
    # Applying ln:  ln(Ɛ_d) = - ln(Λ_0) - (d+1)/2 * ln(Λ)
    # The linear fit performed above gave us slope  = -ln(Λ)
    #                                        offset = -ln(Λ_0)
    lambda_value = float(np.exp(-slope))
    lambda_value_stddev = float(lambda_value * slope_stddev)
    lambda0 = float(np.exp(-offset))
    lambda0_stddev = float(lambda0 * offset_stddev)
    return LambdaResults(lambda_value, lambda_value_stddev, lambda0, lambda0_stddev)


def _lambda_fit_with_direct(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float],
) -> LambdaResults:
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
        # Both parameters below are needed for crazy values of lambda and lambda0 to
        # make sure the method converges to the correct value.
        bounds=(0, np.inf),
        maxfev=10000,
    )
    lamb0_stddev, lamb_stddev = np.sqrt(np.diagonal(cov))
    return LambdaResults(
        float(lamb), float(lamb_stddev), float(lamb0), float(lamb0_stddev)
    )


_LAMBDA_FITTING_METHODS: dict[
    Literal["d", "(d+1)/2", "direct"], _LambdaFittingCallable
] = {
    "d": _lambda_fit_with_d,
    "(d+1)/2": _lambda_fit_with_d_plus_1_over_2,
    "direct": _lambda_fit_with_direct,
}


def calculate_lambda_and_lambda_stddev(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float],
    method: Literal["d", "(d+1)/2", "direct"] = "(d+1)/2",
) -> LambdaResults:
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
        LambdaResults: detailed results of the computation.

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

    # Make sure that there are no even distances.
    if np.any(distances % 2 == 0):
        msg = (
            "Found at least one even distance in the provided distances "
            f"({distances.tolist()}). This is not supported."
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
