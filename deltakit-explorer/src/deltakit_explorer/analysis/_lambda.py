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
        lambda_error_high (float): High error
        lambda_error_low (float): Low error
    """

    lambda_: float
    lambda_stddev: float
    lambda0: float
    lambda0_stddev: float
    lambda_error_high: float
    lambda_error_low: float


_LambdaFittingCallable = Callable[
    [
        npt.NDArray[np.int_] | Sequence[int],
        npt.NDArray[np.float64] | Sequence[float],
        npt.NDArray[np.float64] | Sequence[float],
        npt.NDArray[np.float64] | Sequence[float],
        int,
    ],
    LambdaResults,
]


def _log_leppr_errors(
    lep_per_round: npt.NDArray[np.float64],
    lep_per_round_low: npt.NDArray[np.float64],
    lep_per_round_high: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Propagate asymmetric LEP-per-round errors onto ``log(lep_per_round)``.

    ``log`` is monotonically *increasing*, so the low/high error magnitudes map onto the
    low/high log-space error magnitudes directly. To first order the magnitude of the
    derivative ``d log(leppr) / d leppr`` is ``1 / leppr``, hence the division below.

    Args:
        lep_per_round: logical error probabilities per round.
        lep_per_round_low: lower (low-side) asymmetric error magnitudes on
            ``lep_per_round``.
        lep_per_round_high: upper (high-side) asymmetric error magnitudes on
            ``lep_per_round``.

    Returns:
        the low- and high-side error magnitudes in log space.
    """
    logleppr_low = lep_per_round_low / lep_per_round
    logleppr_high = lep_per_round_high / lep_per_round
    return logleppr_low, logleppr_high


def _lambda_fit_with_d(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_low: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_high: npt.NDArray[np.float64] | Sequence[float],
    num_sigmas: int = 1,
) -> LambdaResults:
    """Compute Λ, Λ_0 and their associated standard deviations by fitting the logarithm
    of ``lep_per_round`` with ``distance``.
    """
    # Prepare data for the fit.
    lep_per_round = np.asarray(lep_per_round, dtype=np.float64)
    logleppr = np.log(lep_per_round)
    logleppr_low, logleppr_high = _log_leppr_errors(
        lep_per_round,
        np.asarray(lep_per_round_low, dtype=np.float64),
        np.asarray(lep_per_round_high, dtype=np.float64),
    )
    # The polyfit estimator below only accepts a single symmetric weight per point, so we
    # feed it the mean of the asymmetric log-space errors. The asymmetry is preserved in
    # the returned ``lambda_error_low``/``lambda_error_high`` via the propagation below.
    logleppr_stddev = (logleppr_low + logleppr_high) / 2
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
    # Asymmetric num_sigmas CI for Λ: propagate the slope Gaussian uncertainty through
    # exp(-2*slope). The CI is symmetric in log space but asymmetric in linear space, so
    # the num_sigmas factor must go *inside* the exponential:
    #   λ·(exp(N·σ) − 1) ≠ N·λ·(exp(σ) − 1).
    lambda_error_high = float(lambda_value * (np.exp(2 * num_sigmas * slope_stddev) - 1))
    lambda_error_low = float(lambda_value * (1 - np.exp(-2 * num_sigmas * slope_stddev)))
    return LambdaResults(
        lambda_value, lambda_value_stddev, lambda0, lambda0_stddev,
        lambda_error_high, lambda_error_low,
    )


def _lambda_fit_with_d_plus_1_over_2(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_low: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_high: npt.NDArray[np.float64] | Sequence[float],
    num_sigmas: int = 1,
) -> LambdaResults:
    """Compute Λ, Λ_0 and their associated standard deviations by fitting the logarithm
    of ``lep_per_round`` with ``(distance + 1) / 2``.
    """
    # Prepare data for the fit.
    distances = np.asarray(distances, dtype=np.int_)
    lep_per_round = np.asarray(lep_per_round, dtype=np.float64)
    logleppr = np.log(lep_per_round)
    logleppr_low, logleppr_high = _log_leppr_errors(
        lep_per_round,
        np.asarray(lep_per_round_low, dtype=np.float64),
        np.asarray(lep_per_round_high, dtype=np.float64),
    )
    # The polyfit estimator below only accepts a single symmetric weight per point, so we
    # feed it the mean of the asymmetric log-space errors. The asymmetry is preserved in
    # the returned ``lambda_error_low``/``lambda_error_high`` via the propagation below.
    logleppr_stddev = (logleppr_low + logleppr_high) / 2
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
    # Asymmetric num_sigmas CI for Λ: propagate the slope Gaussian uncertainty through
    # exp(-slope). The num_sigmas factor goes inside the exponential (see the comment in
    # _lambda_fit_with_d for why this is not the same as scaling the linear-space error).
    lambda_error_high = float(lambda_value * (np.exp(num_sigmas * slope_stddev) - 1))
    lambda_error_low = float(lambda_value * (1 - np.exp(-num_sigmas * slope_stddev)))
    return LambdaResults(
        lambda_value, lambda_value_stddev, lambda0, lambda0_stddev,
        lambda_error_high, lambda_error_low,
    )


def _lambda_fit_with_direct(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_low: npt.NDArray[np.float64] | Sequence[float],
    lep_per_round_high: npt.NDArray[np.float64] | Sequence[float],
    num_sigmas: int = 1,
) -> LambdaResults:
    """Compute Λ, Λ_0 and their associated standard deviations by fitting
    ``lep_per_round`` to ``1 / Λ_0 * Λ**(-(distance + 1) / 2)`` directly.

    This method does not rely on least-square polynomial fitting but rather on a more
    generic method. As such, it requires more time to converge.
    """
    # Prepare data for the fit.
    distances = np.asarray(distances, dtype=np.int_)
    lep_per_round = np.asarray(lep_per_round, dtype=np.float64)
    lep_per_round_low = np.asarray(lep_per_round_low, dtype=np.float64)
    lep_per_round_high = np.asarray(lep_per_round_high, dtype=np.float64)
    x = (distances + 1) / 2

    # Here we are not fitting a polynomial anymore but directly the formula:
    #   Ɛ_d = 1 / [ Λ_0 * Λ**((d+1)/2) ]
    # with ``x`` that is ``(d+1)/2``.
    def _residual(
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        lamb0, lamb = beta
        delta = lep_per_round - 1 / lamb0 * lamb ** (-x)
        # Barlow's dynamic ("variable variance") weighting for asymmetric errors,
        # matching the approach used in :func:`compute_logical_error_per_round`. The
        # effective variance uses the high-side error when the data is above the fit
        # (``delta > 0``) and the low-side error otherwise. With mean error
        # ``sigma = (sigma_high + sigma_low) / 2`` and asymmetry
        # ``sigma_prime = (sigma_high - sigma_low) / 2``, the variance is
        # ``V(delta) = sigma**2 + 2 * sigma_prime * delta``. We clip ``V`` to a small
        # positive value to keep it well defined for large negative deltas.
        sigma = (lep_per_round_high + lep_per_round_low) / 2
        sigma_prime = (lep_per_round_high - lep_per_round_low) / 2
        variance = np.clip(sigma**2 + 2 * sigma_prime * delta, 1e-300, None)
        return delta / np.sqrt(variance)

    # Both parameters need a positive lower bound for crazy values of lambda and lambda0
    # to make sure the method converges to the correct value.
    result = scipy.optimize.least_squares(
        _residual,
        x0=np.array([1.0, 1.0]),
        bounds=(1e-10, np.inf),
        max_nfev=10000,
    )
    lamb0, lamb = float(result.x[0]), float(result.x[1])

    # Recover the parameter covariance matrix from the Jacobian at the solution. The
    # residual is already scaled by 1/sqrt(variance), so the covariance is (JᵀJ)⁻¹. We
    # invert it through the SVD of ``J`` (rather than forming ``JᵀJ`` directly) to avoid
    # squaring the condition number, matching what ``scipy.optimize.curve_fit`` does.
    _, s, vt = np.linalg.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[: s.size]
    cov = (vt.T / s**2) @ vt
    lamb0_stddev, lamb_stddev = (float(v) for v in np.sqrt(np.diagonal(cov)))

    # Asymmetric num_sigmas CI for Λ. The fit is performed in linear space, so the raw
    # ``num_sigmas`` interval is symmetric; we report it as such here (the direct method
    # does not model the log-space asymmetry that the ``d``/``(d+1)/2`` methods do).
    lambda_error_high = float(num_sigmas * lamb_stddev)
    lambda_error_low = float(num_sigmas * lamb_stddev)

    return LambdaResults(
        lamb, lamb_stddev, lamb0, lamb0_stddev, lambda_error_high, lambda_error_low
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
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float] | None = None,
    method: Literal["d", "(d+1)/2", "direct"] = "(d+1)/2",
    num_sigmas: int = 1,
    *,
    lep_per_round_low: npt.NDArray[np.float64] | Sequence[float] | None = None,
    lep_per_round_high: npt.NDArray[np.float64] | Sequence[float] | None = None,
) -> LambdaResults:
    """Calculate the error suppression factor (Λ) and its standard deviation.

    Requires the logical error probability (LEP) per round (which may be approximated
    as LEP / num_rounds for small LEP or computed with
    :func:`compute_logical_error_per_round` for a more precise approximation), and its
    error (also returned by :func:`compute_logical_error_per_round`).

    The per-round error can be provided either as a symmetric standard deviation through
    ``lep_stddev_per_round`` or as the *asymmetric* lower/upper error magnitudes through
    ``lep_per_round_low``/``lep_per_round_high`` (for example the
    ``leppr_error_low``/``leppr_error_high`` returned by
    :func:`compute_logical_error_per_round`). When the asymmetric errors are provided
    they are propagated through the fit using Barlow's variable-variance weighting (see
    R. Barlow, "Asymmetric Statistical Errors", arXiv:physics/0406120), consistently with
    :func:`compute_logical_error_per_round`. Exactly one of ``lep_stddev_per_round`` or
    the ``lep_per_round_low``/``lep_per_round_high`` pair must be provided.

    By providing the logical error probability for increasing code distances,
    one can obtain an estimate for how error suppression scales with distances.
    Note that lambda is a "rule of thumb". This approximation is unreliable near
    threshold and for low code distances. If such a regime is detected, a warning will
    be emitted by this function.

    Args:
        distances (npt.NDArray[numpy.int\\_] | Sequence[int]): Distances at which
            ``lep_per_round`` and its error are provided. Should only
            contain odd distances. Estimations of Λ may be unreliable when data from
            distance 3 is used and the value of Λ is low (see Fig. S15 of Supplementary
            information of "Quantum error correction below the surface code threshold"
            at https://www.nature.com/articles/s41586-024-08449-y#Sec8). If such a
            situation is encountered, a warning will be emitted.
        lep_per_round (npt.NDArray[numpy.float64] | Sequence[float]):
            logical error probabilities per round computed for each code distance in
            ``distances``. Should be the same size as ``distances``.
        lep_stddev_per_round (npt.NDArray[numpy.float64] | Sequence[float] | None):
            standard deviation of the logical error probabilities per round computed for
            each code distance in ``distances``. Should be the same size as
            ``distances``. Mutually exclusive with
            ``lep_per_round_low``/``lep_per_round_high``. Defaults to ``None``.
        method (Literal["d", "(d+1)/2", "direct"]): mathematical method used to fit the
            data. Defaults to "(d+1)/2". All 3 methods show remarkable numerical
            agreement, but "direct" is slower than both "d" and "(d+1)/2", so these last
            2 should be preferred in general.
        lep_per_round_low (npt.NDArray[numpy.float64] | Sequence[float] | None):
            lower (low-side) asymmetric error *magnitude* on each ``lep_per_round`` value
            (such that the lower bound is ``lep_per_round - lep_per_round_low``). When
            provided together with ``lep_per_round_high`` these are used instead of
            ``lep_stddev_per_round``. Should be the same size as ``distances``. Defaults
            to ``None``.
        lep_per_round_high (npt.NDArray[numpy.float64] | Sequence[float] | None):
            upper (high-side) asymmetric error *magnitude* on each ``lep_per_round``
            value (such that the upper bound is ``lep_per_round + lep_per_round_high``).
            See ``lep_per_round_low``. Should be the same size as ``distances``. Defaults
            to ``None``.
        num_sigmas (int): number of standard deviations defining the confidence level of
            the returned asymmetric errors (``lambda_error_high``/``lambda_error_low``).
            For example, ``num_sigmas=3`` returns a 3σ asymmetric confidence interval.
            Defaults to ``1``. The symmetric ``lambda_stddev`` field is always 1σ and is
            not affected by this parameter.

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

    Raises:
        ValueError: if neither, or both, of ``lep_stddev_per_round`` and the
            ``lep_per_round_low``/``lep_per_round_high`` pair are provided, or if only one
            of ``lep_per_round_low``/``lep_per_round_high`` is provided.

    Examples:
        Fitting the Λ value given information for 5, 7, and 9 round of a QEC
        experiment::

            res = calculate_lambda_and_lambda_stddev(
                distances=[5, 7, 9],
                lep_per_round=[1.992e-04, 4.314e-05, 7.556e-06],
                lep_stddev_per_round=[1.2e-05, 9.3e-06, 3.9e-06],
            )
            lambda_, lambda_stddev = res.lambda_, res.lambda_stddev

        Propagating the asymmetric per-round errors returned by
        :func:`compute_logical_error_per_round`::

            res = calculate_lambda_and_lambda_stddev(
                distances=[5, 7, 9],
                lep_per_round=lepprs,
                lep_per_round_low=leppr_errors_low,
                lep_per_round_high=leppr_errors_high,
            )
            lambda_high, lambda_low = res.lambda_error_high, res.lambda_error_low

    """
    method = "d"
    # Validate that exactly one error specification was provided: either the symmetric
    # standard deviation, or the asymmetric low/high pair.
    has_asymmetric = lep_per_round_low is not None or lep_per_round_high is not None
    if has_asymmetric and (lep_per_round_low is None or lep_per_round_high is None):
        msg = (
            "Both 'lep_per_round_low' and 'lep_per_round_high' must be provided together "
            "to use asymmetric errors."
        )
        raise ValueError(msg)
    if (lep_stddev_per_round is None) == (not has_asymmetric):
        msg = (
            "Provide exactly one error specification: either 'lep_stddev_per_round' "
            "(symmetric) or the 'lep_per_round_low'/'lep_per_round_high' pair "
            "(asymmetric)."
        )
        raise ValueError(msg)

    # Make sure that the inputs are numpy arrays sorted by distance
    isort = np.argsort(distances)
    distances = np.asarray(distances)[isort]
    lep_per_round = np.asarray(lep_per_round)[isort]
    if has_asymmetric:
        lep_per_round_low = np.asarray(lep_per_round_low)[isort]
        lep_per_round_high = np.asarray(lep_per_round_high)[isort]
    else:
        # A symmetric standard deviation is the special case where the low- and high-side
        # error magnitudes are equal.
        lep_stddev_per_round = np.asarray(lep_stddev_per_round)[isort]
        lep_per_round_low = lep_stddev_per_round
        lep_per_round_high = lep_stddev_per_round

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
    res = lambda_fit_func(
        distances, lep_per_round, lep_per_round_low, lep_per_round_high, num_sigmas
    )
    if res.lambda_ < 1.5 and min(distances) < 5:
        warnings.warn(
            "Lambda estimation is unreliable at low code distances and low values of "
            "lambda. Please use distance 5 as a minimum.",
        )
    return res
