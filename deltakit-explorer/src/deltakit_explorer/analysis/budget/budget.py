from collections.abc import Iterator, Sequence, Mapping
import math
from pathlib import Path
from deltakit_decode.analysis._run_all_analysis_engine import RunAllAnalysisEngine
from deltakit_explorer.analysis.budget.generation import (
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.budget.memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.budget.post_processing import (
    compute_lambda_and_stddev_from_results,
)
import numpy.typing as npt
import numpy

from deltakit_explorer.analysis.budget.interfaces import NoiseInterface


def _variate_ith_parameter_by(
    central_point: npt.NDArray[numpy.float64],
    variations: npt.NDArray[numpy.float64],
    i: int,
) -> Iterator[npt.NDArray[numpy.float64]]:
    """Returns versions of ``central_point`` where the ``i``-th parameter is
    successively replaced by values in ``variations``.

    Args:
        central_point (npt.NDArray[numpy.float64]): base 1-dimensional array of numbers
            of shape ``(n,)`` that will be copied, modified on the ``i``-th variable and
            returned.
        variations (npt.NDArray[numpy.float64]): 1-dimensional array of shape ``(m,)``
            containing values that should be used to replace the ``i``-th entry of
            ``central_point``.
        i (int): index of the entry in ``central_point`` that should be changed.

    Yields:
        ``m`` arrays of shape ``(n,)`` that are copies of ``central_point`` with the
        ``i``-th coordinate entry replaced with an entry from ``variations``.
    """
    central_point = central_point.reshape((-1, 1))
    variations = variations.reshape((-1,))
    parameters = numpy.tile(central_point, (1, variations.size))
    parameters[i, :] = variations
    yield from parameters.T


def _approximate_derivative_at_point_from_values(
    x: npt.NDArray[numpy.float64],
    y: npt.NDArray[numpy.float64],
    stddevs: npt.NDArray[numpy.float64],
    gradient_approximation_point: float,
    degree: int = 3,
    noise_name: str | None = None,
) -> tuple[float, float]:
    # Perform the approximation using the provided standard deviations
    coefficients, cov = numpy.polyfit(x, y, deg=degree, cov="unscaled", w=1 / stddevs)

    # Flipping the coefficients and covariance matrix to have the index corresponding to
    # the degree (i.e., ``coefficients[i]`` is multiplying ``x**i`` in the polynomial
    # and ``cov[i,i]`` is the variance of ``coefficients[i]``).
    coefficients, cov = numpy.flip(coefficients), numpy.flip(cov)

    # Compute the derivative
    derivative = float(
        sum(
            coefficient * (power + 1) * gradient_approximation_point**power
            for power, coefficient in enumerate(coefficients[1:])
        )
    )
    # Compute the variance of the derivative estimate
    standard_deviation = math.sqrt(
        _get_variance_of_gradient_estimation_at_c(cov, gradient_approximation_point)
    )
    return derivative, standard_deviation


def _get_variance_of_gradient_estimation_at_c(
    cov: npt.NDArray[numpy.floating], c: float
) -> float:
    """Get the variance of the gradient estimation at the point ``c`` for a polynomial
    with uncertainties on its coefficients provided by the covariance matrix ``cov``.

    Note:
        Let this function be called ``f``. ``f`` preserves scalar multiplication on its
        ``cov`` input parameter. That means that for any real value of ``c`` and
        ``alpha`` the following is true: ::

            f(alpha * cov, c) == alpha * f(cov, c)

    Args:
        cov (npt.NDArray[numpy.float64]): an array of shape ``(d + 1, d + 1)``
            representing the covariance matrix of the coefficients defining the degree-d
            polynomial used to estimate the gradient.
        c (float): point at which the degree-d polynomial will be used to estimate the
            gradient value.

    Returns:
        The variance of the gradient estimation at point ``c``.
    """
    # From https://en.wikipedia.org/wiki/Covariance#Covariance_of_linear_combinations we
    # have an easy formula for the variance involving the covariance matrix. We build
    # the terms of the sum in a copy of the ``cov`` matrix.
    # We do not need to use the first row/column of the covariance matrix because it is
    # linked with the intersect, that is not used when computing the derivative.
    derivative_cov = numpy.copy(cov[1:, 1:])
    num_derivative_coefficients = derivative_cov.shape[0]
    # The original polynomial at ``x`` is given by
    #       sum(a[i] * x**i for i in range(0, n+1))
    # Its derivative is then given by
    #       sum(a[i+1] * (i+1) * x**i for i in range(0, n))
    # Removing a[0] from a (and so shifting its indexing by 1, which is what we just did
    # with the covariance matrix above) we have the derivative given by
    #       sum(a[i] * (i+1) * x**i for i in range(0, n))
    # When evaluating the gradient at the point c, the degree i coefficient of the
    # derivative (that is ``a[i]`` above or equivalently the degree (i+1) coefficient of
    # the original polynomial) is multiplied by (i + 1) * c**i. We can ignore (i.e., not
    # change) the degree 0 as the factor is (0 + 1) * c**0 = 1.
    for i in range(1, num_derivative_coefficients):
        derivative_cov[i, :] *= (i + 1) * c**i
        derivative_cov[:, i] *= (i + 1) * c**i
    # Finally, according to
    # https://en.wikipedia.org/wiki/Covariance#Covariance_of_linear_combinations,
    # the variance is given by the sum of the derivative covariance matrix above.
    return float(numpy.sum(derivative_cov))


def _get_interpolation_points(
    a: float,
    b: float,
    c: float,
    num_points: int,
    degree: int,
    x0: npt.NDArray[numpy.float64] | None = None,
) -> npt.NDArray[numpy.float64]:
    """
    Find a good set of points to estimate the gradient at point ``c`` by fitting a
    degree-d polynomial on the interval ``[a, b]``.

    This function finds ``num_points`` points ``x`` between ``a`` and ``b`` that should
    be used to fit a degree-d polynomial and will minimise the variance of the fitted
    polynomial gradient at point ``c``.

    The polynomial regression problem is presented here:
    https://en.wikipedia.org/wiki/Polynomial_regression#Matrix_form_and_calculation_of_estimates.

    Assuming the observation errors (i.e., input errors) are uncorrelated, the
    covariance matrix of the estimated coefficients is: ::

        (XᵀWX)⁻¹

    where X is the Vandermonde matrix (https://en.wikipedia.org/wiki/Vandermonde_matrix)
    obtained from the points on which the function has been evaluated and W is the
    inverse of the covariance matrix of the input parameters.

    In our specific case, the input parameters come from uncorrelated random samplings
    so W is a diagonal matrix with each diagonal entry being 1 / σ(in)² which means that
    it commutes with both Xᵀ and X and so can be factored out of the matrix
    inversion: ::

        W⁻¹ (XᵀX)⁻¹

    Because W is an input parameter outside of our control at that point (this function
    is concerned about providing the best points at which the function to fit should be
    evaluated), we can ignore it from further analysis.

    This means that this function should aim at finding ``num_points`` points in
    ``[a, b]`` such that (XᵀX)⁻¹ (which only depends on the points at which the
    function to fit is evaluated) minimise some metric.

    Because we know in this specific case that we will use that degree-d polynomial to
    approximate the gradient at point ``c``, we know exactly which quantity we want to
    minimise: the variance of the resulting gradient estimate.

    The variance of the gradient estimate is computed from the covariance matrix of the
    polynomial coefficients. One problem is that this covariance matrix cannot be
    evaluated until we have W. One way of solving this problem is to assume that W is
    a constant times the identity matrix (which is equivalent to assuming that each
    result from the original function evaluation is associated with the same standard
    deviation). This should not be too far from the reality in our case because each
    estimation of 1 / Λ is done with the same number of distances, repetitions and
    shots. With:

    1. this assumption (W = aI),
    2. the fact that the function computing the variance of the gradient estimation
       preserves scalar multiplication on its ``cov`` parameter,
    3. and the fact that finding the parameters minimising the cost function C(.) and
       a * C(.) are the same provided that a > 0,

    we can find our optimal points by minimising the variance simply computed from
    ``X``.
    """
    assert a < c < b
    if x0 is None:
        x0 = numpy.linspace(a, b, num_points, dtype=numpy.float64)

    def f(x: npt.NDArray[numpy.floating]) -> float:
        ordered_x = numpy.sort(x)
        X = numpy.vander(ordered_x, degree + 1, increasing=True)
        cov = numpy.linalg.inv(X.T @ X)
        res = _get_variance_of_gradient_estimation_at_c(cov, c)
        return res

    # result = scipy.optimize.minimize(f, x0, bounds=[(a, b) for _ in range(num_points)])

    return numpy.linspace(a, b, num_points, dtype=numpy.float64)


def get_error_budget(
    noise_model: NoiseInterface,
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: list[tuple[float, float]],
    num_points_per_parameters: int = 10,
    num_shots: int = 10_000_000,
    batch_size: int = 10_000,
    memory_generator: MemoryGenerator = get_rotated_surface_code_memory_circuit,
    lep_target_rse: float = 1e-4,
    lep_computation_min_fails: int = 10,
    fitting_degree: int = 3,
    max_workers: int = 1,
    data_file: Path | None = None,
) -> tuple[float, float, npt.NDArray[numpy.float64], npt.NDArray[numpy.float64]]:
    # Getting the points on which we will estimate 1 / Λ into ``noise_parameters``.
    central_point = noise_model.noise_parameters.reshape((-1, 1))
    xis: list[npt.NDArray[numpy.float64]] = [central_point.reshape((-1,))]
    for i, (minimum, maximum) in enumerate(noise_parameters_exploration_bounds):
        variations = _get_interpolation_points(
            minimum,
            maximum,
            noise_model.noise_parameters[i],
            num_points_per_parameters,
            fitting_degree,
        )
        xis.extend(_variate_ith_parameter_by(central_point, variations, i))
    # Note: noise_parameters[:,0] is always ``noise_model.noise_parameters``.
    noise_parameters = numpy.asarray(xis, dtype=numpy.float64).T

    # ``noise_parameters`` contains all the noise parameters we want to evaluate 1 / Λ.
    # Prepare the computation by building the decoder managers.
    decoder_managers = generate_decoder_managers_for_lambda(
        noise_parameters,
        type(noise_model),
        num_rounds_by_distances,
        max_workers,
        memory_generator=memory_generator,
    )

    # Start the computation
    num_points = noise_model.num_noise_parameters * num_points_per_parameters
    engine = RunAllAnalysisEngine(
        experiment_name=f"Estimating Λ on {num_points} points",
        decoder_managers=decoder_managers,
        max_shots=num_shots,
        batch_size=batch_size,
        # Early stopping when we have a low-enough standard deviation
        loop_condition=RunAllAnalysisEngine.loop_until_observable_rse_below_threshold(
            lep_target_rse, lep_computation_min_fails
        ),
        num_parallel_processes=max_workers,
    )
    report = engine.run()
    if data_file is not None:
        report.to_csv(data_file)

    # Post-process the results to get all the estimations for 1 / Λ
    lambdas, lambda_stddevs = compute_lambda_and_stddev_from_results(
        noise_parameters, noise_model.parameter_names, num_rounds_by_distances, report
    )
    lambda_reciprocals = 1 / lambdas
    lambda_reciprocal_stddevs = numpy.abs(lambda_stddevs / lambdas**2)

    # We now have all the estimations of 1 / Λ, we can approximate the gradient
    # Note that ``noise_parameters``, ``lambda_reciprocals`` and
    # ``lambda_reciprocal_stddevs`` have the same shapes: a 2-dimensional array with
    # values as columns. Additionally, the first column corresponds to the original
    # noise model parameters, and each following group of ``num_points_per_parameters``
    # columns correspond to the variation of 1 parameter.
    gradient: list[float] = []
    gradient_stddev: list[float] = []
    for npi, noise_parameter in enumerate(noise_model.noise_parameters):
        start = 1 + num_points_per_parameters * npi
        end = 1 + num_points_per_parameters * (npi + 1)
        # Index 0 is ``noise_model.noise_parameters``, so it can be included in all
        # estimations.
        column_indices = [0, *list(range(start, end))]
        x = noise_parameters[npi, column_indices]
        y = lambda_reciprocals[0, column_indices]
        stddevs = lambda_reciprocal_stddevs[0, column_indices]
        derivative, stddev = _approximate_derivative_at_point_from_values(
            x,
            y,
            stddevs,
            noise_parameter,
            degree=fitting_degree,
            noise_name=noise_model.parameter_names[npi],
        )
        gradient.append(derivative)
        gradient_stddev.append(stddev)

    contributions = numpy.abs(gradient * noise_model.noise_parameters)
    stddevs = numpy.abs(gradient_stddev * noise_model.noise_parameters)
    return (
        lambda_reciprocals[0, 0],
        lambda_reciprocal_stddevs[0, 0],
        contributions,
        stddevs,
    )
