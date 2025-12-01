import math
import warnings
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy
import numpy.typing as npt
import pandas
from deltakit_decode.analysis._run_all_analysis_engine import RunAllAnalysisEngine

from deltakit_explorer.analysis.budget.discretisation import (
    GradientFitDiscretisationGenerator,
    get_logarithmic_points,
)
from deltakit_explorer.analysis.budget.generation import (
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.budget.interfaces import NoiseInterface
from deltakit_explorer.analysis.budget.memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.budget.post_processing import (
    compute_lambda_and_stddev_from_results,
)


def _variate_ith_parameter_by(
    central_point: npt.NDArray[numpy.floating],
    variations: npt.NDArray[numpy.floating],
    i: int,
) -> Iterator[npt.NDArray[numpy.floating]]:
    """Returns versions of ``central_point`` where the ``i``-th parameter is
    successively replaced by values in ``variations``.

    Args:
        central_point (npt.NDArray[numpy.floating]): base 1-dimensional array of numbers
            of shape ``(n,)`` that will be copied, modified on the ``i``-th variable and
            returned.
        variations (npt.NDArray[numpy.floating]): 1-dimensional array of shape ``(m,)``
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
    x: npt.NDArray[numpy.floating],
    y: npt.NDArray[numpy.floating],
    stddevs: npt.NDArray[numpy.floating],
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
        _get_variance_of_gradient_estimation_at_point(cov, gradient_approximation_point)
    )
    return derivative, standard_deviation


def _get_variance_of_gradient_estimation_at_point(
    cov: npt.NDArray[numpy.floating], c: float
) -> float:
    """Get the variance of the gradient estimation at the point ``c`` for a polynomial
    with uncertainties on its coefficients provided by the covariance matrix ``cov``.

    Args:
        cov (npt.NDArray[numpy.floating]): an array of shape ``(d + 1, d + 1)``
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


def compute_1_over_lambda_gradient_at(
    noise_model_type: type[NoiseInterface],
    noise_model_parameters: npt.NDArray[numpy.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: list[tuple[float, float]],
    num_points_per_parameters: int = 10,
    num_shots: int = 10_000_000,
    batch_size: int = 10_000,
    memory_generator: MemoryGenerator = get_rotated_surface_code_memory_circuit,
    lep_target_rse: float = 1e-4,
    lep_computation_min_fails: int = 10,
    discretisation_generator: GradientFitDiscretisationGenerator = get_logarithmic_points,
    fitting_degree: int = 3,
    max_workers: int = 1,
    data_path: Path | None = None,
) -> tuple[npt.NDArray[numpy.floating], npt.NDArray[numpy.floating]]:
    """The gradient of 1 / Λ at the provided ``noise_model_parameters``.

    Args:
        noise_model_type (type[NoiseInterface]): type of the noise model to estimate the
            gradient of.
        noise_model_parameters (npt.NDArray[numpy.floating] | Sequence[float]): valid
            parameters to instantiate the type provided as ``noise_model_type``
            representing the point at which the gradient should be computed.
        num_rounds_by_distances (Mapping[int, Sequence[int]]): a mapping from each code
            distance that should be tested to the number of rounds that should be
            sampled in order to estimate the logical error-probability per round, to
            ultimately get 1 / Λ.
        noise_parameters_exploration_bounds (list[tuple[float, float]]): ``(min, max)``
            bounds for each noise parameter of the provided ``noise_model``. A degree
            ``fitting_degree`` polynomial will be fitted on the interval ``[min, max]``.
            The corresponding noise parameter from the provided ``noise_model`` should
            be strictly contained in ``[min, max]`` (i.e., for any valid ``i``, the
            following is true:
            ``noise_parameters_exploration_bounds[i][0] <
            noise_model.noise_parameters[i] <
            noise_parameters_exploration_bounds[i][1]``). Ideally, the lower (resp.
            upper) bound provided must be such that the logical error probability when
            replacing the parameter with its lower (resp. upper) bound is above
            ``100 / num_shots`` to ensure enough fails are observed with ``num_shots``
            shots (resp. below ``1 / 2`` to ensure that we can compute the logical error
            probability per round).
        num_points_per_parameters (int): number of different values to try for each
            noise parameter. Corresponds to the number of points that will be used to
            fit a degree ``fitting_degree`` polynomial. As such, should be greater than
            ``fitting_degree + 1``.
        num_shots (int): maximum number of shots per sampling task. A sampling task may
            stop with a lower number of samples if additional conditions are met, see
            ``lep_target_rse`` or ``lep_computation_min_fails`` for more details.
        batch_size (int): number of sampling experiments that are submitted per batch.
        memory_generator (MemoryGenerator): a callable that can generate a memory
            experiment. The resulting circuit will go through the provided
            ``noise_model`` for different values of the noise parameters.
        lep_target_rse (float): target relative standard error under which a sampling
            task is considered precise enough and can be stopped before ``num_shots``
            sampling tasks have returned.
        lep_computation_min_fails (int): minimum number of failures that should be
            witnessed before stopping a sampling task. A sampling task may stop with less
            failures, for example if ``num_shots`` shots have been performed.
        discretisation_generator (GradientFitDiscretisationGenerator): a callable
            generating points that can be used to compute 1 / Λ on different values and
            fit a degree ``fitting_degree`` polynomial. Default to logarithmically
            spaced points.
        fitting_degree (int): degree of polynomial that will be used to approximate
            1 / Λ and to compute each of its derivatives. Should be lower than
            ``num_points_per_parameters - 1``. Higher values will incur higher standard
            deviation. Default to 3, which seems to be a good compromise between fit
            accuracy and resulting standard deviation.
        max_workers (int): max number of parallel processes used by the function.
            Default to 1 which means fully sequential.
        data_file (Path | None): if provided, a valid path to which simulation data will
            be saved. Default to not provided, which means nothing is saved on disk.

    Returns:
        the error-budgeting result, which consists of an array of contributions for each
        of the noise parameters of the provided ``noise_model`` along with their
        associated standard deviations. Can also include the estimated value of Λ on the
        provided noise model parameter if ``include_lambda`` is ``True``.
    """

    # Checking inputs.
    if num_points_per_parameters % 2 == 1:
        warnings.warn(
            "Got an odd value for ``num_points_per_parameters``. This is sub-optimal "
            "as the central point will be included twice. Only using "
            f"{num_points_per_parameters - 1} points in the explicit discretisation, "
            "the central point will be included automatically."
        )
        num_points_per_parameters -= 1

    if num_points_per_parameters + 1 < fitting_degree + 2:
        raise ValueError(
            f"Estimation of the standard deviation requires at least "
            f"fitting_degree + 2 = {fitting_degree + 2} discretisation points, but "
            f"only {num_points_per_parameters} + 1 are provided. Please increase "
            f"num_points_per_parameters to at least {fitting_degree + 1}."
        )

    # Make sure that noise_model_parameters is a numpy array, even if a generic Sequence
    # is provided, as this is simpler for later.
    noise_model_parameters = numpy.asarray(noise_model_parameters)

    # Getting the points on which we will estimate 1 / Λ into ``noise_parameters``.
    # This is performing a sweeping for each parameter individually.
    central_point: npt.NDArray[numpy.floating] = noise_model_parameters.reshape((-1, 1))
    xis: list[npt.NDArray[numpy.floating]] = [central_point.reshape((-1,))]
    for i, (minimum, maximum) in enumerate(noise_parameters_exploration_bounds):
        variations = discretisation_generator(
            minimum,
            maximum,
            central_point[i],
            num_points_per_parameters,
            fitting_degree,
        )
        xis.extend(_variate_ith_parameter_by(central_point, variations, i))

    # Note that noise_parameters[:, 0] is always ``central_point``.
    noise_parameters = numpy.asarray(xis, dtype=numpy.floating).T

    if data_path is not None and data_path.exists():
        report = pandas.read_csv(data_path)
    else:
        # ``noise_parameters`` contains all the noise parameters we want to evaluate 1 / Λ.
        # Prepare the computation by building the decoder managers.
        decoder_managers = generate_decoder_managers_for_lambda(
            noise_parameters,
            noise_model_type,
            num_rounds_by_distances,
            max_workers,
            memory_generator=memory_generator,
        )

        # Start the computation
        num_points = noise_model_type.num_noise_parameters * num_points_per_parameters
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
        report.to_csv(data_path)

    # Post-process the results to get all the estimations for 1 / Λ
    lambdas, lambda_stddevs = compute_lambda_and_stddev_from_results(
        noise_parameters,
        noise_model_type.parameter_names,
        num_rounds_by_distances,
        report,
    )
    lambda_reciprocals = 1 / lambdas
    lambda_reciprocal_stddevs = numpy.abs(lambda_stddevs / lambdas**2)

    # We now have all the estimations of 1 / Λ, we can approximate the gradient
    # Note that ``noise_parameters``, ``lambda_reciprocals`` and
    # ``lambda_reciprocal_stddevs`` have the same shapes: a 2-dimensional array with
    # values as columns. Additionally, the first column corresponds to the exact point
    # at which we want the gradient and each following group of
    # ``num_points_per_parameters`` columns correspond to the variation of one
    # parameter.
    gradient: list[float] = []
    gradient_stddev: list[float] = []
    for npi, noise_parameter in enumerate(central_point):
        start = 1 + num_points_per_parameters * npi
        end = 1 + num_points_per_parameters * (npi + 1)
        # Index 0 is ``central_point``, so it can be included in all estimations.
        column_indices = [0, *list(range(start, end))]
        x = noise_parameters[npi, column_indices]
        y = lambda_reciprocals[0, column_indices]
        stddevs = lambda_reciprocal_stddevs[0, column_indices]
        derivative, derivative_stddev = _approximate_derivative_at_point_from_values(
            x,
            y,
            stddevs,
            noise_parameter,
            degree=fitting_degree,
            noise_name=noise_model_type.parameter_names[npi],
        )
        gradient.append(derivative)
        gradient_stddev.append(derivative_stddev)

    return numpy.asarray(gradient), numpy.asarray(gradient_stddev)
