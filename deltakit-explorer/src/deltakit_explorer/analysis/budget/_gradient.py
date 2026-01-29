import math
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from deltakit_circuit._circuit import Circuit
from deltakit_decode.analysis._run_all_analysis_engine import RunAllAnalysisEngine

from deltakit_explorer.analysis.budget._discretisation import (
    GradientFitDiscretisationEnum,
)
from deltakit_explorer.analysis.budget._generation import (
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.budget._memory import (
    MemoryGenerator,
    PreComputedMemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.budget._post_processing import (
    compute_lambda_and_stddev_from_results,
)


def _variate_ith_parameter_by(
    central_point: npt.NDArray[np.floating],
    variations: npt.NDArray[np.floating],
    i: int,
) -> Iterator[npt.NDArray[np.floating]]:
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
    parameters = np.tile(central_point, (1, variations.size))
    parameters[i, :] = variations
    yield from parameters.T


def _approximate_derivative_at_point_from_values(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    stddevs: npt.NDArray[np.floating],
    gradient_approximation_point: float,
    degree: int = 3,
) -> tuple[float, float]:
    """Approximate the gradient at ``gradient_approximation_point`` from the given ``x``
    and ``y``.

    This function fits a degree ``degree`` polynomial to the points given by ``x``,
    ``y`` and ``stddevs`` (the standard deviation of each point in ``y``) and then
    computes the gradient of the fitted polynomial at ``gradient_approximation_points``.

    This algorithm is used to use as much as possible the standard deviation information
    and to avoid non-linear behaviour at the extremities of the interval containing all
    values in ``x`` to affect too much the gradient.

    Warning:
        This function will work better if ``gradient_approximation_point`` is close to
        the "center" of the interval formed by ``x``.

        This is due to the fact that the gradient at ``gradient_approximation_point`` is
        estimated through a polynomial of degree ``degree`` fitted on the provided
        ``x``, ``y`` and ``stddevs``, which inherently suffer from the Runge's
        phenomenon (even with the optimal fitting points for ``x``, see
        https://en.wikipedia.org/wiki/Runge%27s_phenomenon). In this case, Runge's
        phenomenon means that the furthest ``gradient_approximation_point`` will be from
        the "center" of the discretisation interval provided by ``x``, the more likely
        our gradient estimation is impacted significantly by the oscillations.

        It is considered to be the responsibility of the caller to check that the
        provided ``x`` and ``gradient_approximation_point`` are picked such that Runge's
        phenomenon will not impact our estimation too much.

    Args:
        x (npt.NDArray[numpy.floating]): exact values on which we evaluated a noisy
            function. Should be a 1-dimensional array.
        y (npt.NDArray[numpy.floating]): best estimation of the result obtained from the
            noisy function evaluation when evaluated on the corresponding entry in
            ``x``. Should be a 1-dimensional array.
        stddevs (npt.NDArray[numpy.floating]): standard deviation of the estimate in
            ``y``. Should be a 1-dimensional array.
        gradient_approximation_point (float): point at which the gradient should be
            estimated.
        degree (int): degree of the polynomial to fit the provided points and estimate
            the gradient at ``gradient_approximation_point``.

    Returns:
        value of the gradient (a single float because ``x``, ``y`` and ``stddevs`` are
        one-dimensional arrays) and the standard deviation of the estimate.
    """
    # Perform the approximation using the provided standard deviations
    coefficients, cov = np.polyfit(x, y, deg=degree, cov="unscaled", w=1 / stddevs)

    # Flipping the coefficients and covariance matrix to have the index corresponding to
    # the degree (i.e., ``coefficients[i]`` is multiplying ``x**i`` in the polynomial
    # and ``cov[i,i]`` is the variance of ``coefficients[i]``).
    coefficients, cov = np.flip(coefficients), np.flip(cov)

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
    cov: npt.NDArray[np.floating], c: float
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
    derivative_cov = np.copy(cov[1:, 1:])
    num_derivative_coefficients = derivative_cov.shape[0]
    # The original polynomial at ``x`` is given by
    #       sum(a[i] * x**i for i in range(0, n+1))
    # Its derivative is then given by
    #       sum(a[i+1] * (i+1) * x**i for i in range(0, n))
    # Removing a[0] from a (and so shifting its indexing by 1, which is what we just did
    # with the covariance matrix above) we have the derivative given by
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
    return float(np.sum(derivative_cov))


def inverse_lambda_gradient_at(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: list[tuple[float, float]],
    num_points_per_parameters: int = 10,
    num_shots: int = 10_000_000,
    batch_size: int = 10_000,
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
    lep_target_rse: float = 1e-4,
    lep_computation_min_fails: int = 10,
    discretisation_generator: GradientFitDiscretisationEnum = GradientFitDiscretisationEnum.LINEAR,
    fitting_degree: int = 3,
    max_workers: int = 1,
    noise_parameter_names: Sequence[str] | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """The gradient of 1 / Λ at the provided ``noise_model_parameters``.

    Args:
        noise_model (Callable[[Circuit, npt.NDArray[np.floating]], Circuit]): a callable
            adding noise to the provided circuit, according to the parameters provided.
        noise_parameters (npt.NDArray[numpy.floating] | Sequence[float]): valid
            parameters to forward to ``noise_model`` representing the point at which the
            gradient should be computed.
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
        noise_parameter_names: if provided, human-readable names for each of the
            provided ``noise_parameters``. Defaults to the noise parameter index (i.e.,
            "0", "1", ...).

    Returns:
        the error-budgeting result, which consists of an array of contributions for each
        of the noise parameters of the provided ``noise_model`` along with their
        associated standard deviations. Can also include the estimated value of Λ on the
        provided noise model parameter if ``include_lambda`` is ``True``.
    """

    # Checking inputs.
    if num_points_per_parameters + 1 < fitting_degree + 2:
        msg = (
            f"Estimation of the standard deviation requires at least "
            f"fitting_degree + 2 = {fitting_degree + 2} discretisation points, but "
            f"only {num_points_per_parameters} + 1 are provided. Please increase "
            f"num_points_per_parameters to at least {fitting_degree + 1}."
        )
        raise ValueError(msg)

    if isinstance(memory_generator, Mapping):
        memory_generator = PreComputedMemoryGenerator(memory_generator)

    if noise_parameter_names is None:
        noise_parameter_names = [str(i) for i in range(len(noise_parameters))]
    # Make sure that noise_model_parameters is a numpy array, even if a generic Sequence
    # is provided, as this is simpler for later.
    noise_model_parameters = np.asarray(noise_parameters)

    # Getting the points on which we will estimate 1 / Λ into ``noise_parameters``.
    # This is performing a sweeping for each parameter individually.
    central_point: npt.NDArray[np.floating] = noise_model_parameters.reshape((-1, 1))
    xis: list[npt.NDArray[np.floating]] = [central_point.reshape((-1,))]
    for i, (minimum, maximum) in enumerate(noise_parameters_exploration_bounds):
        variations = discretisation_generator(
            minimum,
            maximum,
            central_point[i],
            num_points_per_parameters,
            fitting_degree,
        )
        xis.extend(_variate_ith_parameter_by(central_point, variations, i))

    # Note: noise_parameters[:, 0] is always ``central_point``.
    noise_parameters = np.asarray(xis).T

    # ``noise_parameters`` contains all the noise parameters we want to evaluate 1 / Λ.
    # Prepare the computation by building the decoder managers.
    decoder_managers = generate_decoder_managers_for_lambda(
        noise_parameters,
        noise_model,
        num_rounds_by_distances,
        max_workers,
        memory_generator=memory_generator,
        noise_parameter_names=noise_parameter_names,
    )

    # Start the computation
    num_points = noise_model_parameters.size * num_points_per_parameters
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

    # Post-process the results to get all the estimations for 1 / Λ
    lambdas, lambda_stddevs = compute_lambda_and_stddev_from_results(
        noise_parameters, noise_parameter_names, num_rounds_by_distances, report
    )
    lambda_reciprocals = 1 / lambdas
    lambda_reciprocal_stddevs = np.abs(lambda_stddevs / lambdas**2)

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
            x, y, stddevs, noise_parameter, degree=fitting_degree
        )
        gradient.append(derivative)
        gradient_stddev.append(derivative_stddev)

    return np.asarray(gradient), np.asarray(gradient_stddev)
