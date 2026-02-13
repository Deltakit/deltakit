import math
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from deltakit_circuit._circuit import Circuit
from deltakit_decode.analysis._run_all_analysis_engine import RunAllAnalysisEngine

from deltakit_explorer.analysis.error_budget._generation import (
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    PreComputedMemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.error_budget._parameters import (
    FittingParameters,
    SamplingParameters,
)
from deltakit_explorer.analysis.error_budget._post_processing import (
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
    values in ``x`` to affect the gradient too much.

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
    # have an easy formula for the variance involving the covariance matrix.
    n = cov.shape[0]
    coeff_matrix = np.array(
        [[(i + 1) * (j + 1) * c ** (i + j) for i in range(n - 1)] for j in range(n - 1)]
    )
    return float(np.sum(coeff_matrix * cov[1:, 1:]))


def generate_sweep_parameters(
    central_point: npt.NDArray[np.floating],
    noise_parameters_exploration_bounds: list[tuple[float, float]],
    fitting_parameters: FittingParameters = FittingParameters(),
) -> npt.NDArray[np.floating]:
    """Generate multiple copies of the provided ``central_point`` with one parameter changed.

    Each parameter is changed according to the fitting strategy. The resulting array will contain
    copies of ``central_point`` where, for each copy, only a single entry is different from
    ``central_point``. The value of that entry, and the number of variations, depends on the
    provided ``fitting_parameters``.

    Args:
        central_point: point that should be copied and modified in the resulting array.
        noise_parameters_exploration_bounds: bounds for each parameters, that will impact the values
            that are inserted in the returned array.
        fitting_parameters: parameters of the fitting task that will control how many variations are
            inserted for each parameters and how the values of these variations will be obtained.

    Returns:
        An array of shape
        ``(fitting_parameters.num_points_per_parameters * central_point.size, central_point.size)``
        that contains as rows variations of ``central_point`` where a single parameter has been
        changed each time.
    """
    # Getting the points on which we will estimate 1 / Λ into ``noise_parameters``.
    # This is performing a sweeping for each parameter individually.
    xis: list[npt.NDArray[np.floating]] = [central_point.reshape((-1,))]
    for i, (minimum, maximum) in enumerate(noise_parameters_exploration_bounds):
        variations = fitting_parameters.get_discretisation(
            minimum, maximum, central_point[i]
        )
        xis.extend(_variate_ith_parameter_by(central_point, variations, i))

    return np.asarray(xis).T


def get_decoding_result(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    sweep_noise_parameters: npt.NDArray[np.floating],
    noise_parameter_names: list[str],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator = get_rotated_surface_code_memory_circuit,
) -> pd.DataFrame:
    """Construct, sample and decode the experiments represented by the input parameters
    and returns statistics on the success rate of each experiment.

    Experiments to run are given by the provided ``noise_model``,
    ``sweep_noise_parameters``, ``num_rounds_by_distances``, ``memory_generator``.

    Args:
        noise_model: noise model to use to annotate noise instructions on the circuits.
        sweep_noise_parameters: parameters to use for the noise model. Should be the array
            returned when calling ``generate_sweep_parameters``.
        noise_parameter_names: identifiers for each noise parameter, used to identify
            results in the returned statistics.
        num_rounds_by_distances (Mapping[int, Sequence[int]]): a mapping from each code
            distance that should be tested to the number of rounds that should be
            sampled in order to estimate the logical error-probability per round, to
            ultimately get 1 / Λ.
        fitting_parameters: additional parameters relating to how the gradient is
            estimated.
        sampling_parameters: additional parameters relating to the sampling tasks used to
            estimate 1 / Λ indirectly.
        memory_generator: a callable that can generate a memory experiment. The resulting
            circuit will go through the provided ``noise_model`` for different values of
            the noise parameters.

    Returns:
        A pandas DataFrame containing the statistics resulting from sampling the provided
        experiments.
    """
    # ``noise_parameters`` contains all the noise parameters we want to evaluate 1 / Λ.
    # Prepare the computation by building the decoder managers.
    decoder_managers = generate_decoder_managers_for_lambda(
        sweep_noise_parameters,
        noise_model,
        num_rounds_by_distances,
        sampling_parameters.max_workers,
        memory_generator=memory_generator,
        noise_parameter_names=noise_parameter_names,
    )

    # Start the computation
    num_parameters = sweep_noise_parameters.shape[0]
    num_points = num_parameters * fitting_parameters.num_points_per_parameters
    engine = RunAllAnalysisEngine(
        experiment_name=f"Estimating Λ on {num_points} points",
        decoder_managers=decoder_managers,
        max_shots=sampling_parameters.max_shots,
        batch_size=sampling_parameters.batch_size,
        # Early stopping when we have a low-enough standard deviation
        loop_condition=RunAllAnalysisEngine.loop_until_observable_rse_below_threshold(
            sampling_parameters.lep_target_rse,
            sampling_parameters.lep_computation_min_fails,
        ),
        num_parallel_processes=sampling_parameters.max_workers,
    )
    return engine.run()


def get_lambda_gradient(
    report: pd.DataFrame,
    central_point: npt.NDArray[np.floating],
    sweep_noise_parameters: npt.NDArray[np.floating],
    noise_parameter_names: list[str],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    fitting_parameters: FittingParameters = FittingParameters(),
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Post-process the provided ``report`` to compute an estimation of the gradient of 1 / Λ.

    Args:
        report: statistics returned by ``get_decoding_results``.
        central_point: point at which the gradient of 1 / Λ should be estimated.
        sweep_noise_parameters: parameters to use for the noise model. Should be the array
            returned when calling ``generate_sweep_parameters``.
        noise_parameter_names: identifiers for each noise parameter, used to identify
            results in the returned statistics.
        num_rounds_by_distances (Mapping[int, Sequence[int]]): a mapping from each code
            distance that should be tested to the number of rounds that should be
            sampled in order to estimate the logical error-probability per round, to
            ultimately get 1 / Λ.
        fitting_parameters: additional parameters relating to how the gradient is
            estimated.

    Returns:
        a tuple ``(gradient, stddev)`` containing the estimation of the gradient at the
        provided ``central_point`` and the standard deviation of the estimation.
    """
    # Post-process the results to get all the estimations for 1 / Λ
    lambdas, lambda_stddevs = compute_lambda_and_stddev_from_results(
        sweep_noise_parameters, noise_parameter_names, num_rounds_by_distances, report
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
        start = 1 + fitting_parameters.num_points_per_parameters * npi
        end = 1 + fitting_parameters.num_points_per_parameters * (npi + 1)
        # Index 0 is ``central_point``, so it can be included in all estimations.
        column_indices = [0, *list(range(start, end))]
        x = sweep_noise_parameters[npi, column_indices]
        y = lambda_reciprocals[0, column_indices]
        stddevs = lambda_reciprocal_stddevs[0, column_indices]
        derivative, derivative_stddev = _approximate_derivative_at_point_from_values(
            x, y, stddevs, noise_parameter, degree=fitting_parameters.fitting_degree
        )
        gradient.append(derivative)
        gradient_stddev.append(derivative_stddev)
    return np.asarray(gradient), np.asarray(gradient_stddev)


def inverse_lambda_gradient_at(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: list[tuple[float, float]],
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
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
            ``100 / max_shots`` to ensure enough fails are observed with ``max_shots``
            shots (resp. below ``1 / 2`` to ensure that we can compute the logical error
            probability per round).
        fitting_parameters: additional parameters relating to how the gradient is
            estimated.
        sampling_parameters: additional parameters relating to the sampling tasks used to
            estimate 1 / Λ indirectly.
        memory_generator (MemoryGenerator): a callable that can generate a memory
            experiment. The resulting circuit will go through the provided
            ``noise_model`` for different values of the noise parameters.


    Returns:
        the error-budgeting result, which consists of an array of contributions for each
        of the noise parameters of the provided ``noise_model`` along with their
        associated standard deviations. Can also include the estimated value of Λ on the
        provided noise model parameter if ``include_lambda`` is ``True``.
    """

    # Making sure that the memory generator is an object implementing the MemoryGenerator
    # protocol.
    if isinstance(memory_generator, Mapping):
        memory_generator = PreComputedMemoryGenerator(memory_generator)

    # Make sure that noise_model_parameters is a numpy array, even if a generic Sequence
    # is provided, as this is simpler for later.
    noise_model_parameters = np.asarray(noise_parameters)
    # Create unique identifiers for noise parameters that will be used to discriminate between them
    # in the CSV file storing the simulation results.
    noise_parameter_names = [str(i) for i in range(len(noise_parameters))]

    central_point: npt.NDArray[np.floating] = noise_model_parameters.reshape((-1, 1))
    # Generate the noise parameters at which we will compute Λ.
    sweep_noise_parameters = generate_sweep_parameters(
        central_point, noise_parameters_exploration_bounds, fitting_parameters
    )
    # Sample on these noise parameters.
    report = get_decoding_result(
        noise_model,
        sweep_noise_parameters,
        noise_parameter_names,
        num_rounds_by_distances,
        fitting_parameters,
        sampling_parameters,
        memory_generator,
    )
    # Compute the gradient of Λ from the sampling results.
    return get_lambda_gradient(
        report,
        central_point,
        sweep_noise_parameters,
        noise_parameter_names,
        num_rounds_by_distances,
        fitting_parameters,
    )
