from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from deltakit_explorer.analysis.budget._discretisation import (
    GradientFitDiscretisationGenerator,
    get_logarithmic_points,
)
from deltakit_explorer.analysis.budget._gradient import (
    compute_1_over_lambda_gradient_at,
)
from deltakit_explorer.analysis.budget._interfaces import NoiseInterface
from deltakit_explorer.analysis.budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)


def get_error_budget(
    noise_model_type: type[NoiseInterface],
    noise_model_parameters: npt.NDArray[np.floating] | Sequence[float],
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
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the error budget of the provided ``noise_model``.

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

    Returns:
        the error-budgeting result, which consists of an array of contributions for each
        of the noise parameters of the provided ``noise_model`` along with their
        associated standard deviations.
    """
    # We will compute the gradient at the half point following the methodology outlined
    # in "Exponential suppression of bit or phase errors with cyclic error correction".
    point = np.asarray(noise_model_parameters) / 2
    # Evaluate the gradient.
    gradient, gradient_stddev = compute_1_over_lambda_gradient_at(
        noise_model_type,
        point,
        num_rounds_by_distances,
        noise_parameters_exploration_bounds,
        num_points_per_parameters,
        num_shots,
        batch_size,
        memory_generator,
        lep_target_rse,
        lep_computation_min_fails,
        discretisation_generator,
        fitting_degree,
        max_workers,
    )
    # We computed the gradient at the point ``x / 2``, we can now apply it to the
    # original noise parameters to recover an estimate.
    contributions = np.abs(gradient * noise_model_parameters)
    stddevs = np.abs(gradient_stddev * noise_model_parameters)

    return contributions, stddevs
