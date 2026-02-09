from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from deltakit_circuit._circuit import Circuit
from deltakit_decode.analysis import RunAllAnalysisEngine

from deltakit_explorer.analysis.error_budget._generation import (
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    PreComputedMemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.error_budget._post_processing import (
    compute_lambda_and_stddev_from_results,
)


def inverse_lambda_at(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    num_shots: int = 10_000_000,
    batch_size: int = 10_000,
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
    lep_target_rse: float = 1e-4,
    lep_computation_min_fails: int = 10,
    max_workers: int = 1,
    noise_parameter_names: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Compute 1 / Λ.

    Warning:
        This is a helper function to compute 1 / Λ when you need a **single**
        evaluation.
        For error budgeting, :func:`~deltakit_explorer.analysis.error_budget.get_error_budget`
        will be able to parallelise more efficiently, while also performing several
        checks and optimisations.

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
            witnessed before stopping a sampling task. A sampling task may stop with
            less failures, for example if ``num_shots`` shots have been performed.
        max_workers (int): max number of parallel processes used by the function.
            Default to 1 which means fully sequential.
        noise_parameter_names: if provided, human-readable names for each of the
            provided ``noise_parameters``. Defaults to the noise parameter index (i.e.,
            "0", "1", ...).

    Returns:
        the estimation of 1 / Λ along with the standard deviation of the estimation as
        a 2-tuple.
    """
    if isinstance(memory_generator, Mapping):
        memory_generator = PreComputedMemoryGenerator(memory_generator)

    point = np.asarray(noise_parameters).reshape((-1, 1))

    if noise_parameter_names is None:
        noise_parameter_names = [str(i) for i in range(point.size)]

    decoder_managers = generate_decoder_managers_for_lambda(
        point,
        noise_model,
        num_rounds_by_distances,
        max_workers,
        memory_generator=memory_generator,
        noise_parameter_names=noise_parameter_names,
    )
    engine = RunAllAnalysisEngine(
        experiment_name="Estimating 1 / Λ",
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
    lambdas, lambda_stddevs = compute_lambda_and_stddev_from_results(
        point, noise_parameter_names, num_rounds_by_distances, report
    )
    lambda_reciprocals = 1 / lambdas
    lambda_reciprocal_stddevs = np.abs(lambda_stddevs / lambdas**2)

    return float(lambda_reciprocals[0, 0]), float(lambda_reciprocal_stddevs[0, 0])
