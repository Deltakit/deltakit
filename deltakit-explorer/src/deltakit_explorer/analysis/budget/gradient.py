from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Mapping, Type

from deltakit_explorer.analysis.budget.memory import MemoryGenerator, get_rotated_surface_code_memory_circuit
import numpy
import numpy.typing as npt
from deltakit_decode._mwpm_decoder import PyMatchingDecoder
from deltakit_decode.analysis._matching_decoder_managers import StimDecoderManager
from deltakit_decode.analysis._run_all_analysis_engine import RunAllAnalysisEngine
from scipy.differentiate import derivative


from deltakit_explorer.analysis._analysis import (
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from deltakit_explorer.analysis.budget.generation import (
    generate_decoder_managers_for_lambda
)
from deltakit_explorer.analysis.budget.interfaces import NoiseInterface
from deltakit_explorer.analysis.budget.post_processing import (
    compute_lambda_and_stddev_from_results,
)


def vectorised_lambda_reciprocal(
    xi: npt.NDArray[numpy.float64],
    noise_model_type: Type[NoiseInterface],
    num_rounds_by_distance: Mapping[int, Sequence[int]],
    num_shots: int = 100_000,
    batch_size: int = 10_000,
    target_rse: float = 1e-4,
    data_directory: Path | None = None,
    max_workers: int = 1,
) -> tuple[npt.NDArray[numpy.float64], npt.NDArray[numpy.float64]]:
    """Compute the value of ``1 / Λ`` in a vectorised manner.

    This function sole purpose is to be compatible with the interface needed by
    ``scipy.differentiate.derivative``.

    Arguments:
        xi (npt.NDArray[numpy.float64]): an array of shape ``(m, ...)`` where
            ``m == noise_model.num_noise_parameters``. The first axis represents the
            ``m`` inputs of this function.
        noise_model_type (Type[NoiseInterface]): noise model type to use.
        distances (Sequence[int]): different code distance at which to evaluate the
            logical error probability per round in order to estimate Λ.
        num_rounds (Mapping[int, Sequence[int]]): a mapping from each entry in
            ``distance`` to the different number of rounds at which to evaluate the
            logical error probability in order to estimate the logical error probability
            per round.
        max_workers (int): maximum number of parallel processes used to compute the
            function. Defaults to ``1`` which makes the function serial.

    Returns:
        two arrays of shape ``(1, ...)`` in which the first dimension represents the
        estimated value of ``1 / Λ`` (resp. the standard deviation associated with this
        estimation) and all the other dimensions are like what has been provided in
        ``xi``.

    Raises:
        ValueError: if ``xi.shape[0] != noise_model.num_noise_parameters``.
    """
    if xi.shape[0] != noise_model_type.num_noise_parameters:
        raise ValueError(
            f"Got {xi.shape[0]}-dimensional data in `xi` but the noise model provided "
            f"has {noise_model_type.num_noise_parameters} noise parameters."
        )
    m = xi.shape[0]
    ret_shape: tuple[int, ...] = (1,) + xi.shape[1:]

    # The following computation are costly. Just in case there are some duplicated
    # entries in the provided xi, only compute them once.
    xi = xi.reshape((m, -1))
    unique_xi, inverse_indices = numpy.unique(xi, return_inverse=True, axis=1)
    # 1. Generate the quantum circuits we should evaluate.
    decoder_managers = generate_decoder_managers_for_lambda(
        unique_xi, noise_model_type, num_rounds_by_distance, max_workers
    )
    # 2. Run the simulations
    engine = RunAllAnalysisEngine(
        experiment_name=f"Estimating Λ on {numpy.prod(ret_shape)} points",
        decoder_managers=decoder_managers,
        max_shots=num_shots,
        batch_size=batch_size,
        # Early stopping: we got a low-enough standard deviation
        loop_condition=RunAllAnalysisEngine.loop_until_observable_rse_below_threshold(
            target_rse, 100
        ),
        num_parallel_processes=max_workers,
        data_directory=data_directory,
    )
    report = engine.run()
    # 3. Compute and return
    lambdas, stddevs = compute_lambda_and_stddev_from_results(
        unique_xi, num_rounds_by_distance, report, return_reporters=True
    )
    # Un-uniquify
    lambdas = lambdas[:, inverse_indices].reshape(ret_shape)
    stddevs = stddevs[:, inverse_indices].reshape(ret_shape)

    assert lambdas.shape == ret_shape
    print(xi)
    print(lambdas)
    print(lambdas.flatten())
    return 1 / lambdas, numpy.abs(stddevs / lambdas**2)


def compute_ideal_rounds_for_noise_model_and_distance(
    noise_model: NoiseInterface,
    distance: int,
    max_shots: int,
    batch_size: int,
    initial_round_number: int = 2,
    min_fails: int = 100,
    target_stddev: float = 1e-4,
    max_round_number: int = 1024,
    next_round_number_func: Callable[[int], int] = lambda x: 4 * x,
    memory_generator: MemoryGenerator = get_rotated_surface_code_memory_circuit,
) -> list[int]:
    """Compute the ideal rounds to use to estimate the LEP per round.

    This function tries to efficiently find the ideal values for the number of rounds to
    use in order to estimate the logical error probability per round. It essentially
    wraps :func:`simulate_different_round_numbers_for_lep_per_round_estimation`,
    using a memory experiment with the rotated surface code.
    """
    def generate_surface_code_memory_and_run(
        num_rounds: int,
    ) -> tuple[int, int]:
        circuit = memory_generator(distance, num_rounds)
        noisy_circuit = noise_model.apply(circuit)
        decoder, decoder_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(
            noisy_circuit
        )
        decoder_manager = StimDecoderManager(decoder_circuit, decoder)

        nshots, nfails = decoder_manager.run_batch_shots(batch_size)
        lep = nfails / nshots
        stddev = lep * (1 - lep) / nshots
        while (stddev > target_stddev or nfails < min_fails) and nshots < max_shots:
            ns, nf = decoder_manager.run_batch_shots(
                min(batch_size, max_shots - nshots)
            )
            nshots += ns
            nfails += nf
            lep = nfails / nshots
            stddev = lep * (1 - lep) / nshots

        # print(f"{stddev} > {target_stddev} and {nshots} < {max_shots}")
        # print(stddev > target_stddev, nshots < max_shots)
        print(
            f"    LEP for {num_rounds:>4} rounds in {nshots:>6} shots: {lep:.4g} +/- {stddev:.4g}"
        )
        return nfails, nshots

    nrounds, *_ = simulate_different_round_numbers_for_lep_per_round_estimation(
        simulator=generate_surface_code_memory_and_run,
        heuristic_logical_error_lower_bound=0.2,
        next_round_number_func=next_round_number_func,
        initial_round_number=initial_round_number,
        maximum_round_number=max_round_number,
    )
    return nrounds.tolist()


def get_lambda_reciprocal_gradient(
    xi: npt.NDArray[numpy.float64],
    noise_model_type: Type[NoiseInterface],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    absolute_maximum_xi_steps: npt.NDArray[numpy.float64] | None = None,
    max_shots: int = 1_000_000,
    batch_size: int = 10_000,
    target_rse: float = 1e-3,
    data_directory: Path | None = None,
    max_workers: int = 1,
) -> tuple[ npt.NDArray[numpy.float64], npt.NDArray[numpy.float64]]:
    """Approximates ∇(1/Λ) at the provided ``xi``.

    This function approximates the gradient of 1/Λ with respect to each noise parameter
    in the provided ``noise_model_type`` by using the provided sets of ``distances`` and
    ``num_rounds`` to approximate Λ.
    """
    # We start by correctly approximating Λ with the provided noise parameters. This
    # step is mainly here to devise the correct numbers of rounds that should be used.
    if absolute_maximum_xi_steps is None:
        absolute_maximum_xi_steps = 0.1 * xi
    gradient = numpy.zeros(
        (1, noise_model_type.num_noise_parameters), dtype=numpy.float64
    )
    errors = numpy.zeros_like(gradient)
    for npi, noise_name in enumerate(noise_model_type.parameter_names):
        all_xs: list[float] = []

        def f(x: npt.NDArray[numpy.float64]) -> npt.NDArray[numpy.float64]:
            input_shape = x.shape
            x = numpy.atleast_1d(x)
            xis_shape = (xi.size,) + tuple(1 for _ in x.shape)
            xis = numpy.tile(xi.reshape(xis_shape), (1, *x.shape))
            xis[npi, :] = x
            lambda_, stddevs = vectorised_lambda_reciprocal(
                xis,
                noise_model_type,
                num_rounds_by_distances,
                max_shots,
                batch_size,
                target_rse,
                data_directory,
                max_workers,
            )
            lambda_ = lambda_.reshape(input_shape)
            stddevs = stddevs.reshape(input_shape)
            all_xs.extend(x.flatten().tolist())
            return lambda_

        # Estimating the derivative for each noise parameter separately.
        print(f"Estimating the gradient for {npi + 1}-th parameter '{noise_name}'.")
        res = derivative(
            f,
            xi[npi],
            order=4,
            initial_step=absolute_maximum_xi_steps[npi],
            callback=print,
        )
        gradient[0, npi], errors[0, npi] = res.df, res.error
        print(f"Estimated {res.df:.5g} +/- {res.error:.3g}")
    return gradient, errors


def get_lambda_reciprocal_values(
    xis: npt.NDArray[numpy.float64],
    noise_model_type: Type[NoiseInterface],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    max_shots: int = 10_000_000,
    batch_size: int = 10_000,
    target_rse: float = 1e-3,
    data_directory: Path | None = None,
    max_workers: int = 1,
) -> tuple[npt.NDArray[numpy.float64], npt.NDArray[numpy.float64]]:
    """Approximates 1/Λ at the provided ``xis``."""
    # We start by correctly approximating Λ with the provided noise parameters. This
    # step is mainly here to devise the correct numbers of rounds that should be used.
    lambda_, stddevs = vectorised_lambda_reciprocal(
        xis,
        noise_model_type,
        num_rounds_by_distances,
        max_shots,
        batch_size,
        target_rse,
        data_directory,
        max_workers,
    )
    return lambda_, stddevs
