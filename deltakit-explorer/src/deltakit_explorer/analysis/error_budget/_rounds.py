from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from deltakit_circuit._circuit import Circuit
from deltakit_decode._mwpm_decoder import PyMatchingDecoder
from deltakit_decode.analysis._matching_decoder_managers import StimDecoderManager

from deltakit_explorer.analysis import (
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from deltakit_explorer.analysis._analysis import calculate_lep_and_lep_stddev
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)


def compute_ideal_rounds_for_noise_model_and_distance(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
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
    using a memory experiment as returned by the provided ``memory_generator``.

    Args:
        noise_model (Callable[[Circuit, npt.NDArray[np.floating]], Circuit]): a callable
            adding noise to the provided circuit, according to the parameters provided.
        noise_parameters (npt.NDArray[numpy.floating] | Sequence[float]): valid
            parameters to forward to ``noise_model`` representing the point at which the
            ideal number of rounds should be computed.
        distance (int): code distance for which we want to have the ideal numbers of
            rounds to estimate the logical error probability per round.
        max_shots (int): maximum number of shots performed by the simulations in this
            function. Simulations might perform less shots due to other conditions being
            met, for example a low-enough standard deviation according to
            ``target_stddev``.
        batch_size (int): number of shots to perform per batch. Early-stopping conditions
            are checked after each chunks of ``batch_size`` shots.
        initial_round_number (int): number of rounds to start the exploration with.
            Should be strictly positive. Should likely not be ``1`` because data for
            ``1`` round is often an outlier. Only set this to ``1`` if you understand
            what you are doing and really want it.
        min_fails (int): minimum number of fails that should be observed to be able to
            early-return before ``max_shots`` shots have been performed.
        target_stddev (float): if the standard deviation of logical error probability
            estimation is below that threshold, simulation might early exit.
        max_round_number (int): maximum number of rounds that should be tested.
        next_round_number_func (Callable[[int], int]): an arbitrary callable that should
            return the next number of rounds to simulate from the previous one. It
            effectively describes the rounds that will be tested until the number of
            rounds is high enough to output a logical error probability over ``0.2`` or
            to exceed ``max_round_number``. Default to a geometric progression with a
            multiplicative factor of ``4``, but can be anything provided that it is
            strictly increasing.
        memory_generator (MemoryGenerator): a callable returning a memory experiment
            from provided distance and number of rounds.

    Returns:
        a list containing number of rounds resulting from successive applications of
        ``next_round_number_func`` starting with ``initial_round_number`` (let
        ``f = next_round_number_func`` and ``x = initial_round_number``, the returned
        list will be ``[x, f(x), f(f(x)), ..., y])``) such that ``y``, the last value of
        that list, is below ``max_round_number`` and ideally lead to a logical error
        probability for ``y`` rounds that is above the ``0.2`` heuristic threshold.
    """

    # Ensure that noise_parameters is a numpy array.
    noise_parameters = np.asarray(noise_parameters)

    def generate_surface_code_memory_and_run(
        num_rounds: int,
    ) -> tuple[int, int]:
        circuit = memory_generator(distance, num_rounds)
        noisy_circuit = noise_model(circuit, noise_parameters)
        decoder, decoder_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(
            noisy_circuit
        )
        decoder_manager = StimDecoderManager(decoder_circuit, decoder)

        nshots, nfails = decoder_manager.run_batch_shots(batch_size)
        _, stddev = calculate_lep_and_lep_stddev(nfails, nshots)
        while (stddev > target_stddev or nfails < min_fails) and nshots < max_shots:
            nshots_to_perform = min(batch_size, max_shots - nshots)
            ns, nf = decoder_manager.run_batch_shots(nshots_to_perform)
            nshots += ns
            nfails += nf
            _, stddev = calculate_lep_and_lep_stddev(nfails, nshots)
        return nfails, nshots

    nrounds, *_ = simulate_different_round_numbers_for_lep_per_round_estimation(
        simulator=generate_surface_code_memory_and_run,
        heuristic_logical_error_lower_bound=0.2,
        next_round_number_func=next_round_number_func,
        initial_round_number=initial_round_number,
        maximum_round_number=max_round_number,
    )
    return nrounds.tolist()
