from math import sqrt
from typing import Callable

from deltakit_explorer.analysis.budget.memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_decode._mwpm_decoder import PyMatchingDecoder
from deltakit_decode.analysis._matching_decoder_managers import StimDecoderManager


from deltakit_explorer.analysis._analysis import (
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from deltakit_explorer.analysis.budget.interfaces import NoiseInterface


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
        stddev = sqrt(lep * (1 - lep) / nshots)
        while (stddev > target_stddev or nfails < min_fails) and nshots < max_shots:
            ns, nf = decoder_manager.run_batch_shots(
                min(batch_size, max_shots - nshots)
            )
            nshots += ns
            nfails += nf
            lep = nfails / nshots
            stddev = sqrt(lep * (1 - lep) / nshots)

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
