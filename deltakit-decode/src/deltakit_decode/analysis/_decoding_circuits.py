# (c) Copyright Riverlane 2020-2025.

import numpy as np
import stim
from tqdm import tqdm

from deltakit_decode._abstract_matching_decoders import DecoderProtocol


def run_decoding_on_circuit(
    circuit: stim.Circuit,
    max_shots: int,
    decoder: DecoderProtocol,
    max_batch_size: int = 10_000,
    target_rse: float | None = None,
    min_fails: int = 10
) -> dict[str, int]:
    """Compute LEP of the decoder with given circuit.
    The function samples shots in batches and feeds them to decoders'
    decode_batch_to_logical_flip function and records a total number of shots
    and fails per logical.

    Stopping criteria are the maximum number of shots or relative standard
    error on the estimate of LEP going below the given amount (if not None).
    The RSE stopping criteria is only taken into account if at least
    min_fails are reached.

    Parameters
    ----------
    circuit : stim.Circuit
        Stim circuit to use for sampling shots.
    max_shots : int
        Maximum number of shots to sample.
    decoder : GraphDecoder
        Decoder to use for decoding.
    max_batch_size : int, optional
        Maximal batch size, by default 10_000.
    target_rse : Optional[float], optional
        Target Relative Standard Error (RSE) for early stopping, by default None.
    min_fails : int, optional
        Minimum number of fails before starting to look at RSE, by default 10.

    Returns
    -------
    Dict[str, int]
        Result dictionary with keys:
            shots: The number of times to sample every detector in the circuit
            fails: The number of times the decoder failed to predict the actual observable
    """
    sampler = circuit.compile_detector_sampler()
    batches = [max_batch_size] * (max_shots // max_batch_size)
    if (remaining_shots := max_shots - sum(batches)) > 0:
        batches.append(remaining_shots)
    result: dict[str, int] = {
        "shots": 0,
        "fails": 0,
    }
    result.update({f"fails_log_{log}": 0 for log in range(circuit.num_observables)})
    for batch_size in (pbar := tqdm(batches)):
        syndrome, actual_observables = sampler.sample(
            shots=batch_size, separate_observables=True
        )
        predicted_observables = decoder.decode_batch_to_logical_flip(syndrome)
        result["shots"] += batch_size
        result["fails"] += np.sum(
            np.any(predicted_observables != actual_observables, axis=1)
        ).astype(np.int_)
        fails_per_observable = np.sum(
            predicted_observables != actual_observables, axis=0
        ).astype(np.int_)
        rse = 0
        min_fails_any_log = 0
        for log, fails in enumerate(fails_per_observable):
            log_fails_name = f"fails_log_{log}"
            result[log_fails_name] += fails
            # calculate relative standard error
            lep = result[f"fails_log_{log}"] / result["shots"]
            if lep > 0:
                rse_log = np.sqrt(lep * (1-lep) / result["shots"]) / lep
                rse = max(rse, rse_log)
                min_fails_any_log = max(min_fails_any_log, result[log_fails_name])
        shots, fails = result["shots"], result["fails"]
        pbar.set_description(
            f"Shots: {shots}; Fails: {fails}; RSE: {rse:.3f}")
        if target_rse is not None and \
                rse <= target_rse and min_fails_any_log > min_fails:
            break
    return result
