# (c) Copyright Riverlane 2020-2025.
"""Three tests that run the circuit-generation, simulation, decoding and
analysis pipeline end to end with real components rather than mocks.

A first slice towards https://github.com/Deltakit/deltakit/issues/45.
"""

import warnings

import deltakit_stim as stim
import pytest
from deltakit_circuit.gates import PauliBasis
from deltakit_decode import PyMatchingDecoder
from deltakit_decode.analysis import StimDecoderManager, run_decoding_on_circuit
from deltakit_explorer.analysis import (
    calculate_lep_and_lep_stddev,
    compute_logical_error_per_round,
)
from deltakit_explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit_explorer.qpu import QPU, SI1000Noise


def _decoder_and_stim_circuit(
    distance: int, num_rounds: int, physical_error_rate: float
) -> tuple[PyMatchingDecoder, stim.Circuit]:
    """Build a real MWPM decoder and its matching Stim circuit for a rotated
    planar code Z-memory experiment, compiled onto a QPU with SI1000 noise.

    Args:
        distance: Code distance, used as both the width and height of the
            rotated planar code.
        num_rounds: Number of syndrome-extraction rounds in the memory
            experiment.
        physical_error_rate: SI1000 physical error rate used to compile
            noise onto the QPU.

    Returns:
        The MWPM decoder and its matching Stim circuit.
    """
    code = RotatedPlanarCode(width=distance, height=distance)
    circuit = css_code_memory_circuit(
        css_code=code, num_rounds=num_rounds, logical_basis=PauliBasis.Z
    )
    qpu = QPU(qubits=code.qubits, noise_model=SI1000Noise(p=physical_error_rate))
    noisy_circuit = qpu.compile_and_add_noise_to_circuit(circuit)
    return PyMatchingDecoder.construct_decoder_and_stim_circuit(noisy_circuit)


class TestDecodingPipeline:
    """End-to-end tests spanning circuit generation, simulation, decoding and
    analysis together, instead of mocking any single layer.
    """

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_run_decoding_on_circuit_reports_few_failures_below_threshold(
        self, distance: int
    ) -> None:
        # p is low enough that failures are rare, but high enough that
        # thousands of shots carry non-trivial syndromes, so the decoder is
        # genuinely exercised. A working decoder fails ~8 times at d=3 and
        # is further suppressed at larger distances; one that ignored the
        # syndromes would fail ~1,000+ times, growing with distance, so the
        # bound separates the two by orders of magnitude at every size.
        decoder, stim_circuit = _decoder_and_stim_circuit(
            distance=distance, num_rounds=3, physical_error_rate=1e-4
        )
        max_shots = 200_000
        max_failures = 40

        result = run_decoding_on_circuit(
            stim_circuit, max_shots=max_shots, decoder=decoder
        )

        assert result["shots"] == max_shots
        assert result["fails"] < max_failures
        assert result["fails"] == result["fails_log_0"]

    def test_stim_decoder_manager_is_deterministic_given_a_seed(self) -> None:
        decoder, stim_circuit = _decoder_and_stim_circuit(
            distance=3, num_rounds=3, physical_error_rate=1e-3
        )

        first_run = StimDecoderManager(stim_circuit, decoder, seed=1234)
        first_shots, first_fails = first_run.run_batch_shots(5000)

        second_run = StimDecoderManager(stim_circuit, decoder, seed=1234)
        second_shots, second_fails = second_run.run_batch_shots(5000)

        assert (first_shots, first_fails) == (second_shots, second_fails)
        # Below threshold a working decoder gives fewer than 20 failures
        # here; one that ignored the syndromes would give ~245.
        assert 0 < first_fails < 0.02 * first_shots

    def test_logical_error_rate_per_round_decreases_with_code_distance(self) -> None:
        # Below threshold, a larger code distance should suppress the
        # logical error rate per round; at or above threshold the ordering
        # reverses, so the choice of physical_error_rate is part of what
        # makes this assertion meaningful.
        #
        # The per-round rate comes from compute_logical_error_per_round on
        # a single (rounds, LEP) point per distance. A multi-round sweep at
        # this error rate would need hundreds of rounds per point to reach
        # the LEP of ~0.4 its fit is tuned for; at the round counts used
        # here it emits its R^2 < 0.98 warning, which this repository's
        # filterwarnings = ["error"] would promote to a failure, hence the
        # single-point form (SPAM error neglected) and the scoped
        # suppression below.
        physical_error_rate = 1e-3
        shots = 20_000
        logical_error_rate_per_round = {}

        for distance in (3, 5):
            num_rounds = distance
            decoder, stim_circuit = _decoder_and_stim_circuit(
                distance=distance,
                num_rounds=num_rounds,
                physical_error_rate=physical_error_rate,
            )
            decoder_manager = StimDecoderManager(stim_circuit, decoder, seed=1234)
            _, fails = decoder_manager.run_batch_shots(shots)
            lep, lep_stddev = calculate_lep_and_lep_stddev(int(fails), shots)
            # compute_logical_error_per_round warns that a single data point
            # was provided; that is the intended usage here (see above), and
            # the repository promotes warnings to errors, so suppress it for
            # exactly this call.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                logical_error_rate_per_round[distance] = (
                    compute_logical_error_per_round([num_rounds], lep, lep_stddev).leppr
                )

        # d=3 sits around 1e-3 to 1.5e-3 here; a badly weighted or broken
        # decoder lands several-fold higher.
        max_leppr_distance_3 = 3e-3

        # Ordering: below threshold, larger distance suppresses errors.
        assert logical_error_rate_per_round[5] < logical_error_rate_per_round[3]
        assert logical_error_rate_per_round[3] < max_leppr_distance_3
