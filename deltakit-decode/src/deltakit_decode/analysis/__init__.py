# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.decode.analysis`` namespace here."""

from deltakit_decode.analysis._decoder_manager import (
    DecoderManager,
    InvalidGlobalManagerStateError,
)
from deltakit_decode.analysis._decoding_circuits import run_decoding_on_circuit
from deltakit_decode.analysis._empirical_decoding_error_distribution import (
    EmpiricalDecodingErrorDistribution,
)
from deltakit_decode.analysis._matching_decoder_managers import (
    B8DecoderManager,
    GraphDecoderManager,
    StimDecoderManager,
)
from deltakit_decode.analysis._run_all_analysis_engine import RunAllAnalysisEngine

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
