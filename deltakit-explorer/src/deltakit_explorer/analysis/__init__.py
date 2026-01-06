# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.explorer.analysis`` namespace here."""

from deltakit_explorer.analysis._analysis import (
    calculate_lep_and_lep_stddev,
    get_exp_fit,
    get_lambda_fit,
)
from deltakit_explorer.analysis._lambda import calculate_lambda_and_lambda_stddev
from deltakit_explorer.analysis._leppr import (
    LogicalErrorProbabilityPerRoundResults,
    compute_logical_error_per_round,
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from deltakit_explorer.analysis._quops import (
    predict_distance_for_quops,
    predict_quops_at_distance,
)

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
