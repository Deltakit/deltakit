# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.explorer.analysis`` namespace here."""

from deltakit_explorer.analysis._analysis import (
    get_exp_fit, get_lambda_fit, calculate_lep_and_lep_stddev
)

from deltakit_explorer.analysis.lambda_ import calculate_lambda_and_lambda_stddev

from deltakit_explorer.analysis._leppr import (
    compute_logical_error_per_round, LogicalErrorProbabilityPerRoundResults,
    simulate_different_round_numbers_for_lep_per_round_estimation)

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
