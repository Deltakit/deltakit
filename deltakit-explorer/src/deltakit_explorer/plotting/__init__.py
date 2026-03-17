# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.explorer.visualisation`` namespace here."""

from deltakit_explorer.plotting._correlation_matrix import (
    correlation_matrix,
    defect_diagram,
    defect_rates,
)
from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import plot_logical_error_probability_per_round
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import (
    LambdaResult,
    LogicalErrorProbabilityPerRoundResult,
    interpolate_lambda,
    interpolate_leppr,
)

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
