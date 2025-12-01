# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.explorer.analysis.budget`` namespace here."""

from .budget import get_error_budget, ErrorBudgetingResults
from .discretisation import GradientFitDiscretisationGenerator, get_linear_points, get_logarithmic_points
from .generation import generate_decoder_managers_for_lambda
from .gradient import compute_1_over_lambda_gradient_at
from .interfaces import NoiseInterface
from .lambda_ import compute_1_over_lambda_at
from .memory import MemoryGenerator, get_rotated_surface_code_memory_circuit
from .post_processing import compute_lambda_and_stddev_from_results
from .visualisation import plot_error_budget
