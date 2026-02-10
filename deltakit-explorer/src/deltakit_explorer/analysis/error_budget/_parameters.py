from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from deltakit_explorer.analysis.error_budget._discretisation import (
    DiscretisationStrategy,
)


@dataclass(frozen=True)
class FittingParameters:
    num_points_per_parameters: int = 10
    """Number of different values to try for each noise parameter.

    Corresponds to the number of points that will be used to fit a degree ``fitting_degree``
    polynomial. As such, should be greater than ``fitting_degree + 1``.
    """
    discretisation_strategy: DiscretisationStrategy = DiscretisationStrategy.LOGARITHMIC
    """A strategy to generate points that will be used to compute 1 / Λ on different
    values and fit a degree ``fitting_degree`` polynomial.

    Default to logarithmically spaced points.
    """
    fitting_degree: int = 3
    """Degree of polynomial that will be used to approximate 1 / Λ and to compute each
    of its derivatives.

    Should be lower than ``num_points_per_parameters - 1``. Higher values will incur
    higher standard deviation. Default to ``3``, which seems to be a good compromise
    between fit accuracy and resulting standard deviation.
    """

    def __post_init__(self) -> None:
        if self.num_points_per_parameters + 1 < self.fitting_degree + 2:
            msg = (
                f"Estimation of the standard deviation requires at least "
                f"fitting_degree + 2 = {self.fitting_degree + 2} discretisation points, "
                f"but only {self.num_points_per_parameters} + 1 are provided. Please "
                f"increase num_points_per_parameters to at least "
                f"{self.fitting_degree + 1}."
            )
            raise ValueError(msg)

    def get_discretisation(
        self, a: float, b: float, c: float
    ) -> npt.NDArray[np.floating]:
        return self.discretisation_strategy(
            a, b, c, self.num_points_per_parameters, self.fitting_degree
        )


@dataclass(frozen=True)
class SamplingParameters:
    max_shots: int = 10_000_000
    """Maximum number of shots per sampling task.

    A sampling task may stop with a lower number of samples if additional conditions are
    met, see ``lep_target_rse`` or ``lep_computation_min_fails`` for more details.
    """
    batch_size: int = 10_000
    """Number of sampling experiments that are submitted per batch."""
    lep_target_rse: float = 1e-4
    """Target relative standard error under which a sampling task is considered precise
    enough and can be stopped before ``max_shots`` sampling tasks have returned."""
    lep_computation_min_fails: int = 10
    """Minimum number of failures that should be witnessed before stopping a sampling task.

    A sampling task may stop with less failures, for example if ``max_shots`` shots have
    been performed."""
    max_workers: int = 1
    """Max number of parallel processes used by the function.

    Default to ``1`` which means fully sequential.
    """
