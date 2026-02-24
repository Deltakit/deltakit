from abc import ABC
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class InterpolationPlot(ABC):
    def __init__(self):
        return

    def set_distances(self, distance_grid: npt.NDArray[int]) -> None:
        self.distance_grid = distance_grid

    def _interpolate(self):
        return

    def _interpolate_results(self):
        return


@dataclass
class LambdaPlotResults(InterpolationPlot):
    """Named-tuple-like class containing computation results from
    :func:`calculate_lambda_and_lambda_stddev`.

    Attributes:
        lambda_ (float): computed error suppression factor.
        lambda_stddev (float): lambda standard deviation.
        lambda0 (float): computed error suppression multiplicative offset (value of Λ_0
            in the expression ``Ɛ_d = 1 / [ Λ_0 * Λ**((d+1)/2) ]``).
        lambda0_stddev (float): Λ_0 standard deviation.
        distance_grid (npt.NDArray[np.int_]):
        lambda_interpolated (npt.NDArray[np.floating]):
        lambda_interpolated_low (npt.NDArray[np.floating]):
        lambda_interpolated_high (npt.NDArray[np.floating]):
    """

    lambda_: float
    lambda_stddev: float
    lambda0: float
    lambda0_stddev: float
    distance_grid: npt.NDArray[np.int_] | None = None
    lambda_interpolated: npt.NDArray[np.floating] | None = None
    lambda_interpolated_low: npt.NDArray[np.floating] | None = None
    lambda_interpolated_high: npt.NDArray[np.floating] | None = None

    def _interpolate(
        self,
        lambda_: float | None = None,
        lambda0: float | None = None,
        distance_grid: npt.NDArray[np.int_] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Computes logical error probability per round that would be obtained with the
        provided values.

        Uses the formula ``ε = 1 / Λ_0 * Λ**(-(d + 1) / 2)`` where:

        - ``ε`` is the logical error probability per round,
        - ``Λ_0`` is a multiplicative constant,
        - ``Λ`` is the error suppression factor,
        - ``d`` is the distance of the code,

        to estimate the logical error probability per round from the provided ``lambda_``
        and ``lambda0`` on the provided list of ``distance_grid``.
        Args:
            lambda_: (float | None): - error suppression factor.
            lambda0: (float | None): -  multiplicative constant.
            distance_grid: (npt.NDArray[np.int_] | None) - distance of the code.
        Returns:
            npt.NDArray[np.floating]: List of interpolated points.
        """
        self.lambda_interpolated = (lambda_ ** (-(distance_grid + 1) / 2)) / lambda0
        return self.lambda_interpolated

    def _interpolate_error(
        self, num_sigmas: int | None = 3
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Function for the producing the error band.

        Args:
            num_sigmas: (int|None): - standard deviation.

        Returns:
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: -  Upper and Lower bounds.
        """
        self.lambda_interpolated_low = self._interpolate(
            self.lambda0 - num_sigmas * self.lambda0_stddev,
            self.lambda_ - num_sigmas * self.lambda_stddev,
            self.distance_grid,
        )

        self.lambda_interpolated_high = self._interpolate(
            self.lambda0 + num_sigmas * self.lambda0_stddev,
            self.lambda_ + num_sigmas * self.lambda_stddev,
            self.distance_grid,
        )
        return self.lambda_interpolated_low, self.lambda_interpolated_high


@dataclass
class LepprPlotResult(InterpolationPlot):
    """The dataclass that contains the information for plotting of the
    Logical Error Probability Per Round.

    Attributes:
        leppr (float): Logical Error Probability Per Round (LEPPR).
        leppr_stddev (float): LEPPR standard deviation.
        spam_error (float): computed SPAM error probability.
        spam_error_stddev (float): SPAM error probability standard deviation.
        distance_grid (npt.NDArray[np.int_]): The distance of the code.
        lep_interpolated (npt.NDArray[np.floating]): Interpolated values.
        lep_interpolated_low (npt.NDArray[np.floating]): Lower bound of interpolated values.
        lep_interpolated_high (npt.NDArray[np.floating]): Higher bounds of interpolated value.
    """

    leppr: float
    leppr_stddev: float
    spam_error: float
    spam_error_stddev: float
    distance_grid: npt.NDArray[np.int_] | None = None
    lep_interpolated: npt.NDArray[np.floating] | None = None
    lep_interpolated_low: npt.NDArray[np.floating] | None = None
    lep_interpolated_high: npt.NDArray[np.floating] | None = None

    def _interpolate(
        self,
        spam: float | None,
        leppr: float | None,
        distance_grid: npt.NDArray[np.int_] | None,
    ) -> npt.NDArray[np.floating]:
        """Computes logical error that would be obtained with the provided values.

        Uses the formula ``F = Fs * Fε**r`` where:

        - ``F`` is the expected fidelity of the computation,
        - ``Fs`` is the fidelity of SPAM-related operations,
        - ``Fε`` is the fidelity of one quantum error-correction round,
        - ``r`` is the number of quantum error-correction rounds performed.

        Each fidelity is obtained from the respective error probability with the formula
        ``f = (1 - 2 * e)`` where ``f`` is any of ``F``, ``Fs`` or ``Fε`` and ``e`` is any
        of logical error probability, logical error probability of a SPAM or logical error
        probability per round.

        Args:
            spam: (float | None): - SPAM error.
            leppr: (float | None): - logical error probability per run.
            distance_grid: (npt.NDArray[np.int_] | None) - distance of the code.

        Returns:
            npt.NDArray[np.floating]: List of interpolated points.
        """
        expected_fidelity = (1 - 2 * spam) * (1 - 2 * leppr) ** distance_grid
        self.lep_interpolated = (1 - expected_fidelity) / 2
        return self.lep_interpolated

    def _interpolate_error(
        self, num_sigmas: int = 3
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Function for the producing the error band.
        Args:
            num_sigmas: (int|None): - standard deviation.

        Returns:
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: Upper and Lower bounds.
        """
        self.lep_interpolated_low = self._interpolate(
            self.spam_error - num_sigmas * self.spam_error_stddev,
            self.leppr - num_sigmas * self.leppr_stddev,
            self.distance_grid,
        )
        self.lep_interpolated_high = self._interpolate(
            self.spam_error + num_sigmas * self.spam_error_stddev,
            self.leppr + num_sigmas * self.leppr_stddev,
            self.distance_grid,
        )
        return self.lep_interpolated_low, self.lep_interpolated_high
