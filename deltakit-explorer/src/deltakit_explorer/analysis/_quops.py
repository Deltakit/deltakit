# (c) Copyright Riverlane 2020-2025.
"""Contains functions for calculating QuOps based on error suppression models.

References
----------
- https://doi.org/10.48550/arXiv.2408.13687
"""

import warnings
from math import ceil

import numpy as np


class RotatedPlanarErrorSuppressionCalculator:
    """Class representing how the error probability per round, e_L(d), is suppressed
    as a function of the distance of a code according to

    e_L(d) = p_0 * Lambda ^ (-(d + 1) / 2).

    Parameters
    ----------
    p_0
        Zero offset of the logical error rate per round.
    lambda
        The error suppression factor.

    Raises
    ------
    ValueError
        If either lambda > 1 or lambda/p_0 < 0.5
    """

    def __init__(
        self,
        p_0: float,
        lambda_: float,
        p_0_std: float = 0.0,
        lambda_std: float = 0.0,
    ):
        if lambda_ <= 1 or p_0 / lambda_ ** 2 > 0.5:
            raise ValueError(
                "Error suppression requires lambda > 1 and "
                "e_L(d) < 0.5 for all distances greater than 3"
            )
        self.p_0 = p_0
        self.lambda_ = lambda_
        self.p_0_std = p_0_std
        self.lambda_std = lambda_std

    def calculate_lep(self, distance: int, num_rounds: int) -> float:
        """Returns the probability of observing a logical error on a code of fixed
        distance after a number of rounds.

        It uses the formula in Section VI.B of Supplementary Information in
        https://doi.org/10.48550/arXiv.2408.13687 which is the sum of the probabilities
        of all ways of there being an odd number of errors in fixed number of rounds.
        """
        lep_per_round = self.p_0 * self.lambda_ ** (-(distance + 1) / 2)
        # At `lep_per_round` << 1 this is be approximated as `lep_per_round * num_rounds`
        return 0.5 * (1 - (1 - 2 * lep_per_round) ** num_rounds)

    def predict_quops_at_distance(self, distance: int) -> float:
        """Returns the number of QuOps to the nearest integer at a given distance. This
        uses the definition that the number of QuOps achieveable is 1 / pL, where pL is
        the probability of a logical error occurring in a dxdxd block.

        Parameters
        ----------
        distance
            The distance at which to calculate the number of QuOps.
        """
        return 1. / self.calculate_lep(distance, distance)

    def predict_distance_for_quops(self, num_quops: float) -> int:
        """Returns the nearest distance that achieves the desired number of QuOps to one
        decimal place. Uses the definition that the number of QuOps achievable at a
        particular distance is 1 / pL, where pL is the probability of a logical error
        occurring during a dxdxd memory experiment.

        Parameters
        ----------
        num_quops
            Number of desired QuOps, must be a positive integer greater than 2.

        Warnings
        --------
        UserWarning
            - If the number of QuOps is too small. This occurs when the required pL is
              satisfied for all distances. In this case returns distance=1.
            - If the number of QuOps is too large. This occurs when the required pL
              can only be met by a distance greater than 999. In this case returns
              distance=999.
        """
        required_lep_of_dxdxd_block = 1. / num_quops

        vectorise_lep_of_dxdxd_block = np.vectorize(lambda x: self.calculate_lep(x, x))

        # First evaluate at all interesting odd integer distances.
        distances_coarse = np.arange(3, 1000, 2)
        lep_of_dxdxd_blocks = vectorise_lep_of_dxdxd_block(distances_coarse)
        diffs_coarse = lep_of_dxdxd_blocks - required_lep_of_dxdxd_block

        # There are no roots to f(d) = required_lep - predicted_lep(d)
        if np.all(diffs_coarse < 0):
            warnings.warn(
                "Desired number of QuOps are too small to find a required distance "
                "- returning distance=1"
            )
            return 1

        # Estimate the first root by finding first sign change.
        root_found = False
        for i in range(len(diffs_coarse)-1):
            if diffs_coarse[i] * diffs_coarse[i+1] <= 0:
                root_found = True
                break

        if not root_found:
            warnings.warn("Required distance exceeds 999 - returning 999")
            return 999

        # Do a more fine-grained second pass to the nearest decimal.
        distances_fine = np.arange(distances_coarse[i], distances_coarse[i+1] + 0.1, 0.1)
        lep_of_dxdxd_blocks = vectorise_lep_of_dxdxd_block(distances_fine)
        min_index_fine = np.argmin(
            np.abs(lep_of_dxdxd_blocks - required_lep_of_dxdxd_block)
        )
        return ceil(distances_fine[min_index_fine])
