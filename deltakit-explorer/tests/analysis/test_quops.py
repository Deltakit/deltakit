# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from math import exp, log

import pytest
from deltakit_explorer.analysis import \
    RotatedPlanarErrorSuppressionCalculator


class TestRotatedPlanarErrorSuppressionCalculator:

    @classmethod
    def alternative_lep_per_round(cls, p_0: float, lambda_: float, d: int) -> float:
        return p_0 * exp(-log(lambda_) * (d + 1) / 2)

    @pytest.fixture
    def calculator(self):
        return RotatedPlanarErrorSuppressionCalculator(1e-7, 8)

    @pytest.mark.parametrize("p_0, lambda_", [(0.001, 1), (0.001, -2), (0.8, 1.2)])
    def test_constructor_raises_ValueError_for_invalid_arguments(self, p_0, lambda_):
        with pytest.raises(
            ValueError,
            # Use a raw string to avoid invalid escape sequences (e.g. \( ) becoming a SyntaxWarning under -W error
            match=r"Error suppression requires lambda > 1 and e_L\(d\) < 0.5 for all distances greater than 3"
        ):
            RotatedPlanarErrorSuppressionCalculator(p_0, lambda_)

    @pytest.mark.parametrize("distance", ([13, 15, 17]))
    def test_calculate_lep_method_matches_approximation_at_low_error_rates(
        self,
        calculator: RotatedPlanarErrorSuppressionCalculator,
        distance: int
    ):
        num_rounds = 10
        lep = calculator.calculate_lep(distance, num_rounds)
        expected_lep_per_round = self.alternative_lep_per_round(
            calculator.p_0, calculator.lambda_, distance
        )
        # Approximation that works when lep per round is small.
        assert lep == pytest.approx(expected_lep_per_round * num_rounds)

    @pytest.mark.parametrize("distance", ([3, 5]))
    def test_calculate_lep_method_approaches_a_half_at_large_num_rounds(
        self,
        calculator: RotatedPlanarErrorSuppressionCalculator,
        distance: int
    ):
        assert calculator.calculate_lep(distance, num_rounds=1e11) == pytest.approx(0.5)

    @pytest.mark.parametrize("distance", ([3, 11, 21]))
    @pytest.mark.parametrize("num_rounds", ([1, 5, 1e3, 1e8]))
    def test_calculate_lep_method_reproduces_expected_values(
        self,
        calculator: RotatedPlanarErrorSuppressionCalculator,
        distance: int,
        num_rounds: int
    ):
        expected_lep_per_round = self.alternative_lep_per_round(
            calculator.p_0, calculator.lambda_, distance
        )
        expected_lep = 0.5 * (1 - pow(1 - 2 * expected_lep_per_round, num_rounds))

        lep = calculator.calculate_lep(distance, num_rounds)
        assert lep == pytest.approx(expected_lep)

    @pytest.mark.parametrize("distance", ([3, 11, 19]))
    def test_predict_quops_at_distance_method(
        self,
        calculator: RotatedPlanarErrorSuppressionCalculator,
        distance: int
    ):
        expected_lep_per_round = self.alternative_lep_per_round(
            calculator.p_0, calculator.lambda_, distance
        )
        expected_lep = 0.5 * (1 - pow(1 - 2 * expected_lep_per_round, distance))
        prediction = calculator.predict_quops_at_distance(distance)
        assert pytest.approx(1 / expected_lep) == prediction

    def test_predict_distance_for_quops_method_when_QuOps_too_small(
        self,
        calculator: RotatedPlanarErrorSuppressionCalculator
    ):
        with pytest.warns(
            Warning,
            match=("Desired number of QuOps are too small to find a required distance"
                   " - returning distance=1")
            ):
            distance = calculator.predict_distance_for_quops(1)
            assert distance == 1

    def test_predict_distance_for_quops_method_warns_when_QuOps_too_big(self):
        # Create a system that has a lambda very close to threshold and try and reach
        # a large number of QuOps with this.
        calc = RotatedPlanarErrorSuppressionCalculator(4e-2, 1.05)
        with pytest.warns(
            Warning,
            match="Required distance exceeds 999 - returning 999"
        ):
            distance = calc.predict_distance_for_quops(1e9)
            assert distance == 999
