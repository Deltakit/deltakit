# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from math import exp, log

import pytest
from deltakit_explorer.analysis._quops import \
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
            match="Error suppression requires lambda > 1 and e_L\(d\) < 0.5 for all distances greater than 3"
        ):
            RotatedPlanarErrorSuppressionCalculator(p_0, lambda_)


    def test_fit_from_data_classmethod_returns_ref_values(self):
        ref_lambda_value, ref_lambda_value_std = 3.207, 0.012726
        distances = [5, 7, 9, 11]
        logical_error_rate_per_round = [1.11e-03, 3.54e-04, 1.09e-04, 3.22e-05]
        stddev_logical_error_rate_per_round = [4.71e-06, 2.25e-06, 1.10e-06, 5.42e-07]

        calc = RotatedPlanarErrorSuppressionCalculator.fit_from_data(
                distances,
                logical_error_rate_per_round,
                stddev_logical_error_rate_per_round
            )

        assert ref_lambda_value == pytest.approx(calc.lambda_, rel=0.001)
        assert ref_lambda_value_std == pytest.approx(calc.lambda_std, rel=0.001)


    @pytest.mark.parametrize("p_0, lambda_", [(0.001, 2), (0.03, 6.78)])
    def test_calculate_lep_per_round_values_method(self, p_0, lambda_):
        calc = RotatedPlanarErrorSuppressionCalculator(p_0, lambda_)
        distances = [0, 3, 5, 7, 9, 11]
        errors_per_round = calc.calculate_lep_per_round(distances)

        for d, err_per_round in zip(distances, errors_per_round):
            expected_err_per_round = self.alternative_lep_per_round(p_0, lambda_, d)
            assert expected_err_per_round == pytest.approx(err_per_round)

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

        assert calculator.predict_quops_at_distance(distance) == round(1 / expected_lep)

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

    @pytest.mark.parametrize("quops, calculator", (
        [(1e5, RotatedPlanarErrorSuppressionCalculator(0.1, 1.5)),
         (1e8, RotatedPlanarErrorSuppressionCalculator(1e-4, 5)),
         (1e13, RotatedPlanarErrorSuppressionCalculator(1e-6, 6))]))
    def test_predict_distance_for_quops_method_return_values(
        self,
        quops: int,
        calculator: RotatedPlanarErrorSuppressionCalculator
    ):
        """Here we check that the returned distance leads to a dxdxd block having the
        closest LEP to the required one. Changing the distance by +-0.1 should cause the
        `diff = achieved_LEP - required_LEP` to increase in the positive / negative
        direction.
        """
        required_lep = 1 / quops

        def expected_lep_of_dxdxd_block(calculator, distance):
            expected_lep_per_round = self.alternative_lep_per_round(
                calculator.p_0, calculator.lambda_, distance
            )
            return 0.5 * (1 - pow(1 - 2 * expected_lep_per_round, distance))

        distance = calculator.predict_distance_for_quops(quops)

        achieved_lep = expected_lep_of_dxdxd_block(calculator, distance)
        achieved_lep_next_d = expected_lep_of_dxdxd_block(calculator, distance + 0.1)
        achieved_lep_last_d = expected_lep_of_dxdxd_block(calculator, distance - 0.1)

        diff = achieved_lep - required_lep
        diff_next_d = achieved_lep_next_d - required_lep
        diff_last_d = achieved_lep_last_d - required_lep

        assert min(abs(diff), abs(diff_next_d), abs(diff_last_d)) == abs(diff)
        # Check that there is a sign change on one of the sides.
        assert diff * diff_next_d <= 0 or diff * diff_last_d <= 0
