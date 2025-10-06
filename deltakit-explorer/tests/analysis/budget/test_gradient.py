from collections.abc import Callable
import itertools
from deltakit_explorer.analysis.budget.gradient.schemes import (
    FirstOrderDerivativeDifference,
    SchemeDirection,
)
import pytest
import numpy


class TestCentralDifference:
    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize(
        "order,c", itertools.product(range(1, 10), (0.1, 0.5, 2, 10))
    )
    def test_central_formula_case(self, order: int, c: float):
        scheme = FirstOrderDerivativeDifference(order, c, SchemeDirection.CENTRAL)
        n = (order + 1) // 2
        numpy.testing.assert_allclose(
            scheme.required_multiples_of_h,
            [
                numpy.sign(i) * c ** (numpy.abs(i) - 1.0)
                for i in range(-n, n + 1)
                if i != 0
            ],
        )
        assert len(scheme._coefficients) == len(scheme._required_multiples_of_h)

    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize(
        "order,c,direction",
        itertools.product(
            range(1, 10),
            (0.1, 0.5, 2, 10),
            (SchemeDirection.BACKWARD, SchemeDirection.FORWARD),
        ),
    )
    def test_non_central_formulas_case(
        self, order: int, c: float, direction: SchemeDirection
    ):
        scheme = FirstOrderDerivativeDifference(order, c, direction)
        n = (order + 1) // 2
        factor = -1 if direction == SchemeDirection.BACKWARD else 1
        numpy.testing.assert_allclose(
            scheme.required_multiples_of_h,
            [
                factor * numpy.sign(i) * numpy.sqrt(c) ** (i - 1.0)
                for i in range(2 * n + 1)
            ],
        )
        assert len(scheme._coefficients) == len(scheme._required_multiples_of_h)

    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize("c", (0.1, 0.5, 2, 10))
    def test_second_order_scheme(self, c: float):
        scheme = FirstOrderDerivativeDifference(2, c)
        numpy.testing.assert_allclose(scheme.required_multiples_of_h, [-1, 1])

    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize("order", range(1, 10))
    def test_raise_on_c_too_close_to_1(self, order: int):
        match_str = "^Cannot have a scaling factor too close to 1.$"
        with pytest.raises(RuntimeError, match=match_str):
            FirstOrderDerivativeDifference(order, 1)

    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize(
        "order,c,func_and_derivative,direction",
        itertools.product(
            range(1, 10),
            (numpy.sqrt(0.1), 0.5, 0.9, 1.1, 2, numpy.sqrt(10)),
            [
                (numpy.polynomial.Polynomial((1, 2, 3)), numpy.polynomial.Polynomial((2, 6))),
                (numpy.exp, numpy.exp),
                (lambda x: 1 / x, lambda x: -1 / x**2),
                (numpy.log, lambda x: 1 / x),
                (numpy.sin, numpy.cos),
                (numpy.arctan, lambda x: 1 / (1 + x**2)),
            ],
            tuple(SchemeDirection),
        ),
    )
    def test_on_analytic_functions(
        self,
        order: int,
        c: float,
        func_and_derivative: tuple[Callable[[float], float], Callable[[float], float]],
        direction: SchemeDirection,
    ):
        func, derivative = func_and_derivative
        scheme = FirstOrderDerivativeDifference(order, c, direction)
        h = 1e-4
        x = 2.0

        values = [func(x + dh * h) for dh in scheme.required_multiples_of_h]
        estimation = scheme.approximate(values, h)
        assert pytest.approx(estimation, rel=1e-4) == derivative(x)
