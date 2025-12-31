# (c) Copyright Riverlane 2020-2025.

import pytest
from deltakit_explorer.codes._bivariate_bicycle_code import Monomial, Polynomial


class TestPolynomial:
    def test_Polynomial_init_works_as_expected(self):
        assert Polynomial([Monomial(1, 2, 3, 3)]).monomials == [Monomial(1, 2, 3, 3)]

    @pytest.mark.parametrize(
        "vec, l, m, exp_poly",
        [
            [[0], 1, 1, Polynomial([Monomial(0, 0, 1, 1)])],
            [[0, 1], 1, 2, Polynomial([Monomial(0, 1, 1, 2)])],
            [[1, 0, 0, 0], 2, 2, Polynomial([Monomial(0, 0, 2, 2)])],
            [[0, 1, 0, 0], 2, 2, Polynomial([Monomial(0, 1, 2, 2)])],
            [[0, 0, 1, 0], 2, 2, Polynomial([Monomial(1, 0, 2, 2)])],
            [[0, 0, 0, 1], 2, 2, Polynomial([Monomial(1, 1, 2, 2)])],
            [
                [1, 1, 0, 0],
                2,
                2,
                Polynomial([Monomial(0, 0, 2, 2), Monomial(0, 1, 2, 2)]),
            ],
            [
                [1, 0, 1, 0],
                2,
                2,
                Polynomial([Monomial(0, 0, 2, 2), Monomial(1, 0, 2, 2)]),
            ],
            [
                [1, 1, 1, 0],
                2,
                2,
                Polynomial(
                    [Monomial(0, 0, 2, 2), Monomial(0, 1, 2, 2), Monomial(1, 0, 2, 2)]
                ),
            ],
            [
                [1, 1, 1, 1],
                2,
                2,
                Polynomial(
                    [
                        Monomial(0, 0, 2, 2),
                        Monomial(0, 1, 2, 2),
                        Monomial(1, 0, 2, 2),
                        Monomial(1, 1, 2, 2),
                    ]
                ),
            ],
        ],
    )
    def test_Polynomial_from_vec_works_as_expected(self, vec, l, m, exp_poly):  # noqa: E741
        assert Polynomial.from_vec(vec, l, m) == exp_poly

    @pytest.mark.parametrize(
        "vec, l, m",
        [
            [[], 1, 1],
            [[1], 1, 1],
            [[0, 1], 1, 2],
            [[1, 0, 0, 0], 2, 2],
            [[0, 1, 0, 0], 2, 2],
            [[0, 0, 1, 0], 2, 2],
            [[0, 0, 0, 1], 2, 2],
            [[1, 1, 0, 0], 2, 2],
            [[1, 0, 1, 0], 2, 2],
            [[1, 1, 1, 0], 2, 2],
            [[1, 1, 1, 1], 2, 2],
        ],
    )
    def test_Polynomial_to_vec_works_as_expected(self, vec, l, m):  # noqa: E741
        assert Polynomial.from_vec(vec, l, m).to_vec() == vec

    def test_Polynomial_repr_str_works_as_expected(self):
        assert (
            str(Polynomial([Monomial(0, 1, 2, 2), Monomial(1, 1, 2, 2)]))
            == "['x^0 y^1', 'x^1 y^1']"
        )

    @pytest.mark.parametrize(
        "poly, exp_inv",
        [
            [Polynomial([]), Polynomial([])],
            [Polynomial([Monomial(1, 1, 2, 2)]), Polynomial([Monomial(1, 1, 2, 2)])],
            [Polynomial([Monomial(1, 1, 3, 3)]), Polynomial([Monomial(2, 2, 3, 3)])],
            [
                Polynomial(
                    [
                        Monomial(1, 1, 3, 3),
                        Monomial(1, 2, 3, 3),
                        Monomial(2, 1, 3, 3),
                        Monomial(2, 2, 3, 3),
                    ]
                ),
                Polynomial(
                    [
                        Monomial(2, 2, 3, 3),
                        Monomial(2, 1, 3, 3),
                        Monomial(1, 2, 3, 3),
                        Monomial(1, 1, 3, 3),
                    ]
                ),
            ],
        ],
    )
    def test_Polynomial_inverse_correct(self, poly, exp_inv):
        assert poly.reverse() == exp_inv

    @pytest.mark.parametrize(
        "poly, mon, exp_poly",
        [
            [
                Polynomial([Monomial(0, 0, 2, 2)]),
                Monomial(1, 1, 2, 2),
                Polynomial([Monomial(1, 1, 2, 2)]),
            ],
            [
                Polynomial([Monomial(0, 0, 3, 3), Monomial(1, 1, 3, 3)]),
                Monomial(1, 1, 3, 3),
                Polynomial([Monomial(1, 1, 3, 3), Monomial(2, 2, 3, 3)]),
            ],
        ],
    )
    def test_Polynomial_mult_by_monomial_correct(self, poly, mon, exp_poly):
        assert poly.mult_by_monomial(mon) == exp_poly

    def test_Polynomial_eq_returns_False_if_compared_to_non_Polynomial_type(self):
        assert Polynomial([]) != 1


class TestMonomial:
    @pytest.mark.parametrize(
        "x_pow, y_pow, l, m", [[1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4], [4, 4, 5, 5]]
    )
    def test_Monomial_init_correct_for_valid_values(self, x_pow, y_pow, l, m):  # noqa: E741
        mon = Monomial(x_pow, y_pow, l, m)
        assert mon.x_pow == x_pow
        assert mon.y_pow == y_pow
        assert mon.l == l
        assert mon.m == m

    def test_Monomial_init_throws_ValueError_if_l_m_less_than_1(self):
        with pytest.raises(ValueError, match="l and m must be >= 0"):
            Monomial(0, 0, 0, 0)

    @pytest.mark.parametrize(
        "x_pow, y_pow, l, m, exp_x_pow, exp_y_pow",
        [
            [2, 2, 1, 1, 0, 0],
            [3, 3, 2, 2, 1, 1],
            [3, 2, 2, 2, 1, 0],
            [2, 3, 2, 2, 0, 1],
        ],
    )
    def test_Monomial_init_adjusts_x_pow_y_pow_to_modulo_l_m_respectively(
        self, x_pow, y_pow, l, m, exp_x_pow, exp_y_pow  # noqa: E741
    ):
        mon = Monomial(x_pow, y_pow, l, m)
        assert mon.x_pow == exp_x_pow
        assert mon.y_pow == exp_y_pow

    @pytest.mark.parametrize(
        "mon1, mon2, prod",
        [
            [Monomial(1, 2, 3, 3), Monomial(2, 1, 3, 3), Monomial(0, 0, 3, 3)],
            [Monomial(0, 0, 3, 3), Monomial(2, 1, 3, 3), Monomial(2, 1, 3, 3)],
            [Monomial(0, 3, 3, 3), Monomial(2, 1, 3, 3), Monomial(2, 1, 3, 3)],
        ],
    )
    def test_Monomial_mul_correct(self, mon1, mon2, prod):
        assert mon1 * mon2 == prod

    def test_Monomial_print_works_as_expected(self):
        assert str(Monomial(2, 2, 3, 3)) == "x^2 y^2"

    @pytest.mark.parametrize(
        "mon, inv",
        [
            [Monomial(1, 1, 2, 2), Monomial(1, 1, 2, 2)],
            [Monomial(1, 1, 3, 3), Monomial(2, 2, 3, 3)],
            [Monomial(0, 0, 3, 3), Monomial(0, 0, 3, 3)],
            [Monomial(1, 2, 3, 5), Monomial(2, 3, 3, 5)],
        ],
    )
    def test_Monomial_inverse_works_as_expected(self, mon: Monomial, inv):
        assert mon.inverse() == inv

    def test_Monomial_eq_returns_False_if_compared_to_non_Monomial(self):
        assert Monomial(1, 1, 2, 2) != 2
