from collections.abc import Sequence
from enum import Enum, auto
from math import isclose, sqrt

import numpy
import numpy.typing as npt


class NumericalScheme:
    """Generic implementation of a numerical scheme approximating partial derivatives.

    Warning:
        This class does **not** check that the provided arguments form a valid numerical
        scheme. It is up to the user to ensure that the provided
        ``required_multiples_of_h``, ``coefficients`` and ``derivative_order`` values
        are correct to approximate the desired derivative.

    Args:
        name (str): name of the numerical scheme. Might be used by external functions to
            add metadata to results or name files.
        required_multiples_of_h (Sequence[float]): an **ordered** sequence of numbers
            representing the multiples of the spacing factor ``h``. For example, for the
            second-order central approximation of the second derivative of a function
            (first formula in https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences)
            these would be ``[-1, 0, 1]`` because we need to evaluate the function at
            the points ``x + (-1) * h``, ``x + (0) * h`` and ``x + (1) * h``. Note that
            this sequence should be ordered w.r.t the provided ``coefficients``, i.e.,
            the coefficient in the first entry of the provided ``coefficients`` should
            correspond to the coefficient found in front of the evaluation of the
            function at the point ``x + required_multiples_of_h[0] * h``.
        coefficients (Sequence[float]): weights in the sum that approximates the desired
            derivative. Should be ordered w.r.t the provided ``required_multiples_of_h``.
            For example, for the second-order central approximation of the second
            derivative of a function (first formula in
            https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences)
            these would be ``[1, -2, 1]``.
        derivative_order (int): power at which the step ``h`` should be raised in the
            denominator.
    """

    def __init__(
        self,
        name: str,
        required_multiples_of_h: Sequence[float],
        coefficients: Sequence[float],
        derivative_order: int,
    ) -> None:
        self._name = name
        self._required_multiples_of_h = list(required_multiples_of_h)
        self._coefficients = list(coefficients)
        self._derivative_order = derivative_order

    def approximate(
        self,
        values: Sequence[float] | npt.NDArray[numpy.float64],
        h: float,
    ) -> float:
        """Compute a numerical approximation of the implemented derivative.

        Args:
            values (Sequence[float] | npt.NDArray[numpy.float64]): evaluations of the
                function to be differentiated at the different points in
                ``self.required_multiples_of_h``. Should be in the same order as
                ``self.required_multiples_of_h``.

        Returns:
            an estimation of the derivative.
        """
        derivative_estimation = (
            sum(c * v for c, v in zip(self._coefficients, values))
            / h**self._derivative_order
        )
        return derivative_estimation

    def approximate_with_stddev(
        self,
        values: Sequence[float] | npt.NDArray[numpy.float64],
        stddev: Sequence[float] | npt.NDArray[numpy.float64],
        h: float,
    ) -> tuple[float, float]:
        """Compute a numerical approximation of the implemented derivative and its
        standard error.

        Warning:
            the returned standard error does not take into account the approximation
            made by the numerical scheme.

        Args:
            values (Sequence[float] | npt.NDArray[numpy.float64]): evaluations of the
                function to be differentiated at the different points in
                ``self.required_multiples_of_h``. Should be in the same order as
                ``self.required_multiples_of_h``.
            stddev (Sequence[float] | npt.NDArray[numpy.float64]): standard deviation of
                the provided ``values``.

        Returns:
            an estimation of the derivative along with the corresponding standard error.
        """
        derivative_estimation_stdder = (
            sqrt(sum((c * s) ** 2 for c, s in zip(self._coefficients, stddev)))
            / h**self._derivative_order
        )
        return self.approximate(values, h), derivative_estimation_stdder

    @property
    def required_multiples_of_h(self) -> list[float]:
        return self._required_multiples_of_h

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        string = f"∇^{{{self._derivative_order}}}f(x) ≈ ("
        signs = [" + " if c >= 0 else " - " for c in self._coefficients]
        signs[0] = ""
        for s, coeff, h in zip(
            signs, self._coefficients, self._required_multiples_of_h, strict=True
        ):
            string += f"{s}{abs(coeff):.5g} * f(x + {h:.5g})"
        string += ")"
        return string


class ThreePointCentralDifference(NumericalScheme):
    """∇f(x) ≈ (f(x + h) - f(x - h)) / (2*h)"""

    def __init__(self) -> None:
        super().__init__("central", [-1, 1], [-1 / 2, 1 / 2], 1)


class ThreePointForwardDifference(NumericalScheme):
    """∇f(x) ≈ (-3 * f(x) + 4 * f(x + h) - f(x + 2*h)) / (2*h)"""

    def __init__(self) -> None:
        super().__init__("forward", [0, 1, 2], [-3 / 2, 2, -1 / 2], 1)


class ThreePointBackwardDifference(NumericalScheme):
    """∇f(x) ≈ (f(x - 2*h) - 4 * f(x - h) + 3 * f(x)) / (2*h)"""

    def __init__(self) -> None:
        super().__init__("backward", [-2, -1, 0], [1 / 2, -2, 3 / 2], 1)


class SchemeDirection(Enum):
    CENTRAL = auto()
    FORWARD = auto()
    BACKWARD = auto()


class FirstOrderDerivativeDifference(NumericalScheme):
    """A generic scheme suitable for iterative step-size improvement in numerical
    gradient estimation.

    For some functions, the step size ``h`` that should be used to approximate the
    gradient is not known in advance. This class implements a numerical scheme that is
    using successive powers of ``c`` to weight the step size.

    See the documentation of ``scipy.differentiate.derivative`` for a more in-depth
    overview:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html

    Warning:
        the ``c`` argument does not follow the same convention as ``scipy``. In
        ``scipy``, powers of ``c`` are used to divide ``h``. In this implementation,
        powers of ``c`` are multiplicative factors of ``h``. That means that ``c > 1``
        will make the steps go further away from the central point when ``order`` is
        increased.

    Note:
        the implementation of this class has been strongly inspired by the
        implementation of ``scipy.differentiate._derivative_weights``.

    Args:
        approximation_order: order of the numerical scheme to use.
        c: exponential **factor** of ``h``.

    Raises:
        RuntimeError: when ``c`` is close to ``1`` as that makes the numerical scheme
            degenerate and likely to be of lower order.
    """

    def __init__(
        self,
        approximation_order: int,
        c: float = 2,
        direction: SchemeDirection = SchemeDirection.CENTRAL,
    ):
        if isclose(c, 1):
            raise RuntimeError("Cannot have a scaling factor too close to 1.")
        name = f"central_{approximation_order}_1st_order_derivative"
        n = (approximation_order + 1) // 2
        # Note: strongly inspired from
        # https://github.com/scipy/scipy/blob/0cf8e9541b1a2457992bf4ec2c0c669da373e497/scipy/differentiate/_differentiate.py#L592
        indices: npt.NDArray[numpy.int_]
        factor: float
        match direction:
            case SchemeDirection.CENTRAL:
                indices = numpy.arange(-n, n + 1)
                factor = c
            case SchemeDirection.FORWARD | SchemeDirection.BACKWARD:
                # We compute everything for the forward scheme, inverting the weights at
                # the end if the user wants a backward scheme.
                indices = numpy.arange(2 * n + 1)
                factor = numpy.sqrt(c)

        powers = numpy.abs(indices) - 1.0
        signs = numpy.sign(indices)

        h = signs * factor**powers
        A = numpy.vander(h, increasing=True).T
        b = numpy.zeros(2 * n + 1)
        b[1] = 1
        weights = numpy.linalg.solve(A, b)

        # Enforce identities to improve accuracy if a central scheme is required.
        if direction == SchemeDirection.CENTRAL:
            # 1. The central element is not used in a central difference scheme.
            weights = numpy.delete(weights, [n], axis=0)
            h = numpy.delete(h, [n], axis=0)
            # 2. The weights of non-central elements are symmetric.
            for i in range(n):
                weights[-i - 1] = -weights[i]

        # Invert the weights if a backward scheme is needed
        if direction == SchemeDirection.BACKWARD:
            h, weights = -h, -weights

        super().__init__(name, h.tolist(), weights.tolist(), 1)
