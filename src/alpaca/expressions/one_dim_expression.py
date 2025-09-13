# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""
@authors: kuen,
"""
from __future__ import annotations
import math
from typing import List, Tuple, TYPE_CHECKING

from alpaca.model_data import variable as var, constraint as con
import alpaca.expressions.expression as exn
import alpaca.settings as s

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class OneDimExpression(exn.Expression):
    """One dimensional expression representing a function of a single variable.

    This class serves as a base for various one-dimensional mathematical expressions
    where a representative variable equals some function of an input variable.
    Subclasses implement specific mathematical functions like square, exponential, etc.

    Attributes:
        name: Unique identifier for the expression.
        level: Level of expression in expression tree.
        representative_variable: Variable representing the result of the expression.
    """

    def __init__(
        self,
        name: str,
        model_data: ModelData,
        variable: var.Variable,
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize a one-dimensional expression.

        Args:
            name: Unique identifier for the expression.
            level: Level of expression in expression tree.
            representative_variable: Optional existing variable to represent the expression result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, level, representative_variable)
        self.variable: var.Variable = variable
        self.variable.add_nonlinearity_to_occurring_in(self.__class__.__name__)

    def get_linear_approximation_function_parameters_for_segment(
        self, var_lb: float, var_ub: float
    ) -> tuple[float, float]:
        """Calculate slope and intercept of the linear approximation over [var_lb, var_ub]."""
        slope = (self.f(var_ub) - self.f(var_lb)) / (var_ub - var_lb)
        intercept = self.f(var_lb) - slope * var_lb
        return slope, intercept

    def __repr__(self) -> str:
        """Return string representation of the expression.

        Returns:
            String identifier of the expression.
        """
        return self.name

    def f(self, x: float) -> float:
        """Evaluates the function f(x) for the expression."""
        raise NotImplementedError(
            "Subclasses must implement the function evaluation _f(x)."
        )

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        """Solves f'(x) = m for x."""
        raise NotImplementedError("Subclasses must implement the solver for f'(x) = m.")

    def _get_deviation(self, x: float, m: float, t: float) -> float:
        """Helper method to calculate the deviation f(x) - m*x - t."""
        return self.f(x) - m * x - t

    def get_min_max_deviation(
        self, var_lb: float, var_ub: float, m: float, t: float
    ) -> Tuple[float, float]:
        """
        Calculates the min and max values of f(x) - m*x - t in [var_lb, var_ub].

        The calculation is based on checking the function's value at the
        interval boundaries and at any points within the interval where the
        derivative f'(x) equals m.
        """
        points_to_check = [var_lb, var_ub]
        critical_points = self._solve_for_f_prime_equals_m(m)
        for p in critical_points:
            if var_lb <= p <= var_ub:
                points_to_check.append(p)

        if not points_to_check:
            return float("inf"), float("-inf")

        deviations = [self._get_deviation(p, m, t) for p in points_to_check]

        return min(deviations), max(deviations)


class SquareExpression(OneDimExpression):
    """Square expression representing r = x².

    A one-dimensional expression where the representative variable equals
    the square of the input variable.
    """

    def f(self, x: float) -> float:
        return x**2

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = 2x.  2x = m => x = m/2
        return [m / 2.0]


class ExponentialExpression(OneDimExpression):
    """Exponential expression representing r = e^x.

    A one-dimensional expression where the representative variable equals
    the exponential function (e raised to the power) of the input variable.
    """

    def f(self, x: float) -> float:
        return math.exp(x)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = e^x. e^x = m => x = ln(m). Requires m > 0.
        if m > 0:
            return [math.log(m)]
        return []


class LnExpression(OneDimExpression):
    """Natural logarithm expression representing r = ln(x).

    A one-dimensional expression where the representative variable equals
    the natural logarithm of the input variable.
    """

    def f(self, x: float) -> float:
        if x == 0:
            return -s.StaticSettings.infinity
        return math.log(x)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = 1/x. 1/x = m => x = 1/m. Requires m != 0.
        if m != 0:
            return [1.0 / m]
        return []


class SquareRootExpression(OneDimExpression):
    """Square root expression representing r = √x.

    A one-dimensional expression where the representative variable equals
    the square root of the input variable.
    """

    def f(self, x: float) -> float:
        return math.sqrt(x)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = 1/(2*sqrt(x)). 1/(2*sqrt(x)) = m => x = (1/(2m))^2. Requires m > 0.
        if m > 0:
            return [(0.5 / m) ** 2]
        return []


class SineExpression(OneDimExpression):
    """Sine expression representing r = sin(x).

    A one-dimensional expression where the representative variable equals
    the sine of the input variable (in radians).
    """

    def f(self, x: float) -> float:
        return math.sin(x)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = cos(x). cos(x) = m. Requires |m| <= 1.
        if not -1.0 <= m <= 1.0:
            return []

        solutions = []
        x0 = math.acos(m)  # Principal value in [0, pi]

        # General solutions are 2*k*pi +/- x0. Find all in variable's bounds.
        var_lb, var_ub = self.variable.lb, self.variable.ub

        # Type 1 solutions: 2*k*pi + x0
        k_min = (var_lb - x0) / (2 * math.pi)
        k_max = (var_ub - x0) / (2 * math.pi)
        for k in range(math.ceil(k_min), math.floor(k_max) + 1):
            solutions.append(2 * k * math.pi + x0)

        # Type 2 solutions: 2*k*pi - x0 (if distinct)
        if x0 > 1e-9:  # Avoid duplicates when x0 is 0
            k_min = (var_lb + x0) / (2 * math.pi)
            k_max = (var_ub + x0) / (2 * math.pi)
            for k in range(math.ceil(k_min), math.floor(k_max) + 1):
                solutions.append(2 * k * math.pi - x0)

        return list(set(solutions))


class CosineExpression(OneDimExpression):
    """Cosine expression representing r = cos(x).

    A one-dimensional expression where the representative variable equals
    the cosine of the input variable (in radians).
    """

    def f(self, x: float) -> float:
        return math.cos(x)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = -sin(x). -sin(x) = m => sin(x) = -m. Requires |m| <= 1.
        m_prime = -m
        if not -1.0 <= m_prime <= 1.0:
            return []

        solutions = []
        x0 = math.asin(m_prime)  # Principal value in [-pi/2, pi/2]

        # General solutions for sin(x)=y are 2k*pi+x0 and (2k+1)*pi-x0
        var_lb, var_ub = self.variable.lb, self.variable.ub

        # Type 1: 2*k*pi + x0
        k_min = (var_lb - x0) / (2 * math.pi)
        k_max = (var_ub - x0) / (2 * math.pi)
        for k in range(math.ceil(k_min), math.floor(k_max) + 1):
            solutions.append(2 * k * math.pi + x0)

        # Type 2: (2*k+1)*pi - x0
        pi_minus_x0 = math.pi - x0
        k_min = (var_lb - pi_minus_x0) / (2 * math.pi)
        k_max = (var_ub - pi_minus_x0) / (2 * math.pi)
        for k in range(math.ceil(k_min), math.floor(k_max) + 1):
            solutions.append(2 * k * math.pi + pi_minus_x0)

        return list(set(solutions))


class LogExpression(OneDimExpression):
    """Base-10 logarithm expression representing r = log10(x).

    A one-dimensional expression where the representative variable equals
    the base-10 logarithm of the input variable.
    """

    def f(self, x: float) -> float:
        if x == 0:
            return -s.StaticSettings.infinity
        return math.log10(x)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = 1/(x*ln(10)). 1/(x*ln(10)) = m => x = 1/(m*ln(10)).
        if m != 0:
            return [1.0 / (m * math.log(10))]
        return []


class AbsExpression(OneDimExpression):
    """Absolute value expression representing r = |x|.

    A one-dimensional expression where the representative variable equals
    the absolute value of the input variable.
    """

    def handle_abs_expression(self, model_data: ModelData) -> None:
        """Add constraints to model to represent the absolute value function."""
        binary_abs_variable = model_data.add_variable(
            var.Variable(f"abs_bin_{self.variable.name}", var_type="B")
        )
        model_data.add_constraint(
            con.Constraint(
                f"abs_neg_{self.variable.name}",
                con_type=">=",
                variables=[
                    (1.0, self.representative_variable),
                    (1.0, self.variable),
                ],
            )
        )
        model_data.add_constraint(
            con.Constraint(
                f"abs_pos_{self.variable.name}",
                con_type=">=",
                variables=[
                    (1.0, self.representative_variable),
                    (-1.0, self.variable),
                ],
            )
        )
        model_data.add_constraint(
            con.Constraint(
                f"abs_neg_bigm_{self.variable.name}",
                con_type="<=",
                variables=[
                    (1.0, self.representative_variable),
                    (1.0, self.variable),
                    (-self.variable.ub + self.variable.lb, binary_abs_variable),
                ],
            )
        )
        model_data.add_constraint(
            con.Constraint(
                f"abs_pos_bigm_{self.variable.name}",
                con_type="<=",
                variables=[
                    (1.0, self.representative_variable),
                    (-1.0, self.variable),
                    (2 * self.variable.ub, binary_abs_variable),
                ],
                rhs=2 * self.variable.ub,
            )
        )

    def f(self, x: float) -> float:
        return abs(x)

    def get_min_max_deviation(
        self, var_lb: float, var_ub: float, m: float, t: float
    ) -> Tuple[float, float]:
        """
        Specialized calculation for |x| - m*x - t.
        The function is piecewise linear, so extrema are at the boundaries
        or the "kink" at x=0.
        """
        points_to_check = [var_lb, var_ub]
        if var_lb <= 0.0 <= var_ub:
            points_to_check.append(0.0)

        deviations = [abs(p) - m * p - t for p in points_to_check]
        return min(deviations), max(deviations)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        return []


class TangensHExpression(OneDimExpression):
    """Hyperbolic tangent expression representing r = tanh(x).

    A one-dimensional expression where the representative variable equals
    the hyperbolic tangent of the input variable.
    """

    def f(self, x: float) -> float:
        return math.tanh(x)

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = 1 - tanh^2(x). 1 - tanh^2(x) = m => tanh^2(x) = 1 - m.
        # f'(x) is in (0, 1], so m must be in (0, 1].
        if not 0.0 < m <= 1.0:
            return []

        val_squared = 1.0 - m
        val = math.sqrt(val_squared)

        # x = atanh(y) = 0.5 * log((1+y)/(1-y))
        sol1 = math.atanh(val)
        sol2 = -sol1  # atanh is an odd function
        return [sol1, sol2]


class InverseExpression(OneDimExpression):
    """Inverse expression representing r = x^-1

    A one-dimensional expression where the representative variable equals
    the inverse of the input variable.
    """

    def f(self, x: float) -> float:
        if x == 0:
            return s.StaticSettings.infinity
        return 1.0 / x

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = -1/x^2. -1/x^2 = m => x^2 = -1/m. Requires m < 0.
        if m >= 0:
            return []
        val = math.sqrt(-1.0 / m)
        return [val, -val]
