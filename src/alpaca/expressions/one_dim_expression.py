# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""
@authors: kuen,
"""
from __future__ import annotations
import math
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from alpaca.model_data import variable as var, constraint as con
import alpaca.expressions.expression as exn
import alpaca.settings as s
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

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
        self.variable.add_nonlinearity_to_occurring_in(
            self.__class__.__name__, self.__class__
        )

    @classmethod
    def get_linear_approximation_function_parameters_for_segment(
        cls, var_lb: float, var_ub: float
    ) -> tuple[float, float]:
        """Calculate slope and intercept of the linear approximation over [var_lb, var_ub]."""
        slope = (cls.f(var_ub) - cls.f(var_lb)) / (var_ub - var_lb)
        intercept = cls.f(var_lb) - slope * var_lb
        return slope, intercept

    def __repr__(self) -> str:
        """Return string representation of the expression.

        Returns:
            String identifier of the expression.
        """
        return self.name

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluates the function f(x) for the expression."""
        raise NotImplementedError(lsf.error_subclasses_must_implement_method())

    @classmethod
    def f_derivative(cls, x: float) -> float:
        """Evaluates the function f'(x) for the expression."""
        raise NotImplementedError(lsf.error_subclasses_must_implement_method())

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        """Solves f'(x) = m for x. Checks in interval lb to ub."""
        raise NotImplementedError(lsf.error_subclasses_must_implement_method())

    @classmethod
    def get_deviation(cls, x: float, m: float, t: float) -> float:
        """Helper method to calculate the deviation f(x) - m*x - t."""
        return cls.f(x) - m * x - t

    @classmethod
    def get_min_max_deviation(
        cls, var_lb: float, var_ub: float, m: float, t: float
    ) -> Tuple[float, float]:
        """
        Calculates the min and max values of f(x) - m*x - t in [var_lb, var_ub].

        The calculation is based on checking the function's value at the
        interval boundaries and at any points within the interval where the
        derivative f'(x) equals m.
        """
        points_to_check = [var_lb, var_ub]
        critical_points = cls._solve_for_f_prime_equals_m_in_interval(m, var_lb, var_ub)
        points_to_check.extend(critical_points)

        if not points_to_check:
            return float(lsf.numpy_infinity()), -float(lsf.numpy_infinity())

        deviations = [cls.get_deviation(p, m, t) for p in points_to_check]

        return min(deviations), max(deviations)


class SquareExpression(OneDimExpression):
    """Square expression representing r = x².

    A one-dimensional expression where the representative variable equals
    the square of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for square expression."""
        return lsf.nonlinearity_type_square()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        try:
            return x**2
        except OverflowError:
            return s.StaticSettings.infinity

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return 2 * x

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = 2x.  2x = m => x = m/2
        return [p for p in [m / 2.0] if lb <= p <= ub]


class ExponentialExpression(OneDimExpression):
    """Exponential expression representing r = e^x.

    A one-dimensional expression where the representative variable equals
    the exponential function (e raised to the power) of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for exponential expression."""
        return lsf.nonlinearity_type_exp()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        try:
            return math.exp(x)
        except OverflowError:
            return s.StaticSettings.infinity

    @classmethod
    def f_derivative(cls, x: float) -> float:
        try:
            return math.exp(x)
        except OverflowError:
            return s.StaticSettings.infinity

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = e^x. e^x = m => x = ln(m). Requires m > 0.
        if m > 0:
            return [p for p in [math.log(m)] if lb <= p <= ub]
        return []


class LnExpression(OneDimExpression):
    """Natural logarithm expression representing r = ln(x).

    A one-dimensional expression where the representative variable equals
    the natural logarithm of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for natural logarithm expression."""
        return lsf.nonlinearity_type_ln()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        if x <= 0:
            return -s.StaticSettings.infinity
        return math.log(x)

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return 1 / x

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = 1/x. 1/x = m => x = 1/m. Requires m != 0.
        if m != 0:
            return [p for p in [1.0 / m] if lb <= p <= ub]
        return []


class SquareRootExpression(OneDimExpression):
    """Square root expression representing r = √x.

    A one-dimensional expression where the representative variable equals
    the square root of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for square root expression."""
        return lsf.nonlinearity_type_sqrt()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        if x < 0:
            return -s.StaticSettings.infinity
        return math.sqrt(x)

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return 0 if x == 0 else 1 / (2 * math.sqrt(x))

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = 1/(2*sqrt(x)). 1/(2*sqrt(x)) = m => x = (1/(2m))^2. Requires m > 0.
        if m > 0:
            return [p for p in [(0.5 / m) ** 2] if lb <= p <= ub]
        return []


class SineExpression(OneDimExpression):
    """Sine expression representing r = sin(x).

    A one-dimensional expression where the representative variable equals
    the sine of the input variable (in radians).
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for sine expression."""
        return lsf.nonlinearity_type_sin()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        return math.sin(x)

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return math.cos(x)

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = cos(x). cos(x) = m. Requires |m| <= 1.
        if not -1.0 <= m <= 1.0:
            return []

        solutions = []
        x0 = math.acos(m)  # Principal value in [0, pi]

        # General solutions are 2*k*pi +/- x0. Find all in variable's bounds.
        var_lb, var_ub = lb, ub

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

        return [p for p in set(solutions) if lb <= p <= ub]


class CosineExpression(OneDimExpression):
    """Cosine expression representing r = cos(x).

    A one-dimensional expression where the representative variable equals
    the cosine of the input variable (in radians).
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for cosine expression."""
        return lsf.nonlinearity_type_cos()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        return math.cos(x)

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return -math.sin(x)

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = -sin(x). -sin(x) = m => sin(x) = -m. Requires |m| <= 1.
        m_prime = -m
        if not -1.0 <= m_prime <= 1.0:
            return []

        solutions = []
        x0 = math.asin(m_prime)  # Principal value in [-pi/2, pi/2]

        # General solutions for sin(x)=y are 2k*pi+x0 and (2k+1)*pi-x0
        var_lb, var_ub = lb, ub

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

        return [p for p in set(solutions) if lb <= p <= ub]


class LogExpression(OneDimExpression):
    """Base-10 logarithm expression representing r = log10(x).

    A one-dimensional expression where the representative variable equals
    the base-10 logarithm of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for base-10 logarithm expression."""
        return lsf.nonlinearity_type_log10()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        if x <= 0:
            return -s.StaticSettings.infinity
        return math.log10(x)

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return 1 / (x * math.log(10))

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = 1/(x*ln(10)). 1/(x*ln(10)) = m => x = 1/(m*ln(10)).
        if m != 0:
            return [p for p in [1.0 / (m * math.log(10))] if lb <= p <= ub]
        return []


class AbsExpression(OneDimExpression):
    """Absolute value expression representing r = |x|.

    A one-dimensional expression where the representative variable equals
    the absolute value of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for absolute value expression."""
        return lsf.nonlinearity_type_xabsx()

    def handle_abs_expression(self, model_data: ModelData) -> None:
        """Add constraints to model to represent the absolute value function."""
        binary_abs_variable = model_data.add_variable(
            var.Variable(
                lsf.var_name_binary_abs_reformulation(self.variable.name),
                var_type=lsf.var_type_binary(),
            )
        )
        model_data.add_constraint(
            con.LinearConstraint(
                lsf.con_name_abs_reformulation_negative(self.variable.name),
                con_type=lsf.constraint_geq(),
                variables=[
                    (1.0, self.representative_variable),
                    (1.0, self.variable),
                ],
            )
        )
        model_data.add_constraint(
            con.LinearConstraint(
                lsf.con_name_abs_reformulation_positive(self.variable.name),
                con_type=lsf.constraint_geq(),
                variables=[
                    (1.0, self.representative_variable),
                    (-1.0, self.variable),
                ],
            )
        )
        model_data.add_constraint(
            con.LinearConstraint(
                lsf.con_name_abs_reformulation_negative_big_m(self.variable.name),
                con_type=lsf.constraint_leq(),
                variables=[
                    (1.0, self.representative_variable),
                    (-1.0, self.variable),
                    (2 * self.variable.lb, binary_abs_variable),
                ],
            )
        )
        model_data.add_constraint(
            con.LinearConstraint(
                lsf.con_name_abs_reformulation_positive_big_m(self.variable.name),
                con_type=lsf.constraint_leq(),
                variables=[
                    (1.0, self.representative_variable),
                    (1.0, self.variable),
                    (2 * self.variable.ub, binary_abs_variable),
                ],
                rhs=2 * self.variable.ub,
            )
        )

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        return abs(x)

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return 1.0 if x > 0 else -1.0 if x < 0 else 0.0

    @classmethod
    def get_min_max_deviation(
        cls, var_lb: float, var_ub: float, m: float, t: float
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

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        return []


class TangensHExpression(OneDimExpression):
    """Hyperbolic tangent expression representing r = tanh(x).

    A one-dimensional expression where the representative variable equals
    the hyperbolic tangent of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for hyperbolic tangent expression."""
        return lsf.nonlinearity_type_tanh()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        return math.tanh(x)

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return 1 - math.tanh(x) ** 2

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = 1 - tanh^2(x). 1 - tanh^2(x) = m => tanh^2(x) = 1 - m.
        # f'(x) is in (0, 1], so m must be in (0, 1].
        if not 0.0 < m <= 1.0:
            return []

        val_squared = 1.0 - m
        val = math.sqrt(val_squared)

        # x = atanh(y) = 0.5 * log((1+y)/(1-y))
        sol1 = math.atanh(val)
        sol2 = -sol1  # atanh is an odd function
        return [p for p in [sol1, sol2] if lb <= p <= ub]


class InverseExpression(OneDimExpression):
    """Inverse expression representing r = x^-1

    A one-dimensional expression where the representative variable equals
    the inverse of the input variable.
    """

    @staticmethod
    def nonlinearity_type() -> str:
        """Return the nonlinearity type for inverse expression."""
        return lsf.nonlinearity_type_inverse()

    @classmethod
    def f(cls, x: float | np.ndarray) -> float | np.ndarray:
        if x == 0:
            return s.StaticSettings.infinity
        return 1.0 / x

    @classmethod
    def f_derivative(cls, x: float) -> float:
        return -1.0 / (x**2)

    @classmethod
    def _solve_for_f_prime_equals_m_in_interval(
        cls, m: float, lb: float, ub: float
    ) -> List[float]:
        # f'(x) = -1/x^2. -1/x^2 = m => x^2 = -1/m. Requires m < 0.
        if m >= 0:
            return []
        val = math.sqrt(-1.0 / m)
        return [p for p in [val, -val] if lb <= p <= ub]


class PowerExpression(OneDimExpression):
    """
    Power expression representing r = x^y.

    A one-dimensional expression where the representative variable equals
    the input variable raised to the power of 'y'.

    The key difference from other OneDimExpressions is that 'y' is
    an instance-specific parameter. Therefore, 'f' and 'f_derivative'
    are instance methods, not class methods.
    """

    def __init__(
        self,
        name: str,
        model_data: ModelData,
        variable: var.Variable,
        level: int,
        y: float,
        representative_variable: var.Variable | None = None,
    ):
        """
        Initialize the Power expression.

        Args:
            name: Unique identifier for the expression.
            model_data: The ModelData object.
            variable: The input variable (x).
            level: Level of expression in expression tree.
            y: The exponent for the power function (x^y).
            representative_variable: Optional existing variable to represent
                the expression result. If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.y = y
        self.variable.add_nonlinearity_to_occurring_in(
            f"{self.__class__.__name__}_{y}", self
        )

    def nonlinearity_type(self) -> str:
        """Return the nonlinearity type for power expression."""
        return f"{lsf.nonlinearity_type_power()}_{self.y}"

    def f(  # pylint: disable=arguments-differ
        self, x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Evaluates the function f(x) = x^y.
        """
        try:
            if abs(x) < s.StaticSettings.feasibility_tolerance:
                return 0.0
            return np.power(x, self.y)
        except OverflowError:
            return s.StaticSettings.infinity

    def f_derivative(self, x: float) -> float:  # pylint: disable=arguments-differ
        """
        Evaluates the function f'(x) = y * x^(y-1).
        Note: This is an INSTANCE method, not a @classmethod.
        """
        try:
            if abs(x) < s.StaticSettings.feasibility_tolerance:
                return 0.0
            return self.y * math.pow(x, self.y - 1)
        except OverflowError:
            return s.StaticSettings.infinity

    def _solve_for_f_prime_equals_m_in_interval(  # pylint: disable=arguments-differ
        self, m: float, lb: float, ub: float
    ) -> List[float]:
        """
        Solves f'(x) = m for x, i.e., y * x^(y-1) = m.
        """
        if self.y == 0:
            return []
        val = m / self.y
        exponent = 1.0 / (self.y - 1.0)
        solutions = []
        if val < 0 and exponent % 1 != 0:
            return []
        if val == 0 and exponent < 0:
            return []
        x_sol = math.pow(val, exponent)
        solutions.append(x_sol)
        if (
            abs((self.y - 1) % 2) < s.StaticSettings.feasibility_tolerance
            and (self.y - 1) > 0
            and val > 0
        ):
            if x_sol != 0:
                solutions.append(-x_sol)
        elif (
            abs((1.0 / (self.y - 1.0)) % 2) < s.StaticSettings.feasibility_tolerance
            and val > 0
        ):
            if x_sol != 0:
                solutions.append(-x_sol)
        return [p for p in set(solutions) if lb <= p <= ub]
