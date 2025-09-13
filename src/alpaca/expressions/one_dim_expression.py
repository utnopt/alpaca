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
            model_data: Reference to the containing model data object.
            level: Level of expression in expression tree.
            representative_variable: Optional existing variable to represent the expression result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, level, representative_variable)
        self.model_data = model_data
        self.variable: var.Variable = variable

    def apply_piecewise_linear_relaxation(self, approximation=False) -> None:
        """Apply piecewise linear relaxation to the expression.
        If approximation is True, the approximation error term is set to 0
        """
        if self.model_data.settings.pwl_method == "multiple-choice":
            if approximation:
                self._apply_multiple_choice_method_approximation()
            else:
                self._apply_multiple_choice_method_relaxation()

    def _apply_multiple_choice_method_approximation(self):
        variables_in_constraint = [(-1.0, self.representative_variable)]
        for i, bp in enumerate(self.variable.breakpoints[:-1]):
            slope, intercept = (
                self._get_linear_approximation_function_parameters_for_segment(
                    bp, self.variable.breakpoints[i + 1]
                )
            )
            variables_in_constraint.append(
                (slope, self.variable.pwl_variables_continuous[i])
            )
            variables_in_constraint.append(
                (intercept, self.variable.pwl_variables_binary[i])
            )
        self.model_data.add_constraint(
            con.Constraint(
                f"mc_{self.name}",
                con_type="==",
                variables=variables_in_constraint,
            )
        )

    def _apply_multiple_choice_method_relaxation(self):
        continuous_variables_in_constraint = [(-1.0, self.representative_variable)]
        binary_variables_in_underestimating_constraint = []
        binary_variables_in_overestimating_constraint = []
        for i, bp in enumerate(self.variable.breakpoints[:-1]):
            slope, intercept = (
                self._get_linear_approximation_function_parameters_for_segment(
                    bp, self.variable.breakpoints[i + 1]
                )
            )
            min_deviation, max_deviation = self._get_min_max_deviation(
                bp, self.variable.breakpoints[i + 1], slope, intercept
            )
            continuous_variables_in_constraint.append(
                (slope, self.variable.pwl_variables_continuous[i])
            )
            binary_variables_in_underestimating_constraint.append(
                (intercept + min_deviation, self.variable.pwl_variables_binary[i])
            )
            binary_variables_in_overestimating_constraint.append(
                (intercept + max_deviation, self.variable.pwl_variables_binary[i])
            )
        self.model_data.add_constraint(
            con.Constraint(
                f"mc_under_{self.name}",
                con_type="<=",
                variables=continuous_variables_in_constraint
                + binary_variables_in_underestimating_constraint,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"mc_over_{self.name}",
                con_type=">=",
                variables=continuous_variables_in_constraint
                + binary_variables_in_overestimating_constraint,
            )
        )

    def _get_linear_approximation_function_parameters_for_segment(
        self, var_lb: float, var_ub: float
    ) -> tuple[float, float]:
        slope = (self._f(var_ub) - self._f(var_lb)) / (var_ub - var_lb)
        intercept = self._f(var_lb) - slope * var_lb
        return slope, intercept

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [self._f(bp) for bp in self.variable.breakpoints]

    def apply_piecewise_constant_approximation(self):
        """Apply piecewise constant approximation to the expression."""

    def __repr__(self) -> str:
        """Return string representation of the expression.

        Returns:
            String identifier of the expression.
        """
        return self.name

    def _f(self, x: float) -> float:
        """Evaluates the function f(x) for the expression."""
        raise NotImplementedError(
            "Subclasses must implement the function evaluation _f(x)."
        )

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        """Solves f'(x) = m for x."""
        raise NotImplementedError("Subclasses must implement the solver for f'(x) = m.")

    def _get_deviation(self, x: float, m: float, t: float) -> float:
        """Helper method to calculate the deviation f(x) - m*x - t."""
        return self._f(x) - m * x - t

    def _get_min_max_deviation(
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable being squared.
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
        """Initialize a square expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to be squared.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("square")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = x^2.

        The lower bound of r is 0 if the interval for x contains 0, otherwise
        it's the minimum of x.lb^2 and x.ub^2. The upper bound is the maximum
        of x.lb^2 and x.ub^2.
        """
        ub = max(self.variable.ub, -self.variable.lb) ** 2
        lb = self.variable.lb**2 if self.variable.lb >= 0 else 0.0
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
        return x**2

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = 2x.  2x = m => x = m/2
        return [m / 2.0]


class ExponentialExpression(OneDimExpression):
    """Exponential expression representing r = e^x.

    A one-dimensional expression where the representative variable equals
    the exponential function (e raised to the power) of the input variable.

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable in the exponent.
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
        """Initialize an exponential expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to be used in the exponent.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("exp")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = e^x.

        Since e^x is monotonically increasing, the new bounds for r are
        [e^(x.lb), e^(x.ub)]. Handles potential OverflowError during calculation.
        """
        try:
            lb = math.exp(self.variable.lb)
            ub = math.exp(self.variable.ub)
        except OverflowError:
            return
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which logarithm is applied.
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
        """Initialize a natural logarithm expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which logarithm is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("ln")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = ln(x).

        The natural logarithm is only defined for x > 0. If this condition
        is met, the new bounds for r are [ln(x.lb), ln(x.ub)] because
        ln(x) is monotonically increasing.
        """
        if self.variable.lb <= 0:
            return
        lb = math.log(self.variable.lb)
        ub = math.log(self.variable.ub)
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which square root is applied.
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
        """Initialize a square root expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which square root is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("sqrt")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = sqrt(x).

        The square root is only defined for x >= 0. If this condition is met,
        the new bounds for r are [sqrt(x.lb), sqrt(x.ub)] because sqrt(x)
        is monotonically increasing.
        """
        if self.variable.lb < 0:
            return
        lb = math.sqrt(self.variable.lb)
        ub = math.sqrt(self.variable.ub)
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which sine function is applied.
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
        """Initialize a sine expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which sine function is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("sin")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = sin(x).

        The range of sin(x) is [-1, 1]. This method tightens the bounds of the
        representative variable to be within this range. A more precise
        propagation would consider the specific interval of x, but this
        provides a simple and correct outer approximation.
        """
        lb = -1.0
        ub = 1.0
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which cosine function is applied.
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
        """Initialize a cosine expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which cosine function is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("cos")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = cos(x).

        The range of cos(x) is [-1, 1]. This method tightens the bounds of the
        representative variable to be within this range. A more precise
        propagation would consider the specific interval of x, but this
        provides a simple and correct outer approximation.
        """
        lb = -1.0
        ub = 1.0
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which logarithm is applied.
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
        """Initialize a base-10 logarithm expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which logarithm is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("log10")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = log10(x).

        The base-10 logarithm is only defined for x > 0. If this holds,
        the new bounds for r are [log10(x.lb), log10(x.ub)] because log10(x)
        is monotonically increasing.
        """
        if self.variable.lb <= 0:
            return
        lb = math.log10(self.variable.lb)
        ub = math.log10(self.variable.ub)
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which absolute value is applied.
        representative_variable: Variable representing the result of the expression.
    """

    def apply_piecewise_linear_relaxation(self, approximation=False) -> None:
        binary_abs_variable = self.model_data.add_variable(
            var.Variable(f"abs_bin_{self.variable.name}", var_type="B")
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"abs_neg_{self.variable.name}",
                con_type=">=",
                variables=[
                    (1.0, self.representative_variable),
                    (1.0, self.variable),
                ],
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"abs_pos_{self.variable.name}",
                con_type=">=",
                variables=[
                    (1.0, self.representative_variable),
                    (-1.0, self.variable),
                ],
            )
        )
        self.model_data.add_constraint(
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
        self.model_data.add_constraint(
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

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = |x|.

        If the interval for x contains 0, the lower bound of r is 0.
        Otherwise, the lower bound is min(|x.lb|, |x.ub|). The upper
        bound is always max(|x.lb|, |x.ub|).
        """
        lb = (
            min(abs(self.variable.lb), abs(self.variable.ub))
            if self.variable.lb * self.variable.ub >= 0
            else 0
        )
        ub = max(abs(self.variable.lb), abs(self.variable.ub))
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
        return abs(x)

    def _get_min_max_deviation(
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which hyperbolic tangent is applied.
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
        """Initialize a hyperbolic tangent expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which hyperbolic tangent is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("tanh")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = tanh(x).

        Since tanh(x) is monotonically increasing, the new bounds for r are
        [tanh(x.lb), tanh(x.ub)].
        """
        lb = math.tanh(self.variable.lb)
        ub = math.tanh(self.variable.ub)
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
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

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which inverse is applied.
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
        """Initialize a inverse expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which inverse is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, variable, level, representative_variable)
        self.variable.add_nonlinearity_to_occurring_in("inverse")

    def propagate_variable_bounds(self) -> None:
        """Propagates bounds for r = 1/x.

        The inverse function is only defined if the interval for x does not
        contain 0. If the interval is strictly positive or strictly negative,
        the function is monotonically decreasing, so the new bounds for r
        are [1/x.ub, 1/x.lb].
        """
        if not (
            (self.variable.lb < 0 and self.variable.ub < 0)
            or (self.variable.lb > 0 and self.variable.ub > 0)
        ):
            return
        lb = 1 / self.variable.ub
        ub = 1 / self.variable.lb
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)

    def _f(self, x: float) -> float:
        return 1.0 / x

    def _solve_for_f_prime_equals_m(self, m: float) -> List[float]:
        # f'(x) = -1/x^2. -1/x^2 = m => x^2 = -1/m. Requires m < 0.
        if m >= 0:
            return []
        val = math.sqrt(-1.0 / m)
        return [val, -val]
