# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""
@authors: kuen,
"""
import math

from alpaca.model_data import variable as var, constraint as con
import alpaca.expressions.expression as exn


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
        model_data: "ModelData",
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

    def propagate_variable_bounds(self):
        """Propagate variables bounds."""

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the expression.

        This method is implemented by subclasses to create appropriate linear constraints
        that approximate the nonlinear function represented by this expression.
        """
        if self.model_data.settings.pwl_method == "multiple-choice":
            self._apply_multiple_choice_method()

    def _apply_multiple_choice_method(self):
        reference_points = self._get_reference_points_multiple_choice()
        self.model_data.constraints.update(
            {
                f"mc_{self.name}": con.Constraint(
                    f"mc_{self.name}",
                    con_type="==",
                    variables=[(-1.0, self.representative_variable)]
                    + [
                        (
                            (reference_points[i + 1] - reference_point)
                            / (
                                self.variable.breakpoints[i + 1]
                                - self.variable.breakpoints[i]
                            ),
                            self.variable.pwl_variables_continuous[i],
                        )
                        for i, reference_point in enumerate(reference_points[:-1])
                    ]
                    + [
                        (reference_point, self.variable.pwl_variables_binary[i])
                        for i, reference_point in enumerate(reference_points[:-1])
                    ],
                )
            }
        )

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return []

    def __repr__(self) -> str:
        """Return string representation of the expression.

        Returns:
            String identifier of the expression.
        """
        return self.name


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [bp * bp for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        self.representative_variable.ub = max(self.variable.ub, -self.variable.lb) ** 2
        self.representative_variable.lb = min(self.variable.lb**2, 0)


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [math.exp(bp) for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        self.representative_variable.lb = math.exp(self.variable.lb)
        self.representative_variable.ub = math.exp(self.variable.ub)


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [math.log(bp) for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        assert self.variable.lb > 0, "Invalid bounds for ln expression"
        self.representative_variable.lb = math.log(self.variable.lb)
        self.representative_variable.ub = math.log(self.variable.ub)


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [math.sqrt(bp) for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        assert self.variable.lb >= 0, "Invalid bounds for sqrt expression"
        self.representative_variable.lb = math.sqrt(self.variable.lb)
        self.representative_variable.ub = math.sqrt(self.variable.ub)


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [math.sin(bp) for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        self.representative_variable.lb = -1.0
        self.representative_variable.ub = 1.0


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [math.cos(bp) for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        self.representative_variable.lb = -1.0
        self.representative_variable.ub = 1.0


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [math.log10(bp) for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        assert self.variable.lb > 0, "Invalid bounds for log10 expression"
        self.representative_variable.lb = math.log10(self.variable.lb)
        self.representative_variable.ub = math.log10(self.variable.ub)


class AbsExpression(OneDimExpression):
    """Absolute value expression representing r = |x|.

    A one-dimensional expression where the representative variable equals
    the absolute value of the input variable.

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which absolute value is applied.
        representative_variable: Variable representing the result of the expression.
    """

    def apply_piecewise_linear_approximation(self):
        binary_abs_variable = var.Variable(
            f"abs_bin_{self.variable.name}", var_type="B"
        )
        self.model_data.variables.update(
            {f"abs_bin_{self.variable.name}": binary_abs_variable}
        )
        self.model_data.constraints.update(
            {
                f"abs_neg_{self.variable.name}": con.Constraint(
                    f"abs_neg_{self.variable.name}",
                    con_type=">=",
                    variables=[
                        (1.0, self.representative_variable),
                        (1.0, self.variable),
                    ],
                ),
                f"abs_pos_{self.variable.name}": con.Constraint(
                    f"abs_pos_{self.variable.name}",
                    con_type=">=",
                    variables=[
                        (1.0, self.representative_variable),
                        (-1.0, self.variable),
                    ],
                ),
                f"abs_neg_bigm_{self.variable.name}": con.Constraint(
                    f"abs_neg_bigm_{self.variable.name}",
                    con_type="<=",
                    variables=[
                        (1.0, self.representative_variable),
                        (1.0, self.variable),
                        (-self.variable.ub + self.variable.lb, binary_abs_variable),
                    ],
                ),
                f"abs_pos_bigm_{self.variable.name}": con.Constraint(
                    f"abs_pos_bigm_{self.variable.name}",
                    con_type="<=",
                    variables=[
                        (1.0, self.representative_variable),
                        (-1.0, self.variable),
                        (2 * self.variable.ub, binary_abs_variable),
                    ],
                    rhs=2 * self.variable.ub,
                ),
            }
        )

    def propagate_variable_bounds(self) -> None:
        self.representative_variable.lb = max(self.variable.lb, 0)
        self.representative_variable.ub = max(self.variable.ub, -self.variable.lb)


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
        model_data: "ModelData",
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

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [math.tanh(bp) for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        self.representative_variable.lb = math.tanh(self.variable.lb)
        self.representative_variable.ub = math.tanh(self.variable.ub)


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
        model_data: "ModelData",
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
        self.variable.add_nonlinearity_to_occurring_in("tanh")

    def _get_reference_points_multiple_choice(self) -> list[float]:
        return [1 / bp for bp in self.variable.breakpoints]

    def propagate_variable_bounds(self) -> None:
        assert (self.variable.lb < 0 and self.variable.ub < 0) or (
            self.variable.lb > 0 and self.variable.ub > 0
        ), "invalid bounds for inverse expression"
        self.representative_variable.lb = 1 / self.variable.ub
        self.representative_variable.ub = 1 / self.variable.lb
