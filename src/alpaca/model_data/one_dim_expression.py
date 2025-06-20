# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var
import alpaca.settings as s


class OneDimExpression:
    """One dimensional expression representing a function of a single variable.

    This class serves as a base for various one-dimensional mathematical expressions
    where a representative variable equals some function of an input variable.
    Subclasses implement specific mathematical functions like square, exponential, etc.

    Attributes:
        name: Unique identifier for the expression.
        representative_variable: Variable representing the result of the expression.
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        representative_variable: var.Variable | None = None,
    ):
        """Initialize a one-dimensional expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            representative_variable: Optional existing variable to represent the expression result.
                If None, a new variable will be created.
        """
        self.name = name
        # pylint: disable=duplicate-code
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.variables.setdefault(
                f"r_{name}", var.Variable(f"r_{name}", lb=-s.StaticSettings.infinity)
            )
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the expression.

        This method is implemented by subclasses to create appropriate linear constraints
        that approximate the nonlinear function represented by this expression.
        """

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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the square function.

        Creates linear constraints that approximate the square function
        over the domain of the input variable using breakpoints.
        """


class DivisionExpression(OneDimExpression):
    """Division expression representing r = 1/x.

    A one-dimensional expression where the representative variable equals
    the reciprocal of the input variable.

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable being reciprocated.
        representative_variable: Variable representing the result of the expression.
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variable: var.Variable,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize a division expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to be reciprocated.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, representative_variable)
        self.variable = variable


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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the exponential function.

        Creates linear constraints that approximate the exponential function
        over the domain of the input variable using breakpoints.
        """


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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the natural logarithm function.

        Creates linear constraints that approximate the logarithm function
        over the domain of the input variable using breakpoints.
        """


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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the square root function.

        Creates linear constraints that approximate the square root function
        over the domain of the input variable using breakpoints.
        """


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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the sine function.

        Creates linear constraints that approximate the sine function
        over the domain of the input variable using breakpoints.
        """


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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the cosine function.

        Creates linear constraints that approximate the cosine function
        over the domain of the input variable using breakpoints.
        """


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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the base-10 logarithm function.

        Creates linear constraints that approximate the logarithm function
        over the domain of the input variable using breakpoints.
        """


class AbsExpression(OneDimExpression):
    """Absolute value expression representing r = |x|.

    A one-dimensional expression where the representative variable equals
    the absolute value of the input variable.

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to which absolute value is applied.
        representative_variable: Variable representing the result of the expression.
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variable: var.Variable,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize an absolute value expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to which absolute value is applied.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, representative_variable)
        self.variable = variable


class PowerExpression(OneDimExpression):
    """Power expression representing r = x^p (for some constant p).

    A one-dimensional expression where the representative variable equals
    the input variable raised to a constant power.

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable being raised to a power.
        representative_variable: Variable representing the result of the expression.
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variable: var.Variable,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize a power expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable being raised to a power.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the power function.

        Creates linear constraints that approximate the power function
        over the domain of the input variable using breakpoints.
        """


class MinExpression(OneDimExpression):
    """Minimum expression representing r = min(x, c) for some constant c.

    A one-dimensional expression where the representative variable equals
    the minimum of the input variable and a constant.

    Attributes:
        name: Unique identifier for the expression.
        variable: Input variable to compare with constant.
        representative_variable: Variable representing the result of the expression.
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variable: var.Variable,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize a minimum expression.

        Args:
            name: Unique identifier for the expression.
            model_data: Reference to the containing model data object.
            variable: Input variable to compare with constant.
            representative_variable: Optional existing variable to represent the result.
                If None, a new variable will be created.
        """
        super().__init__(name, model_data, representative_variable)
        self.variable = variable


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
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the hyperbolic tangent function.

        Creates linear constraints that approximate the hyperbolic tangent function
        over the domain of the input variable using breakpoints.
        """
