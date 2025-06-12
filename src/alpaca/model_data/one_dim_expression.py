# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var
import alpaca.settings as s


class OneDimExpression:
    """One dimensional expression. representative_variable = function(variable)."""

    def __init__(self, name: str, model_data, representative_variable=None):
        self.name = name
        # pylint: disable=duplicate-code
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.variables.setdefault(
                f"r_{name}", var.Variable(f"r_{name}", lb=-s.StaticSettings.infinity)
            )
        )

    def __repr__(self):
        return self.name


class SquareExpression(OneDimExpression):
    """Square expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class DivisionExpression(OneDimExpression):
    """Division expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable


class ExponentialExpression(OneDimExpression):
    """Exponential expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class LnExpression(OneDimExpression):
    """Ln expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class SquareRootExpression(OneDimExpression):
    """Square root expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class SinusExpression(OneDimExpression):
    """Sinus expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class CosinusExpression(OneDimExpression):
    """Cosinus expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class LogExpression(OneDimExpression):
    """Log 10 expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class AbsExpression(OneDimExpression):
    """Absolute value expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable


class InverseExpression(OneDimExpression):
    """Inverse expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable


class PowerExpression(OneDimExpression):
    """Power expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""


class MinExpression(OneDimExpression):
    """Min expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable


class TangensHExpression(OneDimExpression):
    """Tangens hyperbolicus expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""
