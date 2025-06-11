# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var


class OneDimExpression:
    """One dimensional expression. representative_variable = function(variable)."""

    def __init__(self, name: str, model_data, representative_variable=None):
        self.name = name
        # pylint: disable=duplicate-code
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.variables.setdefault(f"r_{name}", var.Variable(f"r_{name}"))
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
    """Tengens hyperbolicus expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variable: var.Variable,
        representative_variable=None,
    ):
        super().__init__(name, model_data, representative_variable)
        self.variable = variable
