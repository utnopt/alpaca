# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var


class OneDimExpression:
    """One dimensional expression."""

    def __init__(self, name: str, model_data):
        self.name = name
        self.representative_variable = var.Variable(f"r_{self.name}")
        model_data.variables[f"r_{self.name}"] = self.representative_variable

    def __repr__(self):
        return self.name


class QuadraticExpression(OneDimExpression):
    """Quadratic expression."""

    def __init__(self, name: str, model_data, variable: var.Variable):
        super().__init__(name, model_data)
        self.variable = variable
