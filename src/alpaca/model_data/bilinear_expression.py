# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var


class BilinearExpression:
    """Bilinear expression."""

    def __init__(
        self, name: str, fist_var: var.Variable, second_var: var.Variable, model_data
    ):
        self.name = name
        self.first_var = fist_var
        self.second_var = second_var
        self.representative_variable = var.Variable(f"r_{self.name}")
        model_data.variables[f"r_{self.name}"] = self.representative_variable

    def __repr__(self):
        return self.name
