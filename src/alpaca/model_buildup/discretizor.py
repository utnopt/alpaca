# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class Discretizor:
    """Handles the discretization of variables."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data
        self.settings = model_data.settings

    def discretize_variables(self) -> None:
        """Creates piecewise linear approximations for variables marked for discretization."""
        pwl_variables = []
        pwl_constraints = []
        for variable in self.model_data.variables.values():
            if variable.is_discretized and variable.var_type != "B":
                variable.add_binary_pwl(
                    self.settings.number_of_breakpoints,
                    self.settings.pwl_method,
                )
                variable.add_continuous_pwl(self.settings.pwl_method)
                pwl_variables.extend(
                    variable.pwl_variables_binary + variable.pwl_variables_continuous
                )
                pwl_constraints.extend(variable.pwl_constraints)
        for variable in pwl_variables:
            self.model_data.add_variable(variable)
        for constraint in pwl_constraints:
            self.model_data.add_constraint(constraint)
