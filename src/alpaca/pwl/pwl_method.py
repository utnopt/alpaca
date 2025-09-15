# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""

from alpaca.model_data import variable as var, constraint as con
import alpaca.expressions.one_dim_expression as ode


class PWLMethod:
    """Represents the a method for pwl."""

    def __init__(self, variable: var.Variable):
        self.variable = variable
        self.pwl_variables_binary: list[var.Variable] = []
        self.pwl_variables_continuous: list[var.Variable] = []
        self.pwl_constraints: list[con.Constraint] = []
        self._represent_domain()

    def _represent_domain(self):
        """Represent the domain of a variable using the specific PWL method."""

    def couple_domain_to_function_value_one_dim(
        self, expression: ode.OneDimExpression, approximation=False
    ) -> list[con.Constraint]:
        """Couple the domain representation to the function value representation."""
