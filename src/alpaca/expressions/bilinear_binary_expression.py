# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from alpaca.model_data import variable as var, constraint as con
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class BilinearBinaryExpression:
    """Represents a bilinear expression z = x * y where x and y are binary variables.

    Attributes:
        name: Identifier for the expression
        first_var: First variable (x) in the expression
        second_var: Second variable (y) in the expression
        representative_variable: Variable representing the product (z)
    """

    def __init__(
        self,
        name: str,
        model_data: ModelData,
        variables: list[var.Variable],
        representative_variable: var.Variable | None = None,
    ):
        """Initialize bilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            variables: Tuple containing the two input variables (x, y)
            representative_variable: Optional existing variable to represent product
        """
        self.name = name
        self.first_var, self.second_var = variables
        self.model_data = model_data
        self.representative_variable = model_data.add_variable(
            var.Variable(lsf.representative_variable_name(name)),
            representative_variable=representative_variable,
        )
        self.representative_variable.var_type = lsf.var_type_binary()
        self.representative_variable.lb = 0.0
        self.representative_variable.ub = 1.0
        self._add_mc_cormick_constraints()

    def _add_mc_cormick_constraints(self):
        """Add McCormick constraints for bilinear binary expressions."""
        self.model_data.add_constraint(
            con.Constraint(
                lsf.con_name_mc_cormick_binary_ub_ub(self.name),
                con_type=lsf.constraint_geq(),
                variables=[
                    (-1.0, self.first_var),
                    (-1.0, self.second_var),
                    (1.0, self.representative_variable),
                ],
                rhs=-1.0,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                lsf.con_name_mc_cormick_binary_ub_lb(self.name),
                con_type=lsf.constraint_geq(),
                variables=[(1.0, self.first_var), (-1.0, self.representative_variable)],
                rhs=0.0,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                lsf.con_name_mc_cormick_binary_lb_ub(self.name),
                con_type=lsf.constraint_geq(),
                variables=[
                    (1.0, self.second_var),
                    (-1.0, self.representative_variable),
                ],
                rhs=0.0,
            )
        )
