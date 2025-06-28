# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var
import alpaca.expressions.expression as exn


class MultilinearExpression(exn.Expression):
    """Represents a multilinear expression with multiple variables.

    A multilinear expression is a product of multiple variables where each variable appears
    with degree at most 1. The expression is represented by a set of variables and a
    representative variable that holds the result of the multilinear operation.

    Attributes:
        name: Identifier for the expression
        variables: List of variables involved in the multilinear expression
        level: Level of expression in expression tree
        representative_variable: Variable representing the result of the expression
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variables: list[var.Variable],
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize multilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            variables: List of variables involved in the multilinear expression
            level: Level of expression in expression tree
            representative_variable: Optional existing variable to represent result
        """
        super().__init__(name, model_data, level, representative_variable)
        self.variables = variables
        self.representative_variable.add_nonlinearity_to_occurring_in("multilinear")
        for variable in self.variables:
            variable.add_nonlinearity_to_occurring_in("multilinear")

    def apply_piecewise_constant_approximation(self):
        """Apply piecewise constant approximation."""

    def propagate_variable_bounds(self):
        """Propagate variables bounds."""
        lb, ub = self.variables[0].lb, self.variables[0].ub
        for v in self.variables[1:]:
            candidates = [lb * v.lb, lb * v.ub, ub * v.lb, ub * v.ub]
            lb, ub = min(candidates), max(candidates)
        self.representative_variable.lb = max(self.representative_variable.lb, lb)
        self.representative_variable.ub = min(self.representative_variable.lb, ub)
