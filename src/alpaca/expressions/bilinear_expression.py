# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var
import alpaca.expressions.expression as exn


class BilinearExpression(exn.Expression):
    """Represents a bilinear expression z = x * y.

    Attributes:
        name: Identifier for the expression
        first_var: First variable (x) in the expression
        second_var: Second variable (y) in the expression
        level: Level of expression in expression tree
        representative_variable: Variable representing the product (z)
    """

    def __init__(
        self,
        name: str,
        model_data,
        variables: tuple[var.Variable, var.Variable],
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Initialize bilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            variables: Tuple containing the two input variables (x, y)
            level: Level of expression in expression tree
            representative_variable: Optional existing variable to represent product
        """
        super().__init__(name, model_data, level, representative_variable)
        self.first_var, self.second_var = variables
        self.representative_variable.add_nonlinearity_to_occurring_in("bilinear")
        self.first_var.add_nonlinearity_to_occurring_in("bilinear")
        self.second_var.add_nonlinearity_to_occurring_in("bilinear")

    def apply_piecewise_constant_approximation(self):
        """Apply piecewise constant approximation."""

    def propagate_variable_bounds(self):
        """Propagate variable bounds for bilinear expression."""
        p1 = self.first_var.lb * self.second_var.lb
        p2 = self.first_var.lb * self.second_var.ub
        p3 = self.first_var.ub * self.second_var.lb
        p4 = self.first_var.ub * self.second_var.ub
        self.representative_variable.lb = max(
            self.representative_variable.lb, min(p1, p2, p3, p4)
        )
        self.representative_variable.ub = min(
            self.representative_variable.ub, max(p1, p2, p3, p4)
        )
