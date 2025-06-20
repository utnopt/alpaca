# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var
import alpaca.settings as s


class BilinearExpression:
    """Represents a bilinear expression z = x * y.

    Attributes:
        name: Identifier for the expression
        first_var: First variable (x) in the expression
        second_var: Second variable (y) in the expression
        representative_variable: Variable representing the product (z)
    """

    def __init__(
        self,
        name: str,
        model_data,
        variables: tuple[var.Variable, var.Variable],
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
        # pylint: disable=duplicate-code
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.variables.setdefault(
                f"r_{name}", var.Variable(f"r_{name}", lb=-s.StaticSettings.infinity)
            )
        )
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""

    def __repr__(self):
        return self.name
