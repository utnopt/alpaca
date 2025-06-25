# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var
import alpaca.settings as s


class MultilinearExpression:
    """Represents a multilinear expression with multiple variables.

    A multilinear expression is a product of multiple variables where each variable appears
    with degree at most 1. The expression is represented by a set of variables and a
    representative variable that holds the result of the multilinear operation.

    Attributes:
        name: Identifier for the expression
        variables: List of variables involved in the multilinear expression
        representative_variable: Variable representing the result of the expression
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variables: list[var.Variable],
        representative_variable: var.Variable | None = None,
    ):
        """Initialize multilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            variables: List of variables involved in the multilinear expression
            representative_variable: Optional existing variable to represent result
        """
        self.name = name
        self.variables = variables
        # pylint: disable=duplicate-code
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.variables.setdefault(
                f"r_{name}", var.Variable(f"r_{name}", lb=-s.StaticSettings.infinity)
            )
        )
        self.representative_variable.add_nonlinearity_to_occurring_in("multilinear")
        for variable in self.variables:
            variable.add_nonlinearity_to_occurring_in("multilinear")

    def apply_piecewise_linear_approximation(self) -> None:
        """Apply piecewise linear approximation to the multilinear expression."""

    def __repr__(self) -> str:
        return self.name
