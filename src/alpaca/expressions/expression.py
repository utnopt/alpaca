# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var


class Expression:
    """Represents an expression z = ...

    Attributes:
        name: Identifier for the expression
        representative_variable: Variable representing the product (z)
        level: Level of the expression in expression tree
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize bilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            level: Level of the expression in expression tree
            representative_variable: Optional existing variable to represent product
        """
        self.name = name
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.add_variable(var.Variable(f"r_{name}"))
        )
        self.level = level

    def propagate_variable_bounds(self):
        """Propagate variables bounds."""

    def apply_piecewise_constant_approximation(self):
        """Apply piecewise constant approximation to the expression."""

    def __repr__(self):
        return self.name
