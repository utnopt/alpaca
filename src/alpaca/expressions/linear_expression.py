# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var, constraint as con
import alpaca.expressions.expression as exn


class LinearExpression(exn.Expression):
    """Represents a linear expression z = ax + t.

    Attributes:
        name: Identifier for the expression
        variables: Tuples of coeff, variable
        constant: Constant value t
        level: Level of expression in expression tree
        model_data: Model data
        representative_variable: Variable representing the product (z)
    """

    def __init__(
        self,
        name: str,
        model_data,
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize linear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            level: Level of expression in expression tree
            representative_variable: Optional existing variable to represent product
        """
        super().__init__(name, model_data, level, representative_variable)
        self.variables: list[tuple[float, var.Variable]] = []
        self.constant: float = 0.0
        self.model_data: "ModelData" = model_data

    def add_constraint_from_linear_expression(self):
        """Add constraint from linear expression."""
        self.model_data.add_constraint(
            con.Constraint(
                f"c_{self.name}",
                con_type="==",
                rhs=-self.constant,
                variables=self.variables + [(-1.0, self.representative_variable)],
            )
        )

    def propagate_variable_bounds(self):
        """Propagate variables bounds."""
        lb = self.constant + sum(
            min(coeff * variable.lb, coeff * variable.ub)
            for coeff, variable in self.variables
        )
        ub = self.constant + sum(
            max(coeff * variable.lb, coeff * variable.ub)
            for coeff, variable in self.variables
        )
        self.representative_variable.lb = max(lb, self.representative_variable.lb)
        self.representative_variable.ub = min(ub, self.representative_variable.ub)
