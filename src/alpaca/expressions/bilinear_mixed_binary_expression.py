# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var, constraint as con
import alpaca.expressions.multilinear_expression as mle


class BilinearMixedBinaryExpression(mle.MultilinearExpression):
    """Represents a bilinear expression z = x * y where x is binary.
    Attributes:
        name: Identifier for the expression
        variables: List of variables involved in the bilinear expression
        level: Level of expression in expression tree
        representative_variable: Variable representing the result of the expression
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
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
        super().__init__(name, model_data, variables, level, representative_variable)
        self._add_indicator_constraints()

    def _add_indicator_constraints(self):
        """Add indicator constraints for bilinear mixed binary expressions."""
        b_var, c_var = tuple(self.variables)
        self.model_data.add_constraint(
            con.Constraint(
                f"indicator_{self.name}_1",
                con_type="<=",
                variables=[
                    (-c_var.ub, b_var),
                    (1.0, self.representative_variable),
                ],
                rhs=0.0,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"indicator_{self.name}_2",
                con_type=">=",
                variables=[
                    (-c_var.lb, b_var),
                    (1.0, self.representative_variable),
                ],
                rhs=0.0,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"indicator_{self.name}_3",
                con_type="<=",
                variables=[
                    (-1.0, c_var),
                    (-c_var.lb, b_var),
                    (1.0, self.representative_variable),
                ],
                rhs=-c_var.lb,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"indicator_{self.name}_4",
                con_type=">=",
                variables=[
                    (-1.0, c_var),
                    (-c_var.ub, b_var),
                    (1.0, self.representative_variable),
                ],
                rhs=-c_var.ub,
            )
        )
