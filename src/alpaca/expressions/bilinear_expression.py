# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.expressions import (
    multilinear_expression as mle,
    one_dim_expression as ode,
)
import alpaca.model_data.constraint as con


class BilinearExpression(mle.MultilinearExpression):
    """Represents a bilinear expression z = x * y.

    Attributes:
        name: Identifier for the expression
        variables: List of variables involved in the multilinear expression
        level: Level of expression in expression tree
        representative_variable: Variable representing the result of the expression
    """

    def reformulate_to_sum_of_squares(self):
        """Reformulate bilinear expression to sum of squares.
        xy = 0.5 * (x² + y² - p²), where p = x - y.
        If a variable is binary (B), its square is equal to the variable itself.
        """
        first_var = self.variables[0]
        second_var = self.variables[1]
        master_linear_expression = self.model_data.add_linear_expression(
            f"le_{self.name}_master",
            self.level,
            representative_variable=self.representative_variable,
        )

        # If the first variable is binary, x^2 = x.
        # Otherwise, create a new expression for the squared term.
        if first_var.var_type == "B":
            first_var_squared_rep = first_var
        else:
            square_first_var = self.model_data.add_one_dim_expression(
                ode.SquareExpression,
                f"fvs_{self.name}",
                first_var,
                self.level + 1,
            )
            first_var_squared_rep = square_first_var.representative_variable

        # If the second variable is binary, y^2 = y.
        # Otherwise, create a new expression for the squared term.
        if second_var.var_type == "B":
            second_var_squared_rep = second_var
        else:
            square_second_var = self.model_data.add_one_dim_expression(
                ode.SquareExpression,
                f"svs_{self.name}",
                second_var,
                self.level + 1,
            )
            second_var_squared_rep = square_second_var.representative_variable

        # Create a helper variable p = x - y. This is always needed.
        sub_linear_expression = self.model_data.add_linear_expression(
            f"le_{self.name}_sub",
            self.level + 2,
        )
        sub_linear_expression.variables = [
            (1.0, first_var),
            (-1.0, second_var),
        ]

        # The helper variable p is not necessarily binary, so we always square it.
        square_helper_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            f"hvs_{self.name}",
            sub_linear_expression.representative_variable,
            self.level + 1,
        )

        # The master expression becomes: z = 0.5 * (x^2_rep + y^2_rep - p^2)
        master_linear_expression.variables = [
            (0.5, first_var_squared_rep),
            (0.5, second_var_squared_rep),
            (-0.5, square_helper_var.representative_variable),
        ]

    def add_mccormick_envelope(self):
        """Add McCormick envelope constraints for bilinear expression."""
        x = self.variables[0]
        y = self.variables[1]
        z = self.representative_variable

        # McCormick envelope constraints
        self.model_data.add_constraint(
            con.Constraint(
                f"mcclu_{self.name}",
                con_type="<=",
                variables=[
                    (1.0, z),
                    (-x.lb, y),
                    (-y.ub, x),
                ],
                rhs=-x.lb * y.ub,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"mccul_{self.name}",
                con_type="<=",
                variables=[
                    (1.0, z),
                    (-x.ub, y),
                    (-y.lb, x),
                ],
                rhs=-x.ub * y.lb,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"mccll_{self.name}",
                con_type=">=",
                variables=[
                    (1.0, z),
                    (-x.lb, y),
                    (-y.lb, x),
                ],
                rhs=-x.lb * y.lb,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"mccuu_{self.name}",
                con_type=">=",
                variables=[
                    (1.0, z),
                    (-x.ub, y),
                    (-y.ub, x),
                ],
                rhs=-x.ub * y.ub,
            )
        )
