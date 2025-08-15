# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import bisect

from alpaca.model_data import variable as var, constraint as con
from alpaca.expressions import (
    expression as exn,
    linear_expression as lie,
    one_dim_expression as ode,
)


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
        self.model_data = model_data
        self.representative_variable.add_nonlinearity_to_occurring_in("bilinear")
        self.first_var.add_nonlinearity_to_occurring_in("bilinear")
        self.second_var.add_nonlinearity_to_occurring_in("bilinear")
        self.piecewise_constant_relation = {}

    def apply_piecewise_constant_approximation(self):
        """Apply piecewise constant approximation."""
        mid_values_first = [
            (self.first_var.breakpoints[i + 1] + breakpoint_first) / 2
            for i, breakpoint_first in enumerate(self.first_var.breakpoints[:-1])
        ]
        mid_values_second = [
            (self.second_var.breakpoints[i + 1] + breakpoint_second) / 2
            for i, breakpoint_second in enumerate(self.second_var.breakpoints[:-1])
        ]
        for i, mid_value_first in enumerate(mid_values_first):
            for j, mid_value_second in enumerate(mid_values_second):
                implied_value = mid_value_first * mid_value_second
                implied_index = min(
                    bisect.bisect_left(
                        self.representative_variable.breakpoints, implied_value
                    )
                    - 1,
                    len(self.representative_variable.breakpoints) - 2,
                )
                self.piecewise_constant_relation[(i, j)] = (implied_index,)
                self.model_data.add_constraint(
                    con.Constraint(
                        f"mc_{self.name}_{i}_{j}",
                        con_type="<=",
                        variables=[
                            (
                                -1.0,
                                self.representative_variable.pwl_variables_binary[
                                    implied_index
                                ],
                            ),
                            (1.0, self.first_var.pwl_variables_binary[i]),
                            (1.0, self.second_var.pwl_variables_binary[j]),
                        ],
                        rhs=1.0,
                    )
                )

    def reformulate_to_sum_of_squares(self):
        """Reformulate bilinear expression to sum of squares.
        xy = 0.5 (x² + y² − p²), p = x - y."""
        master_linear_expression = self.model_data.add_linear_expression(
            lie.LinearExpression(
                f"le_{self.name}_master",
                self.model_data,
                self.level,
                representative_variable=self.representative_variable,
            )
        )
        square_first_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression(
                f"fvs_{self.name}",
                self.model_data,
                self.first_var,
                self.level + 1,
            )
        )
        square_second_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression(
                f"svs_{self.name}",
                self.model_data,
                self.second_var,
                self.level + 1,
            )
        )
        sub_linear_expression = self.model_data.add_linear_expression(
            lie.LinearExpression(
                f"le_{self.name}_sub",
                self.model_data,
                self.level + 2,
            )
        )
        sub_linear_expression.variables = [
            (1.0, self.first_var),
            (-1.0, self.second_var),
        ]
        square_helper_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression(
                f"hvs_{self.name}",
                self.model_data,
                sub_linear_expression.representative_variable,
                self.level + 1,
            )
        )
        master_linear_expression.variables = [
            (0.5, square_first_var.representative_variable),
            (0.5, square_second_var.representative_variable),
            (-0.5, square_helper_var.representative_variable),
        ]

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
