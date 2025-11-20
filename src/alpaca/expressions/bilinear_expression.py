# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np

from alpaca.expressions import (
    multilinear_expression as mle,
    one_dim_expression as ode,
)
import alpaca.model_data.constraint as con
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


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
            lsf.linear_expression_bilinear_to_sum_of_squares(self.name),
            self.level,
            representative_variable=self.representative_variable,
        )

        # If the first variable is binary, x^2 = x.
        # Otherwise, create a new expression for the squared term.
        if first_var.var_type == lsf.var_type_binary():
            first_var_squared_rep = first_var
        else:
            square_first_var = self.model_data.add_one_dim_expression(
                ode.SquareExpression,
                lsf.expression_hash_square(first_var.name),
                first_var,
                self.level + 1,
            )
            first_var_squared_rep = square_first_var.representative_variable

        # If the second variable is binary, y^2 = y.
        # Otherwise, create a new expression for the squared term.
        if second_var.var_type == lsf.var_type_binary():
            second_var_squared_rep = second_var
        else:
            square_second_var = self.model_data.add_one_dim_expression(
                ode.SquareExpression,
                lsf.expression_hash_square(second_var.name),
                second_var,
                self.level + 1,
            )
            second_var_squared_rep = square_second_var.representative_variable

        # Create a helper variable p = x - y. This is always needed.
        sub_linear_expression = self.model_data.add_linear_expression(
            lsf.linear_expression_bilinear_to_sum_of_squares_helper(self.name),
            self.level + 2,
        )
        sub_linear_expression.variables = [
            (1.0, first_var),
            (-1.0, second_var),
        ]

        # The helper variable p is not necessarily binary, so we always square it.
        square_helper_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            lsf.expression_hash_square(
                sub_linear_expression.representative_variable.name
            ),
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
        self.mc_cormick_constraints["overestimator"].append(
            self.model_data.add_constraint(
                con.LinearConstraint(
                    lsf.con_name_mc_cormick_continuous_lb_ub(self.name),
                    con_type=lsf.constraint_leq(),
                    variables=[
                        (1.0, z),
                        (-x.lb, y),
                        (-y.ub, x),
                    ],
                    rhs=-x.lb * y.ub,
                )
            )
        )
        self.mc_cormick_constraints["overestimator"].append(
            self.model_data.add_constraint(
                con.LinearConstraint(
                    lsf.con_name_mc_cormick_continuous_ub_lb(self.name),
                    con_type=lsf.constraint_leq(),
                    variables=[
                        (1.0, z),
                        (-x.ub, y),
                        (-y.lb, x),
                    ],
                    rhs=-x.ub * y.lb,
                )
            )
        )
        self.mc_cormick_constraints["underestimator"].append(
            self.model_data.add_constraint(
                con.LinearConstraint(
                    lsf.con_name_mc_cormick_continuous_lb_lb(self.name),
                    con_type=lsf.constraint_geq(),
                    variables=[
                        (1.0, z),
                        (-x.lb, y),
                        (-y.lb, x),
                    ],
                    rhs=-x.lb * y.lb,
                )
            )
        )
        self.mc_cormick_constraints["underestimator"].append(
            self.model_data.add_constraint(
                con.LinearConstraint(
                    lsf.con_name_mc_cormick_continuous_ub_ub(self.name),
                    con_type=lsf.constraint_geq(),
                    variables=[
                        (1.0, z),
                        (-x.ub, y),
                        (-y.ub, x),
                    ],
                    rhs=-x.ub * y.ub,
                )
            )
        )

    def volume_and_max_diff_improvement(self):  #pylint: disable=too-many-locals
        """Calculate volume and max difference improvement of bilinear relaxation."""
        x = self.variables[0]
        y = self.variables[1]
        z = self.representative_variable
        max_z_interval = z.ub - z.lb
        mc_cormick_interval_sizes = []
        locatelli_interval_sizes = []

        for x_grid in np.linspace(
            x.lb,
            x.ub,
            self.model_data.settings.feature_stair_locatelli_evaluation_grid_size,
        ):
            for y_grid in np.linspace(
                y.lb,
                y.ub,
                self.model_data.settings.feature_stair_locatelli_evaluation_grid_size,
            ):
                upper_bounds_locatelli = [
                    -[
                        coefficient
                        for coefficient, variable in constraint.variables
                        if variable == x
                    ][0]
                    * x_grid
                    + -[
                        coefficient
                        for coefficient, variable in constraint.variables
                        if variable == y
                    ][0]
                    * y_grid
                    + constraint.rhs
                    for constraint in self.linear_relaxation_for_bilinear[
                        "overestimator"
                    ]
                ]
                upper_bound_mc_cormick = min(
                    (
                        -[
                            coefficient
                            for coefficient, variable in constraint.variables
                            if variable == x
                        ][0]
                        * x_grid
                        + -[
                            coefficient
                            for coefficient, variable in constraint.variables
                            if variable == y
                        ][0]
                        * y_grid
                        + constraint.rhs
                        for constraint in self.mc_cormick_constraints["overestimator"]
                    )
                )
                upper_bound_locatelli = (
                    min([upper_bound_mc_cormick] + upper_bounds_locatelli)
                    if upper_bounds_locatelli
                    else upper_bound_mc_cormick
                )

                lower_bounds_locatelli = [
                    -[
                        coefficient
                        for coefficient, variable in constraint.variables
                        if variable == x
                    ][0]
                    * x_grid
                    + -[
                        coefficient
                        for coefficient, variable in constraint.variables
                        if variable == y
                    ][0]
                    * y_grid
                    + constraint.rhs
                    for constraint in self.linear_relaxation_for_bilinear[
                        "underestimator"
                    ]
                ]
                lower_bound_mc_cormick = max(
                    (
                        -[
                            coefficient
                            for coefficient, variable in constraint.variables
                            if variable == x
                        ][0]
                        * x_grid
                        + -[
                            coefficient
                            for coefficient, variable in constraint.variables
                            if variable == y
                        ][0]
                        * y_grid
                        + constraint.rhs
                        for constraint in self.mc_cormick_constraints["underestimator"]
                    )
                )
                lower_bound_locatelli = (
                    max([lower_bound_mc_cormick] + lower_bounds_locatelli)
                    if lower_bounds_locatelli
                    else lower_bound_mc_cormick
                )
                mc_cormick_interval_sizes.append(
                    upper_bound_mc_cormick - lower_bound_mc_cormick
                )
                locatelli_interval_sizes.append(
                    max(0, upper_bound_locatelli - lower_bound_locatelli)
                )
        if sum(mc_cormick_interval_sizes) == 0:
            return 0, 0
        volume_improvement = (
            sum(mc_cormick_interval_sizes) - sum(locatelli_interval_sizes)
        ) / sum(mc_cormick_interval_sizes)
        max_diff_improvement = (
            max(
                mc_cormick_interval_size - locatelli_interval_size
                for mc_cormick_interval_size, locatelli_interval_size in zip(
                    mc_cormick_interval_sizes, locatelli_interval_sizes
                )
            )
            / max_z_interval
        )
        return volume_improvement, max_diff_improvement
