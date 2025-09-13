# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools

import alpaca.pwl.pwl_method as pwm
from alpaca.model_data import constraint as con, variable as var
from alpaca.expressions import one_dim_expression as ode, multilinear_expression as mle


class MultipleChoiceMethod(pwm.PWLMethod):
    """Represents the multiple choice method for pwl."""

    def _represent_domain(self):
        """Represent the domain of a variable using the multiple choice method."""
        self._add_binary_pwl()
        self._add_continuous_pwl()

    def _add_binary_pwl(self) -> None:
        for breakpoint_index in range(len(self.variable.breakpoints) - 1):
            self.pwl_variables_binary.append(
                var.Variable(
                    f"{self.variable.name}_bp_{breakpoint_index}", var_type="B"
                )
            )

    def _add_continuous_pwl(self) -> None:
        """Link continuous variables to the breakpoints."""
        self._add_continuous_variables_multiple_choice()
        self._add_pwl_constraints_multiple_choice()

    def _add_continuous_variables_multiple_choice(self) -> None:
        for breakpoint_index in range(len(self.variable.breakpoints) - 1):
            self.pwl_variables_continuous.append(
                var.Variable(
                    f"{self.variable.name}_c_{breakpoint_index}",
                    var_type="C",
                    lb=min(0.0, self.variable.lb),
                    ub=max(0.0, self.variable.ub),
                )
            )

    def _add_pwl_constraints_multiple_choice(self) -> None:
        self.pwl_constraints.append(
            con.Constraint(
                f"mc_varlink_cont_{self.variable.name}",
                con_type="==",
                variables=[
                    (1.0, variable) for variable in self.pwl_variables_continuous
                ]
                + [(-1.0, self.variable)],
            )
        )
        self.pwl_constraints.append(
            con.Constraint(
                f"mc_varlink_{self.variable.name}",
                con_type="==",
                variables=[(1.0, variable) for variable in self.pwl_variables_binary],
                rhs=1.0,
            )
        )
        for breakpoint_index in range(len(self.variable.breakpoints) - 1):
            self.pwl_constraints.append(
                con.Constraint(
                    f"mc_lb_{self.variable.name}_{breakpoint_index}",
                    con_type="<=",
                    variables=[
                        (
                            self.variable.breakpoints[breakpoint_index],
                            self.pwl_variables_binary[breakpoint_index],
                        ),
                        (-1.0, self.pwl_variables_continuous[breakpoint_index]),
                    ],
                )
            )
            self.pwl_constraints.append(
                con.Constraint(
                    f"mc_ub_{self.variable.name}_{breakpoint_index}",
                    con_type=">=",
                    variables=[
                        (
                            self.variable.breakpoints[breakpoint_index + 1],
                            self.pwl_variables_binary[breakpoint_index],
                        ),
                        (-1.0, self.pwl_variables_continuous[breakpoint_index]),
                    ],
                )
            )

    def couple_domain_to_function_value_one_dim(
        self, expression: ode.OneDimExpression, approximation=False
    ) -> list[con.Constraint]:
        """Apply piecewise linear relaxation to the expression.
        If approximation is True, the approximation error term is set to 0
        """
        if approximation:
            constraints = self._apply_multiple_choice_method_approximation(expression)
        else:
            constraints = self._apply_multiple_choice_method_relaxation(expression)
        return constraints

    def _apply_multiple_choice_method_approximation(
        self, expression: ode.OneDimExpression
    ):
        variables_in_constraint = [(-1.0, expression.representative_variable)]
        for i, bp in enumerate(self.variable.breakpoints[:-1]):
            slope, intercept = (
                expression.get_linear_approximation_function_parameters_for_segment(
                    bp, self.variable.breakpoints[i + 1]
                )
            )
            variables_in_constraint.append((slope, self.pwl_variables_continuous[i]))
            variables_in_constraint.append((intercept, self.pwl_variables_binary[i]))
        return [
            con.Constraint(
                f"mc_{expression.name}",
                con_type="==",
                variables=variables_in_constraint,
            )
        ]

    def _apply_multiple_choice_method_relaxation(
        self, expression: ode.OneDimExpression
    ):
        continuous_variables_in_constraint = [(-1.0, expression.representative_variable)]
        binary_variables_in_underestimating_constraint = []
        binary_variables_in_overestimating_constraint = []
        for i, bp in enumerate(self.variable.breakpoints[:-1]):
            slope, intercept = (
                expression.get_linear_approximation_function_parameters_for_segment(
                    bp, self.variable.breakpoints[i + 1]
                )
            )
            min_deviation, max_deviation = expression.get_min_max_deviation(
                bp, self.variable.breakpoints[i + 1], slope, intercept
            )
            continuous_variables_in_constraint.append(
                (slope, self.pwl_variables_continuous[i])
            )
            binary_variables_in_underestimating_constraint.append(
                (intercept + min_deviation, self.pwl_variables_binary[i])
            )
            binary_variables_in_overestimating_constraint.append(
                (intercept + max_deviation, self.pwl_variables_binary[i])
            )
        return [
            con.Constraint(
                f"mc_under_{expression.name}",
                con_type="<=",
                variables=continuous_variables_in_constraint
                + binary_variables_in_underestimating_constraint,
            ),
            con.Constraint(
                f"mc_over_{expression.name}",
                con_type=">=",
                variables=continuous_variables_in_constraint
                + binary_variables_in_overestimating_constraint,
            ),
        ]

    def apply_pwc_relaxation(
        self, expression, approximation: bool = False
    ):
        """
        Apply piecewise constant relaxation for multilinear expressions.

        This function iterates through all combinations of variable intervals
        and applies constraints based on either an approximation or a strict
        lower/upper bound calculation.
        """
        all_variables_mid_values_with_indices = (
            expression.get_all_variables_mid_values_with_indices()
        )
        constraints = []
        for combination_with_indices in itertools.product(
            *all_variables_mid_values_with_indices
        ):
            current_variable_indices = [index for _, index in combination_with_indices]
            implied_indices = expression.calculate_implied_indices(
                combination_with_indices, current_variable_indices, approximation
            )
            constraints.append(
                self._add_piecewise_constraint(
                    current_variable_indices, implied_indices, expression
                )
            )
        return constraints

    def _add_piecewise_constraint(
        self,
        current_variable_indices: list[int],
        implied_indices: tuple[int, ...],
        expression: mle.MultilinearExpression,
    ):
        """
        Builds and adds a single piecewise relaxation constraint to the model.

        Args:
            current_variable_indices: The list of indices for the active variable intervals.
            implied_indices: The resulting indices for the representative variable.
        """
        constraint_name = (
            f"mc_{expression.name}_{'_'.join(map(str, current_variable_indices))}"
        )

        # Define the initial part of the constraint involving the representative variable
        constraint_variables = [
            (
                -1.0,
                self.pwl_variables_binary[implied_index],
            )
            for implied_index in implied_indices
        ] + [
            (
                1.0,
                expression.variables[var_idx].pwl.pwl_variables_binary[
                    index_in_combination
                ],
            )
            for var_idx, index_in_combination in enumerate(current_variable_indices)
        ]

        constraint = con.Constraint(
            name=constraint_name,
            con_type="<=",
            variables=constraint_variables,
            rhs=len(expression.variables) - 1.0,
        )
        return constraint
