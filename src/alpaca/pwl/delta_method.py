# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools

import alpaca.pwl.pwl_method as pwm
from alpaca.model_data import constraint as con, variable as var
from alpaca.expressions import one_dim_expression as ode, multilinear_expression as mle
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class DeltaMethod(pwm.PWLMethod):
    """Represents the delta method for pwl."""

    def _represent_domain(self):
        """Represent the domain of a variable using the delta method."""
        self._add_binary_pwl()
        self._add_continuous_pwl()

    def _add_binary_pwl(self) -> None:
        for breakpoint_index in range(len(self.variable.breakpoints) - 2):
            self.pwl_variables_binary.append(
                var.Variable(
                    lsf.var_name_pwl_delta_binary(self.variable.name, breakpoint_index),
                    var_type=lsf.var_type_binary(),
                )
            )

    def _add_continuous_pwl(self) -> None:
        """Link continuous variables to the breakpoints."""
        self._add_continuous_variables_delta()
        self._add_pwl_constraints_delta()

    def _add_continuous_variables_delta(self) -> None:
        for breakpoint_index in range(len(self.variable.breakpoints) - 1):
            self.pwl_variables_continuous.append(
                var.Variable(
                    lsf.var_name_pwl_delta_continuous(
                        self.variable.name, breakpoint_index
                    ),
                    var_type=lsf.var_type_continuous(),
                    lb=0.0,
                    ub=1.0,
                )
            )

    def _add_pwl_constraints_delta(self) -> None:
        self.pwl_constraints.append(
            con.LinearConstraint(
                lsf.con_name_pwl_delta_variable_link_continuous(self.variable.name),
                con_type=lsf.constraint_eq(),
                variables=[
                    (
                        self.variable.breakpoints[i + 1] - self.variable.breakpoints[i],
                        variable,
                    )
                    for i, variable in enumerate(self.pwl_variables_continuous)
                ]
                + [(-1.0, self.variable)],
                rhs=-self.variable.lb,
            )
        )
        for breakpoint_index in range(len(self.variable.breakpoints) - 2):
            self.pwl_constraints.append(
                con.LinearConstraint(
                    lsf.con_name_pwl_delta_interval_lb(
                        self.variable.name, breakpoint_index
                    ),
                    con_type=lsf.constraint_leq(),
                    variables=[
                        (
                            1.0,
                            self.pwl_variables_continuous[breakpoint_index + 1],
                        ),
                        (-1.0, self.pwl_variables_binary[breakpoint_index]),
                    ],
                )
            )
            self.pwl_constraints.append(
                con.LinearConstraint(
                    lsf.con_name_pwl_delta_interval_ub(
                        self.variable.name, breakpoint_index
                    ),
                    con_type=lsf.constraint_geq(),
                    variables=[
                        (
                            1.0,
                            self.pwl_variables_continuous[breakpoint_index],
                        ),
                        (-1.0, self.pwl_variables_binary[breakpoint_index]),
                    ],
                )
            )

    def couple_domain_to_function_value_one_dim(
        self, expression: ode.OneDimExpression, approximation=False
    ) -> list[con.LinearConstraint]:
        """Apply piecewise linear relaxation to the expression.
        If approximation is True, the approximation error term is set to 0
        """
        if approximation:
            constraints = self._apply_delta_method_approximation(expression)
        else:
            constraints = self._apply_delta_method_relaxation(expression)
        return constraints

    def _apply_delta_method_approximation(self, expression: ode.OneDimExpression):
        return [
            con.LinearConstraint(
                lsf.con_name_pwl_delta_approximation(expression.name),
                con_type=lsf.constraint_eq(),
                variables=[(-1.0, expression.representative_variable)]
                + [
                    (
                        expression.f(self.variable.breakpoints[i + 1])
                        - expression.f(self.variable.breakpoints[i]),
                        continuous_var,
                    )
                    for i, continuous_var in enumerate(self.pwl_variables_continuous)
                ],
                rhs=-expression.f(self.variable.breakpoints[0]),
            )
        ]

    def _apply_delta_method_relaxation(self, expression: ode.OneDimExpression):
        min_deviations = []
        max_deviations = []
        for i, bp in enumerate(self.variable.breakpoints[:-1]):
            slope, intercept = (
                expression.get_linear_approximation_function_parameters_for_segment(
                    bp, self.variable.breakpoints[i + 1]
                )
            )
            min_deviation, max_deviation = expression.get_min_max_deviation(
                bp, self.variable.breakpoints[i + 1], slope, intercept
            )
            min_deviations.append(min_deviation)
            max_deviations.append(max_deviation)
        if not min_deviations:
            pass
        return [
            con.LinearConstraint(
                lsf.con_name_pwl_delta_underestimation(expression.name),
                con_type=lsf.constraint_leq(),
                variables=[(-1.0, expression.representative_variable)]
                + [
                    (
                        expression.f(self.variable.breakpoints[i + 1])
                        - expression.f(self.variable.breakpoints[i]),
                        continuous_var,
                    )
                    for i, continuous_var in enumerate(self.pwl_variables_continuous)
                ]
                + [
                    (min_deviations[i + 1] - min_deviations[i], binary_var)
                    for i, binary_var in enumerate(self.pwl_variables_binary)
                ],
                rhs=-expression.f(self.variable.breakpoints[0]) - min_deviations[0],
            ),
            con.LinearConstraint(
                lsf.con_name_pwl_delta_overestimation(expression.name),
                con_type=lsf.constraint_geq(),
                variables=[(-1.0, expression.representative_variable)]
                + [
                    (
                        expression.f(self.variable.breakpoints[i + 1])
                        - expression.f(self.variable.breakpoints[i]),
                        continuous_var,
                    )
                    for i, continuous_var in enumerate(self.pwl_variables_continuous)
                ]
                + [
                    (max_deviations[i + 1] - max_deviations[i], binary_var)
                    for i, binary_var in enumerate(self.pwl_variables_binary)
                ],
                rhs=-expression.f(self.variable.breakpoints[0]) - max_deviations[0],
            ),
        ]

    def apply_pwc_relaxation(self, expression, approximation: bool = False):
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
                self._add_piecewise_constant_constraint(
                    current_variable_indices, implied_indices, expression
                )
            )
        return constraints

    def _add_piecewise_constant_constraint(
        self,
        implying_variable_indices: list[int],
        implied_indices: tuple[int, ...],
        expression: mle.MultilinearExpression,
    ):
        """
        Builds and adds a single piecewise constant relaxation constraint to the model.

        Args:
            implying_variable_indices: The list of indices for the active variable intervals.
            implied_indices: The resulting indices for the representative variable.
        """
        extended_pwl_variables_binary_implied = (
            [1.0] + self.pwl_variables_binary + [0.0]
        )
        extended_pwl_variables_binary_implying = {
            var_idx: [1.0]
            + expression.variables[var_idx].pwl.pwl_variables_binary
            + [0.0]
            for var_idx in range(len(implying_variable_indices))
        }
        constraint_variables = (
            [
                (
                    -1.0,
                    extended_pwl_variables_binary_implied[implied_index],
                )
                for implied_index in implied_indices
            ]
            + [
                (
                    1.0,
                    extended_pwl_variables_binary_implied[implied_index + 1],
                )
                for implied_index in implied_indices
            ]
            + [
                (
                    1.0,
                    extended_pwl_variables_binary_implying[var_idx][
                        index_in_combination
                    ],
                )
                for var_idx, index_in_combination in enumerate(
                    implying_variable_indices
                )
            ]
            + [
                (
                    -1.0,
                    extended_pwl_variables_binary_implying[var_idx][
                        index_in_combination + 1
                    ],
                )
                for var_idx, index_in_combination in enumerate(
                    implying_variable_indices
                )
            ]
        )
        rhs = (
            len(expression.variables)
            - 1.0
            - sum(
                coeff * variable
                for coeff, variable in constraint_variables
                if isinstance(variable, float)
            )
        )
        cleaned_constraint_variables = [
            (coeff, variable)
            for coeff, variable in constraint_variables
            if not isinstance(variable, float)
        ]
        constraint = con.LinearConstraint(
            name=lsf.con_name_pwc_multiple_choice_multilinear(
                expression.name, implying_variable_indices
            ),
            con_type=lsf.constraint_leq(),
            variables=cleaned_constraint_variables,
            rhs=rhs,
        )
        return constraint
