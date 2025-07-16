# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import bisect

from alpaca.model_data import variable as var, constraint as con
import alpaca.expressions.expression as exn


class MultilinearExpression(exn.Expression):
    """Represents a multilinear expression with multiple variables.

    A multilinear expression is a product of multiple variables where each variable appears
    with degree at most 1. The expression is represented by a set of variables and a
    representative variable that holds the result of the multilinear operation.

    Attributes:
        name: Identifier for the expression
        variables: List of variables involved in the multilinear expression
        level: Level of expression in expression tree
        representative_variable: Variable representing the result of the expression
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variables: list[var.Variable],
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Initialize multilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            variables: List of variables involved in the multilinear expression
            level: Level of expression in expression tree
            representative_variable: Optional existing variable to represent result
        """
        super().__init__(name, model_data, level, representative_variable)
        self.variables = variables
        self.model_data = model_data
        self.representative_variable.add_nonlinearity_to_occurring_in("multilinear")
        self.piecewise_constant_relation = {}
        for variable in self.variables:
            variable.add_nonlinearity_to_occurring_in("multilinear")

    def apply_piecewise_constant_approximation(self):
        """Apply piecewise constant approximation for multilinear expressions."""
        all_variables_mid_values_with_indices = []
        for variable in self.variables:
            mid_values_for_current_var = []
            for i, breakpoint_val in enumerate(variable.breakpoints[:-1]):
                mid_value = (variable.breakpoints[i + 1] + breakpoint_val) / 2
                mid_values_for_current_var.append((mid_value, i))
            all_variables_mid_values_with_indices.append(mid_values_for_current_var)

        for combination_with_indices in itertools.product(
            *all_variables_mid_values_with_indices
        ):
            implied_value = 1.0
            current_variable_indices = []

            for mid_value, index in combination_with_indices:
                implied_value *= mid_value
                current_variable_indices.append(index)
            # pylint: disable=duplicate-code
            implied_index = min(
                bisect.bisect_left(
                    self.representative_variable.breakpoints, implied_value
                )
                - 1,
                len(self.representative_variable.breakpoints) - 2,
            )
            self.piecewise_constant_relation[tuple(current_variable_indices)] = (
                implied_index,
            )
            constraint_variables = [
                (
                    -1.0,
                    self.representative_variable.pwl_variables_binary[implied_index],
                )
            ]

            for var_idx, index_in_combination in enumerate(current_variable_indices):
                constraint_variables.append(
                    (
                        1.0,
                        self.variables[var_idx].pwl_variables_binary[
                            index_in_combination
                        ],
                    )
                )
            self.model_data.constraints.update(
                {
                    f"mc_{self.name}_"
                    f"{'_'.join(map(str, current_variable_indices))}": con.Constraint(
                        f"mc_{self.name}_{'_'.join(map(str, current_variable_indices))}",
                        con_type="==",
                        variables=constraint_variables,
                        rhs=1.0,
                    )
                }
            )

    def propagate_variable_bounds(self):
        """Propagate variables bounds."""
        lb, ub = self.variables[0].lb, self.variables[0].ub
        for v in self.variables[1:]:
            candidates = [lb * v.lb, lb * v.ub, ub * v.lb, ub * v.ub]
            lb, ub = min(candidates), max(candidates)
        self.representative_variable.lb = max(self.representative_variable.lb, lb)
        self.representative_variable.ub = min(self.representative_variable.ub, ub)
