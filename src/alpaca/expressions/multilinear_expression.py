# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import bisect

from alpaca.model_data import variable as var, constraint as con
from alpaca.expressions import expression as exn, bilinear_expression as ble


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

    def apply_piecewise_constant_relaxation(self, approximation: bool = False):
        """
        Apply piecewise constant relaxation for multilinear expressions.

        This function iterates through all combinations of variable intervals
        and applies constraints based on either an approximation or a strict
        lower/upper bound calculation.
        """
        all_variables_mid_values_with_indices = (
            self._get_all_variables_mid_values_with_indices()
        )

        for combination_with_indices in itertools.product(
            *all_variables_mid_values_with_indices
        ):
            current_variable_indices = [index for _, index in combination_with_indices]

            # Calculate the implied indices based on the approximation flag
            implied_indices = self._calculate_implied_indices(
                combination_with_indices, current_variable_indices, approximation
            )

            # Store the relationship
            self.piecewise_constant_relation[tuple(current_variable_indices)] = (
                implied_indices
            )

            # Create and add the corresponding constraint to the model
            self._add_piecewise_constraint(current_variable_indices, implied_indices)

    def _calculate_implied_indices(
        self,
        combination_with_indices: tuple[tuple[float, int], ...],
        current_variable_indices: list[int],
        approximation: bool,
    ) -> tuple[int, ...]:
        """
        Calculates the tuple of implied indices for the representative variable.

        Args:
            combination_with_indices: A tuple containing (mid_value, index) for each variable.
            current_variable_indices: A list of the current variable indices for the combination.
            approximation: A boolean flag to determine the calculation method.

        Returns:
            A tuple of integer indices for the representative variable.
        """
        if approximation:
            implied_value = 1.0
            for mid_value, _ in combination_with_indices:
                implied_value *= mid_value
            implied_index = self._get_implied_index_from_implied_value(implied_value)
            return (implied_index,)
        lb_ub_list = [
            (
                self.variables[i].breakpoints[index],
                self.variables[i].breakpoints[index + 1],
            )
            for i, index in enumerate(current_variable_indices)
        ]
        implied_value_lb, implied_value_ub = self._get_implied_lb_and_ub(lb_ub_list)
        implied_index_lb = self._get_implied_index_from_implied_value(
            implied_value_lb
        )
        implied_index_ub = self._get_implied_index_from_implied_value(
            implied_value_ub
        )
        return tuple(range(implied_index_lb, implied_index_ub + 1))

    def _add_piecewise_constraint(
        self, current_variable_indices: list[int], implied_indices: tuple[int, ...]
    ):
        """
        Builds and adds a single piecewise relaxation constraint to the model.

        Args:
            current_variable_indices: The list of indices for the active variable intervals.
            implied_indices: The resulting indices for the representative variable.
        """
        # Assuming 'con' is available in the class scope
        constraint_name = (
            f"mc_{self.name}_{'_'.join(map(str, current_variable_indices))}"
        )

        # Define the initial part of the constraint involving the representative variable
        constraint_variables = [
            (
                -1.0,
                self.representative_variable.pwl_variables_binary[implied_index],
            )
            for implied_index in implied_indices
        ] + [
            (
                1.0,
                self.variables[var_idx].pwl_variables_binary[index_in_combination],
            )
            for var_idx, index_in_combination in enumerate(current_variable_indices)
        ]

        constraint = self.model_data.add_constraint(
            con.Constraint(
                name=constraint_name,
                con_type="<=",
                variables=constraint_variables,
                rhs=len(self.variables) - 1.0,
            )
        )

        # Add the original variables to the constraint
        for var_idx, index_in_combination in enumerate(current_variable_indices):
            constraint.variables.append(
                (
                    1.0,
                    self.variables[var_idx].pwl_variables_binary[index_in_combination],
                )
            )

    def _get_all_variables_mid_values_with_indices(
        self,
    ) -> list[list[tuple[float, int]]]:
        all_variables_mid_values_with_indices = []
        for variable in self.variables:
            mid_values_for_current_var = []
            for i, breakpoint_val in enumerate(variable.breakpoints[:-1]):
                mid_value = (variable.breakpoints[i + 1] + breakpoint_val) / 2
                mid_values_for_current_var.append((mid_value, i))
            all_variables_mid_values_with_indices.append(mid_values_for_current_var)
        return all_variables_mid_values_with_indices

    def _get_implied_index_from_implied_value(self, implied_value: float) -> int:
        return min(
            bisect.bisect_left(self.representative_variable.breakpoints, implied_value)
            - 1,
            len(self.representative_variable.breakpoints) - 2,
        )

    def reformulate_to_bilinear_expressions(self):
        """Reformulate multilinear expression to bilinear expressions."""
        if len(self.variables) > 3:
            sub_bi_multilinear = self.model_data.add_multilinear_expression(
                MultilinearExpression(
                    f"mb_{self.name}_sub",
                    self.model_data,
                    self.variables[1:],
                    self.level + 1,
                )
            )
            sub_bi_multilinear.reformulate_to_bilinear_expressions()
        else:
            sub_bi_multilinear = self.model_data.add_bilinear_expression(
                ble.BilinearExpression(
                    f"mb_{self.name}_sub",
                    self.model_data,
                    (self.variables[1], self.variables[2]),
                    self.level + 1,
                )
            )
        self.model_data.add_bilinear_expression(
            ble.BilinearExpression(
                f"mb_{self.name}",
                self.model_data,
                (self.variables[0], sub_bi_multilinear.representative_variable),
                self.level,
                representative_variable=self.representative_variable,
            )
        )

    def propagate_variable_bounds(self):
        """Propagate variables bounds."""
        lb_ub_list = [(variable.lb, variable.ub) for variable in self.variables]
        implied_lb, implied_ub = self._get_implied_lb_and_ub(lb_ub_list)
        self.representative_variable.lb = max(
            self.representative_variable.lb, implied_lb
        )
        self.representative_variable.ub = min(
            self.representative_variable.ub, implied_ub
        )

    @staticmethod
    def _get_implied_lb_and_ub(
        lb_ub_list: list[tuple[float, float]],
    ) -> tuple[float, float]:
        lb, ub = lb_ub_list[0]
        for next_lb, next_ub in lb_ub_list[1:]:
            candidates = [lb * next_lb, lb * next_ub, ub * next_lb, ub * next_ub]
            lb, ub = min(candidates), max(candidates)
        return lb, ub
