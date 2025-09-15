# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import itertools
import bisect

from alpaca.model_data import variable as var
from alpaca.expressions import expression as exn

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


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

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        name: str,
        model_data: ModelData,
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
        super().__init__(name, model_data, level, representative_variable)
        self.variables = variables
        self.model_data = model_data
        self.piecewise_constant_relation = {}
        for variable in self.variables:
            variable.add_nonlinearity_to_occurring_in(f"multilinear{len(variables)}")

    def extract_mpip_relation(self, approximation: bool = False):
        """Extract the piecewise constant relation for the multilinear expression."""
        if not self.representative_variable.is_discretized:
            return
        all_variables_mid_values_with_indices = (
            self.get_all_variables_mid_values_with_indices()
        )

        for combination_with_indices in itertools.product(
            *all_variables_mid_values_with_indices
        ):
            current_variable_indices = [index for _, index in combination_with_indices]
            implied_indices = self.calculate_implied_indices(
                combination_with_indices, current_variable_indices, approximation
            )
            self.piecewise_constant_relation[tuple(current_variable_indices)] = (
                implied_indices
            )

    def calculate_implied_indices(
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
        implied_value_lb, implied_value_ub = self.get_implied_lb_and_ub(lb_ub_list)
        implied_index_lb = self._get_implied_index_from_implied_value(implied_value_lb)
        implied_index_ub = self._get_implied_index_from_implied_value(implied_value_ub)
        return tuple(range(implied_index_lb, implied_index_ub + 1))

    def get_all_variables_mid_values_with_indices(
        self,
    ) -> list[list[tuple[float, int]]]:
        """Get mid values and their indices for all variables."""
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
            max(
                0,
                bisect.bisect_left(
                    self.representative_variable.breakpoints, implied_value
                )
                - 1,
            ),
            len(self.representative_variable.breakpoints) - 2,
        )

    def reformulate_to_bilinear_expressions(self):
        """Reformulate multilinear expression to bilinear expressions."""
        if len(self.variables) > 3:
            sub_bi_multilinear = MultilinearExpression(
                f"mb_{self.name}_sub",
                self.model_data,
                self.variables[1:],
                self.level + 1,
            )
            sub_bi_multilinear.reformulate_to_bilinear_expressions()
        else:
            sub_bi_multilinear = self.model_data.add_bilinear_expression(
                f"mb_{self.name}_sub",
                [self.variables[1], self.variables[2]],
                self.level + 1,
            )
        self.model_data.add_bilinear_expression(
            f"mb_{self.name}",
            [self.variables[0], sub_bi_multilinear.representative_variable],
            self.level,
            representative_variable=self.representative_variable,
        )

    @staticmethod
    def get_implied_lb_and_ub(
        lb_ub_list: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """Calculate the implied lower and upper bounds from a list of (lb, ub) tuples."""
        lb, ub = lb_ub_list[0]
        for next_lb, next_ub in lb_ub_list[1:]:
            candidates = [lb * next_lb, lb * next_ub, ub * next_lb, ub * next_ub]
            lb, ub = min(candidates), max(candidates)
        return lb, ub
