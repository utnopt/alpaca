# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import alpaca.model_data.constraint as con
from alpaca.expressions import (
    one_dim_expression as ode,
    multilinear_expression as mle,
    linear_expression as lie,
)

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class BoundPropagator:
    """Handles bound propagation for constraints and expressions."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data

    def propagate_linear_constraints(self):
        """Performs bound propagation on linear equality constraints."""
        for _ in range(self.model_data.settings.bound_propagation_rounds):
            for constraint in self.model_data.constraints.values():
                if constraint.con_type == "==":
                    self._propagate_bounds_equation(constraint)

    @staticmethod
    def _propagate_bounds_equation(constraint: con.Constraint):
        total_min = total_max = 0
        for a_j, x_j in constraint.variables:
            if a_j > 0:
                total_min += a_j * x_j.lb
                total_max += a_j * x_j.ub
            else:
                total_min += a_j * x_j.ub
                total_max += a_j * x_j.lb

        for a_i, x_i in constraint.variables:
            # Remove x_i's contribution to get sum of other variables
            if a_i > 0:
                other_min = total_min - a_i * x_i.lb
                other_max = total_max - a_i * x_i.ub
            else:
                other_min = total_min - a_i * x_i.ub
                other_max = total_max - a_i * x_i.lb

            # Update bounds for x_i
            if constraint.con_type in ("<=", "=="):
                if a_i > 0:
                    x_i.ub = min(x_i.ub, (constraint.rhs - other_min) / a_i)
                else:  # a_i < 0
                    x_i.lb = max(x_i.lb, (constraint.rhs - other_min) / a_i)
            if constraint.con_type in (">=", "=="):
                if a_i > 0:
                    x_i.lb = max(x_i.lb, (constraint.rhs - other_max) / a_i)
                else:  # a_i < 0
                    x_i.ub = min(x_i.ub, (constraint.rhs - other_max) / a_i)

    def propagate_expressions(self):
        """Performs bound propagation on all expressions for a set number of rounds."""
        sorted_expressions = sorted(
            self.model_data.expressions.all_low_dim_expressions(),
            key=lambda e: -e.level,
        )
        for _ in range(self.model_data.settings.bound_propagation_rounds):
            for expression in sorted_expressions:
                if isinstance(expression, ode.OneDimExpression):
                    self._propagate_bounds_one_dim_expression(expression)
                elif isinstance(expression, lie.LinearExpression):
                    self._propagate_bounds_linear_expressions(expression)
                else:
                    self._propagate_bounds_multilinear_expressions(expression)

    @staticmethod
    def _propagate_bounds_one_dim_expression(expression: ode.OneDimExpression):
        if isinstance(expression, (ode.SineExpression, ode.CosineExpression)):
            lb = -1.0
            ub = 1.0
        else:
            left = expression.f(expression.variable.lb)
            right = expression.f(expression.variable.ub)
            root = (
                expression.f(0)
                if expression.variable.lb <= 0 <= expression.variable.ub
                else left
            )
            lb, _, ub = sorted([left, right, root])
        expression.representative_variable.lb = max(
            lb, expression.representative_variable.lb
        )
        expression.representative_variable.ub = min(
            ub, expression.representative_variable.ub
        )

    @staticmethod
    def _propagate_bounds_multilinear_expressions(
        expression: mle.MultilinearExpression,
    ):
        lb_ub_list = [(variable.lb, variable.ub) for variable in expression.variables]
        implied_lb, implied_ub = expression.get_implied_lb_and_ub(lb_ub_list)
        expression.representative_variable.lb = max(
            expression.representative_variable.lb, implied_lb
        )
        expression.representative_variable.ub = min(
            expression.representative_variable.ub, implied_ub
        )

    @staticmethod
    def _propagate_bounds_linear_expressions(expression: lie.LinearExpression):
        lb = expression.constant + sum(
            min(coeff * variable.lb, coeff * variable.ub)
            for coeff, variable in expression.variables
        )
        ub = expression.constant + sum(
            max(coeff * variable.lb, coeff * variable.ub)
            for coeff, variable in expression.variables
        )
        expression.representative_variable.lb = max(
            lb, expression.representative_variable.lb
        )
        expression.representative_variable.ub = min(
            ub, expression.representative_variable.ub
        )
