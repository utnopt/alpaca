# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from alpaca.utils.logger import logger
import alpaca.model_data.constraint as con
from alpaca.expressions import (
    one_dim_expression as ode,
    multilinear_expression as mle,
    linear_expression as lie,
)
from alpaca.external_solvers import mip_model as mm, solver_wrapper as sw
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf

if TYPE_CHECKING:
    from alpaca.model_data import model_data as mda, variable as var


class BoundPropagator:
    """Handles bound propagation for constraints and expressions."""

    def __init__(self, model_data: mda.ModelData):
        self.model_data = model_data

    def propagate_bounds(self):
        """Performs bound propagation on constraints and expressions."""
        logger.info(lsf.info_propagate_bounds())
        for _ in range(self.model_data.settings.bound_propagation_rounds):
            self._propagate_linear_constraints()
            self._propagate_expressions()

    def apply_obbt(self):
        """Use optimization-based bound tightening (OBBT)."""
        logger.info(lsf.info_apply_obbt())
        external_solver = mm.MIPModel(self.model_data)
        external_solver.opt_model.hide_output()
        if self.model_data.settings.bound_propagation == 2:
            selected_variables = []
            for expression in self.model_data.expressions.bilinear_expressions.values():
                selected_variables.append(expression.variables[0].name)
                selected_variables.append(expression.variables[1].name)
            selected_variables = set(selected_variables)
        else:
            selected_variables = set(self.model_data.variables.keys())
        if len(selected_variables) == 0:
            return
        external_solver.opt_model.set_time_limit(
            self.model_data.settings.bound_propagation_obbt_time_limit
            / len(selected_variables)
        )
        for variable_name in selected_variables:
            variable = self.model_data.variables[variable_name]
            if (
                variable.var_type != lsf.var_type_binary()
                and variable.name != lsf.objective_var()
            ):
                self._tighten_bounds_via_obbt(variable, external_solver.opt_model)

    @staticmethod
    def _tighten_bounds_via_obbt(variable: var.Variable, opt_model: sw.SolverWrapper):
        # Minimize to find lower bound
        opt_model.set_objective(
            variable.solver_variable, sense=lsf.objective_sense_minimize()
        )
        opt_model.optimize()
        if opt_model.is_infeasible():
            return
        try:
            variable.lb = max(variable.lb, opt_model.get_objective_bound())
        except AttributeError:
            pass
        # Maximize to find upper bound
        opt_model.set_objective(
            variable.solver_variable, sense=lsf.objective_sense_maximize()
        )
        opt_model.optimize()
        if opt_model.is_infeasible():
            return
        try:
            variable.ub = min(variable.ub, opt_model.get_objective_bound())
        except AttributeError:
            pass

    def _propagate_linear_constraints(self):
        """Performs bound propagation on linear equality constraints."""
        for constraint in self.model_data.constraints.values():
            self._propagate_bounds_equation(constraint)

    @staticmethod
    def _propagate_bounds_equation(constraint: con.LinearConstraint):
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
            if constraint.con_type in (lsf.constraint_leq(), lsf.constraint_eq()):
                if a_i > 0:
                    x_i.ub = min(x_i.ub, (constraint.rhs - other_min) / a_i)
                else:  # a_i < 0
                    x_i.lb = max(x_i.lb, (constraint.rhs - other_min) / a_i)
            if constraint.con_type in (lsf.constraint_geq(), lsf.constraint_eq()):
                if a_i > 0:
                    x_i.lb = max(x_i.lb, (constraint.rhs - other_max) / a_i)
                else:  # a_i < 0
                    x_i.ub = min(x_i.ub, (constraint.rhs - other_max) / a_i)

    def _propagate_expressions(self):
        """Performs bound propagation on all expressions for a set number of rounds."""
        sorted_expressions = sorted(
            self.model_data.expressions.all_low_dim_expressions(),
            key=lambda e: -e.level,
        )
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
