# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import math

from alpaca.model_data import model_data as mda
from alpaca.utils.logger import logger
import alpaca.external_solvers.solver_wrapper as sw
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class MIPModel:
    """Optimization model object."""

    def __init__(self, data: mda.ModelData, nonlinear=False):
        logger.info(lsf.info_init_mip_model_buildup())
        self.opt_model = sw.SolverWrapper(data.settings.external_solver)
        self.data = data
        self.settings = data.settings
        self.nonlinear = nonlinear
        self._build_optimization_model()

    def _build_optimization_model(self):
        """
        Buildup optimization model.
        """
        self._add_variables()
        self._add_constraints()
        if self.nonlinear:
            self._add_nonlinear_constraints()
        self._add_objective()
        self._set_parameters()

    def _add_variables(self):
        for variable in self.data.variables.values():
            variable.solver_variable = self.opt_model.add_variable(
                name=variable.name,
                vtype=variable.var_type,
                lb=variable.lb,
                ub=variable.ub,
            )

    def _add_constraints(self):
        for constraint in self.data.constraints.values():
            if constraint.con_type == lsf.constraint_eq():
                constraint.solver_constraint = self.opt_model.add_constraint(
                    sum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    == constraint.rhs,
                    name=constraint.name,
                )
            elif constraint.con_type == lsf.constraint_leq():
                constraint.solver_constraint = self.opt_model.add_constraint(
                    sum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    <= constraint.rhs,
                    name=constraint.name,
                )
            elif constraint.con_type == lsf.constraint_geq():
                constraint.solver_constraint = self.opt_model.add_constraint(
                    -sum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    <= -constraint.rhs,
                    name=constraint.name,
                )

    def _add_nonlinear_constraints(self):
        if not self.settings.reformulate_multilinear_to_bilinear:
            for expression in self.data.expressions.multilinear_expressions.values():
                expression.solver_constraint = self.opt_model.add_constraint(
                    math.prod(
                        [variable.solver_variable for variable in expression.variables]
                    )
                    == expression.representative_variable.solver_variable,
                    name=expression.name,
                )
        if self.settings.bilinear_handling == 3:
            for expression in self.data.expressions.bilinear_expressions.values():
                expression.solver_constraint = self.opt_model.add_constraint(
                    (
                        expression.variables[0].solver_variable
                        * expression.variables[1].solver_variable
                        == expression.representative_variable.solver_variable
                    ),
                    name=expression.name,
                )
        for expression in self.data.expressions.one_dim_expressions.values():
            expression.solver_constraint = self.opt_model.add_nonlinear_constraint(
                expression.representative_variable.solver_variable,
                self.opt_model.get_nonlinear_function(expression.nonlinearity_type())(
                    expression.variable.solver_variable
                ),
                expression.name,
            )

    def _add_objective(self):
        self.opt_model.set_objective(
            self.data.variables[lsf.objective_var()].solver_variable,
            sense=lsf.objective_sense_minimize(),
        )

    def _set_parameters(self):
        """Set parameters for the SCIP model."""
        self.opt_model.set_time_limit(self.settings.solver_time_limit)
        self.opt_model.set_thread_limit(self.settings.solver_thread_limit)
        self.opt_model.set_seed(self.settings.seed)
