# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import model_data as mda
from alpaca.settings import UserSettings
from alpaca.utils.logger import logger
import alpaca.external_solvers.solver_wrapper as sw
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class MIPModel:
    """Optimization model object."""

    def __init__(self, data: mda.ModelData, settings: UserSettings, solver_name: str):
        logger.info(lsf.info_init_mip_model_buildup())
        self.opt_model = sw.SolverWrapper(solver_name)
        self.data = data
        self.settings = settings
        self._build_optimization_model()

    def _build_optimization_model(self):
        """
        Buildup optimization model.
        """
        self._add_variables()
        self._add_constraints()
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
