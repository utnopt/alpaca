# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import pyscipopt as scip

from alpaca.model_data import model_data as mda
from alpaca.settings import UserSettings
from alpaca.utils.logger import logger


class ModelScip:
    """Optimization model object."""

    def __init__(self, data: mda.ModelData, settings: UserSettings):
        self.opt_model = scip.Model()
        self.data = data
        self.settings = settings
        self._build_optimization_model()

    def add_mip_start(self):
        """Add MIP start to model."""
        start_solution = self.opt_model.createSol()
        for variable in self.data.variables.values():
            self.opt_model.setSolVal(
                start_solution, variable.solver_variable, variable.mip_start
            )
        self.opt_model.addSol(start_solution)

    def _build_optimization_model(self):
        """
        Buildup optimization model.
        """
        logger.info("Creating model..")
        self._add_variables()
        self._add_constraints()
        self._add_objective()
        self._set_parameters()

    def _add_variables(self):
        for variable in self.data.variables.values():
            variable.solver_variable = self.opt_model.addVar(
                name=variable.name,
                vtype=variable.var_type,
                lb=variable.lb,
                ub=variable.ub,
            )

    def _add_constraints(self):
        for constraint in self.data.constraints.values():
            if constraint.con_type == "==":
                constraint.solver_constraint = self.opt_model.addCons(
                    scip.quicksum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    == constraint.rhs,
                    name=constraint.name,
                )
            elif constraint.con_type == "<=":
                constraint.solver_constraint = self.opt_model.addCons(
                    scip.quicksum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    <= constraint.rhs,
                    name=constraint.name,
                )
            elif constraint.con_type == ">=":
                constraint.solver_constraint = self.opt_model.addCons(
                    -scip.quicksum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    <= -constraint.rhs,
                    name=constraint.name,
                )

    def _add_objective(self):
        self.opt_model.setObjective(
            self.data.variables["x_-1"].solver_variable,
            sense="minimize",
        )

    def _set_parameters(self):
        """Set parameters for the SCIP model."""
        self.opt_model.setRealParam("limits/time", self.settings.solver_time_limit)
        self.opt_model.setParam("parallel/maxnthreads", 4)
        self.opt_model.setParam("numerics/feastol", 1e-05)
        # all_params = self.opt_model.getParams()
        # for param in all_params:
        #     if param.startswith("heuristics/") and param.endswith("/freq"):
        #         self.opt_model.setParam(param, -1)
        # self.opt_model.setParam("separating/maxroundsroot", 10)
        # self.opt_model.hideOutput(True)
