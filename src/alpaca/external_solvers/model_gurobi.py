# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import gurobipy as gp

from alpaca.model_data import model_data as mda
from alpaca.settings import UserSettings
from alpaca.utils.logger import logger


class ModelGurobi:
    """Optimization model object."""

    def __init__(self, data: mda.ModelData, settings: UserSettings):
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 1)
        env.start()
        self.opt_model = gp.Model(env=env)
        self.opt_model.setParam("OutputFlag", 1)
        self.data = data
        self.settings = settings
        self._build_optimization_model()

    def _build_optimization_model(self):
        """
        Buildup optimization model.
        """
        logger.info("Creating model..")
        self._add_variables()
        self._add_constraints()
        self._add_objective()
        self._set_parameters()

    def add_solution_to_mip_start(self):
        """Add current solution to MIP start."""
        self.opt_model.optimize()
        for variable in self.data.variables.values():
            variable.mip_start = variable.solver_variable.x

    def add_mip_start(self):
        """Add MIP start to model."""
        for variable in self.data.variables.values():
            variable.solver_variable.start = variable.mip_start

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
                constraint.solver_constraint = self.opt_model.addConstr(
                    gp.quicksum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    == constraint.rhs,
                    name=constraint.name,
                )
            elif constraint.con_type == "<=":
                constraint.solver_constraint = self.opt_model.addConstr(
                    gp.quicksum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    <= constraint.rhs,
                    name=constraint.name,
                )
            elif constraint.con_type == ">=":
                constraint.solver_constraint = self.opt_model.addConstr(
                    -gp.quicksum(
                        coeff * variable.solver_variable
                        for coeff, variable in constraint.variables
                    )
                    <= -constraint.rhs,
                    name=constraint.name,
                )

    def _add_objective(self):
        self.opt_model.setObjective(
            self.data.variables["x_-1"].solver_variable,
            sense=gp.GRB.MINIMIZE,
        )

    def _set_parameters(self):
        """Set Gurobi parameters based on user settings."""
        self.opt_model.setParam("TimeLimit", self.settings.solver_time_limit)
        self.opt_model.setParam("Threads", 4)
