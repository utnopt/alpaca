# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""
from my_project.opt.constraints import Constraints
from my_project.opt.variables import Variables
from my_project.read.process_data import Data
from my_project.settings import UserSettings
from my_project.utils.logger import logger
import pyscipopt as opt_solver


# or
# import gurobipy as opt_solver # Env, Model, GRB, read, quicksum
class Model:
    """Optimization model object."""

    def __init__(self, data: Data, settings: UserSettings):
        self.opt_model = opt_solver.Model()
        self.data = data
        self.settings = settings
        self.variables = Variables(data, self.opt_model, settings)
        self.constraints = Constraints(data, self.opt_model, settings)

    def build_optimization_model(self):
        """
        Buildup optimization model.
        """
        logger.info("Creating model..")
        self._add_variables()
        self._add_constraints()

    def _add_variables(self):
        self.variables.add_variables_to_model()

    def _add_constraints(self):
        self.constraints.add_constraints_to_model()

    def set_parameters_and_optimize(self):
        """
        Optimize model.
        """
        logger.info("Solving model..")
