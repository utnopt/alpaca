# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pyscipopt as opt_solver

from alpaca.model_data import model_data as mda
from alpaca.settings import UserSettings
from alpaca.utils.logger import logger


class ModelScip:
    """Optimization model object."""

    def __init__(self, data: mda.ModelData, settings: UserSettings):
        self.opt_model = opt_solver.Model()
        self.data = data
        self.settings = settings

    def build_optimization_model(self):
        """
        Buildup optimization model.
        """
        logger.info("Creating model..")
        self._add_variables()
        self._add_constraints()

    def _add_variables(self):
        pass

    def _add_constraints(self):
        pass

    def set_parameters_and_optimize(self):
        """
        Optimize model.
        """
        logger.info("Solving model..")
