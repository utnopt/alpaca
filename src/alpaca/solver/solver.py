# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.external_solvers import model_scip as msc, model_gurobi as mgu
from alpaca.settings import UserSettings
from alpaca.utils.logger import logger


class Solver:
    """Solver object."""

    def __init__(
        self, external_solver: msc.ModelScip | mgu.ModelGurobi, settings: UserSettings
    ):
        self.external_solver = external_solver
        self.settings = settings
        self.mpip_separation_handler = None

    def solve_instance(self):
        """Solve instance."""
        logger.info("Solve instance..")
        self._attach_event_handlers()
        self.external_solver.opt_model.optimize()
        self.external_solver.opt_model.computeIIS()
        self.external_solver.opt_model.write("model.ilp")

    def _attach_event_handlers(self):
        if self.mpip_separation_handler is not None:
            self.external_solver.opt_model.includeEventhdlr(
                self.mpip_separation_handler,
                "mpip_event_handler",
                "Event handler that separates mpip cuts",
            )
