# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import time

from alpaca.external_solvers import model_scip as msc, model_gurobi as mgu
import alpaca.mpip.separation.separationhandler_gurobi as seg
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
        self.gurobi_callback_function = None

    def solve_instance(self):
        """Solve instance."""
        logger.info("Solve instance..")
        self._attach_event_handlers()
        start_time = time.time()
        self.external_solver.opt_model.optimize(self.gurobi_callback_function)
        runtime = time.time() - start_time
        return runtime

    def _attach_event_handlers(self):
        if self.settings.external_solver == "scip":
            self._attach_event_handlers_scip()
        elif self.settings.external_solver == "gurobi":
            self._attach_event_handlers_gurobi()

    def _attach_event_handlers_scip(self):
        if self.mpip_separation_handler is not None:
            self.external_solver.opt_model.includeEventhdlr(
                self.mpip_separation_handler,
                "mpip_event_handler",
                "Event handler that separates mpip cuts",
            )

    def _attach_event_handlers_gurobi(self):
        if self.mpip_separation_handler is not None:
            # pylint: disable=protected-access
            self.external_solver.opt_model._separation_handler = (
                self.mpip_separation_handler
            )
            self.gurobi_callback_function = seg.separation_callback
