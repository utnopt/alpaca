# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_scip import model_scip as msc
from alpaca.settings import UserSettings
from alpaca.utils.logger import logger


class Solver:
    """Solver object."""

    def __init__(self, model_scip: msc.ModelScip, settings: UserSettings):
        self.model_scip = model_scip
        self.settings = settings
        self.mpip_separation_handler = None

    def solve_instance(self):
        """Solve instance."""
        logger.info("Solve instance..")
        self._attach_event_handlers()
        self.model_scip.opt_model.optimize()

    def _attach_event_handlers(self):
        if self.mpip_separation_handler is not None:
            self.model_scip.opt_model.includeEventhdlr(
                self.mpip_separation_handler,
                "mpip_event_handler",
                "Event handler that separates mpip cuts",
            )
