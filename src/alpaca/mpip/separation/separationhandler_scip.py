# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import pyscipopt as scip
from pyscipopt import SCIP_RESULT

from alpaca.mpip import mpiphandler as mph
from alpaca.mpip.separation import separator_scip as mps


class SeparationHandler(scip.Sepa):
    """Handler for multiple MPIP separation routines."""

    def __init__(
        self, mpip_handler: mph.MPIPHandler, opt_model: scip.Model | scip.Eventhdlr
    ) -> None:
        self.mpip_handler = mpip_handler
        self.iteration = 0
        self.opt_model = opt_model
        self.nr_added_cuts = 0
        self._add_separation_to_mpip_instances()
        scip.Eventhdlr.__init__(opt_model)

    def _add_separation_to_mpip_instances(self) -> None:
        """Add separation model to all mpip instances."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator = mps.Separator(self, mpip, self.opt_model)
            mpip.separator.build_separation_model()

    def add_mc_cormick_constraints(self) -> None:
        """Add McCormick constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_mc_cormick_constraints()

    def add_stripe_constraints(self) -> None:
        """Add stripe constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_stripe_constraints()

    def add_stair_constraints(self) -> None:
        """Add stair constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_stair_constraints()

    def separate_solution(self) -> dict:
        """Perform separation for current solution."""
        self.iteration += 1
        separated = False
        for mpip in self.mpip_handler.mpip_dict.values():
            if mpip.separator.separate_solution():
                separated = True
        if separated:
            return {"result": SCIP_RESULT.SEPARATED}
        return {"result": SCIP_RESULT.DIDNOTFIND}

    def sepaexeclp(self):
        """Run callback event."""
        self.opt_model = self.model
        return self.separate_solution()
