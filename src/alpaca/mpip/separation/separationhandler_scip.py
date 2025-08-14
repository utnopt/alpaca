# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import pyscipopt as scip

import alpaca.settings as s
from alpaca.mpip import mpiphandler as mph
from alpaca.mpip.separation import separator_scip as mps


class SeparationHandler(scip.Eventhdlr):
    """Handler for multiple MPIP separation routines."""

    def __init__(
        self, mpip_handler: mph.MPIPHandler, opt_model: scip.Model | scip.Eventhdlr
    ) -> None:
        self.mpip_handler = mpip_handler
        self.iteration = 0
        self.cut_pool: list[mps.MPIPCut] = []
        self.opt_model = opt_model
        self.nr_added_cuts = 0
        self._add_separation_to_mpip_instances()
        scip.Eventhdlr.__init__(opt_model)

    def _add_separation_to_mpip_instances(self) -> None:
        """Add separation model to all mpip instances."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator = mps.Separator(mpip, self.opt_model)
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

    def separate_solution(self) -> int:
        """Perform separation for current solution."""
        self.iteration += 1
        self.cut_pool = []

        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.separate_solution()
            if mpip.separator.cut.rhs:  # Non-zero rhs indicates valid cut
                self.cut_pool.append(mpip.separator.cut)

        if self.cut_pool:
            self._add_cuts_to_model()

        return self.nr_added_cuts

    def _add_cuts_to_model(self) -> None:
        """Add cuts meeting violation threshold to model."""
        max_violation = max(cut.violation for cut in self.cut_pool)
        min_violation = s.StaticSettings.max_violation_relation * max_violation
        self.nr_added_cuts = 0

        for cut in self.cut_pool:
            if cut.violation >= min_violation:
                self.opt_model.addCons(cut.lhs <= cut.rhs)
                self.nr_added_cuts += 1

    def eventinit(self):
        """Catch callback event."""
        self.opt_model.catchEvent(scip.SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexit(self):
        """Stop callback event."""
        self.opt_model.dropEvent(scip.SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexec(self, _):
        """Run callback event."""
        self.separate_solution()
