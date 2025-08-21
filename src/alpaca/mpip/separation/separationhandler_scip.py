# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import pyscipopt as scip
from pyscipopt import SCIP_RESULT

import alpaca.settings as s
from alpaca.mpip import mpiphandler as mph
from alpaca.mpip.separation import separator_scip as mps


class SeparationHandler(scip.Sepa):
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

    def separate_solution(self) -> dict:
        """Perform separation for current solution."""
        self.iteration += 1
        self.cut_pool = []

        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.separate_solution()
            if mpip.separator.cut.rhs:  # Non-zero rhs indicates valid cut
                self.cut_pool.append(mpip.separator.cut)
        if self.cut_pool:
            return self._add_cuts_to_model()
        return {"result": SCIP_RESULT.DIDNOTFIND}

    def _add_cuts_to_model(self) -> dict:
        """Add cuts meeting violation threshold to model."""
        max_violation = max(cut.violation for cut in self.cut_pool)
        min_violation = s.StaticSettings.max_violation_relation * max_violation
        self.nr_added_cuts = 0
        result = SCIP_RESULT.DIDNOTFIND
        for cut in self.cut_pool:
            if cut.violation >= min_violation:
                cut_to_separate = self.opt_model.createEmptyRowSepa(
                    self,
                    f"mpip{self.nr_added_cuts}_x{self.iteration}",
                    lhs=None,
                    rhs=cut.rhs,
                )
                for coeff, var in cut.lhs:
                    self.opt_model.addVarToRow(cut_to_separate, var, coeff)
                self.opt_model.cacheRowExtensions(cut_to_separate)
                self.nr_added_cuts += 1
                self.opt_model.flushRowExtensions(cut_to_separate)
                infeasible = self.opt_model.addCut(cut_to_separate, forcecut=True)
                if infeasible:
                    result = SCIP_RESULT.CUTOFF
                else:
                    result = SCIP_RESULT.SEPARATED
                self.opt_model.releaseRow(cut_to_separate)
        return {"result": result}

    def sepaexeclp(self):
        """Run callback event."""
        self.opt_model = self.model
        return self.separate_solution()
