# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import gurobipy as gp

import alpaca.settings as s
from alpaca.mpip import mpiphandler as mph
from alpaca.mpip.separation import separator_gurobi as msg


class SeparationHandler:
    """Handler for multiple MPIP separation routines."""

    def __init__(self, mpip_handler: mph.MPIPHandler, opt_model: gp.Model) -> None:
        self.mpip_handler = mpip_handler
        self.iteration = 0
        self.cut_pool: list[msg.MPIPCut] = []
        self.opt_model = opt_model
        self.nr_added_cuts = 0
        self._add_separation_to_mpip_instances()

    def _add_separation_to_mpip_instances(self) -> None:
        """Add separation model to all mpip instances."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator = msg.Separator(mpip, self.opt_model)
            mpip.separator.build_separation_model()

    def add_mc_cormick_constraints(self) -> None:
        """Add McCormick constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_mc_cormick_constraints()

    def add_stripe_constraints(self) -> None:
        """Add stripe constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_stripe_constraints()

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
                self.opt_model.cbCut(cut.lhs <= cut.rhs)
                self.nr_added_cuts += 1


def separation_callback(grb_model, where):
    """Callback for mpip separation"""
    if where == gp.GRB.Callback.MIPNODE:
        if grb_model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.Status.OPTIMAL:
            # pylint: disable=protected-access
            grb_model._separation_handler.separate_solution()
