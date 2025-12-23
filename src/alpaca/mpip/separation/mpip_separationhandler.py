# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.mpip import mpip_handler as mph
import alpaca.external_solvers.solver_wrapper as sw
import alpaca.utils.decorators as dec
from alpaca.mpip.separation import mpip_separator as mps


class MPIPSeparationHandler:
    """Handler for multiple MPIP separation routines."""

    def __init__(
        self, mpip_handler: mph.MPIPHandler, opt_model: sw.SolverWrapper
    ) -> None:
        self.mpip_handler = mpip_handler
        self.settings = mpip_handler.model_data.settings
        self.iteration = 0
        self.opt_model = opt_model
        self.nr_added_cuts = 0
        self._add_separation_to_mpip_instances()

    def _add_separation_to_mpip_instances(self) -> None:
        """Add separation model to all mpip instances."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator = mps.MPIPSeparator(mpip, self.opt_model)
            mpip.separator.build_separation_model()

    @dec.check_pwl_method_for_mpip_feature
    def add_mc_cormick_constraints(self) -> None:
        """Add McCormick constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_mc_cormick_constraints()

    @dec.check_pwl_method_for_mpip_feature
    def add_bar_constraints(self) -> None:
        """Add bar constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_bar_constraints()

    @dec.check_pwl_method_for_mpip_feature
    def add_stripe_constraints(self) -> None:
        """Add stripe constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            mpip.separator.add_stripe_constraints()

    @dec.check_pwl_method_for_mpip_feature
    def add_corner_constraints(self) -> None:
        """Add corner constraints for all MPIPs."""
        for mpip in self.mpip_handler.mpip_dict.values():
            if len(mpip.implying_variables) == 2:
                mpip.separator.add_corner_constraints()

    def separate_solution(self) -> bool:
        """Perform separation for current solution."""
        self.iteration += 1
        separated = False
        for mpip in self.mpip_handler.mpip_dict.values():
            if self.iteration % self.settings.feature_mpip_reset_interval == 0:
                mpip.separator.reset_useless_counter()
            if mpip.separator.usefulness < self.settings.feature_mpip_useless_threshold:
                continue
            if mpip.separator.separate_solution():
                separated = True
        if separated:
            return True
        return False
