# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import time

from alpaca.external_solvers import mip_model as mm, solver_wrapper as sw
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class Solver:
    """Solver object."""

    def __init__(self, external_solver: mm.MIPModel):
        self.external_solver = external_solver
        self.settings = external_solver.settings
        self.mpip_separation_handler = None
        self.gurobi_callback_function = None

    def solve_instance(self):
        """Solve instance."""
        logger.info(lsf.info_init_solver())
        start_time = time.time()
        self._activate_mpip_features()
        self._attach_event_handlers()
        self.external_solver.opt_model.optimize(self.gurobi_callback_function)
        runtime = time.time() - start_time
        return runtime

    def compute_optimal_mip_start(self):
        """Compute an optimal MIP start using the external solver."""
        self.external_solver.opt_model.optimize(self.gurobi_callback_function)
        self.external_solver.save_solution_to_mip_start()
        self.external_solver.opt_model.reset_model()
        self.external_solver.set_mip_start()
        self._activate_mpip_features()
        self._attach_event_handlers()
        self.external_solver.opt_model.turn_off_heuristics()

    def _attach_event_handlers(self):
        if self.settings.external_solver == lsf.solver_name_scip():
            self._attach_event_handlers_scip()
        elif self.settings.external_solver == lsf.solver_name_gurobi():
            self._attach_event_handlers_gurobi()

    def _activate_mpip_features(self):
        if self.settings.feature_mpip_mccormick:
            if self.settings.pwl_method == lsf.pwl_method_none():
                logger.warning(lsf.warning_mpip_features_disabled_for_pwl_method_none())
                return
            self.mpip_separation_handler.add_mc_cormick_constraints()
        if self.settings.feature_mpip_stripe:
            if self.settings.pwl_method == lsf.pwl_method_none():
                logger.warning(lsf.warning_mpip_features_disabled_for_pwl_method_none())
                return
            self.mpip_separation_handler.add_stripe_constraints()

    def _attach_event_handlers_scip(self):
        if self.settings.feature_mpip_separation:
            if self.settings.pwl_method == lsf.pwl_method_none():
                logger.warning(lsf.warning_mpip_features_disabled_for_pwl_method_none())
                return
            scip_mpip_separation = sw.ScipSeparation(self.mpip_separation_handler)
            for mpip in self.mpip_separation_handler.mpip_handler.mpip_dict.values():
                mpip.separator.separation_handler = scip_mpip_separation
            self.external_solver.opt_model.model.includeSepa(
                scip_mpip_separation,
                lsf.scip_separator_name_mpip(),
                lsf.scip_separator_description_mpip(),
                priority=536870911,
                freq=1,
            )

    def _attach_event_handlers_gurobi(self):
        if self.settings.feature_mpip_separation:
            if self.settings.pwl_method == lsf.pwl_method_none():
                logger.warning(lsf.warning_mpip_features_disabled_for_pwl_method_none())
                return
            # pylint: disable=protected-access
            self.external_solver.opt_model.model._mpip_separation_handler = (
                self.mpip_separation_handler
            )
            self.gurobi_callback_function = sw.gurobi_separation_callback
