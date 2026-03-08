# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import time

import alpaca.locatelli.stair_locatelli as slo
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

if TYPE_CHECKING:
    from alpaca.main import Alpaca


class Statistics:
    """Collects and manages statistics related to the optimization process."""

    def __init__(self, alp_instance: Alpaca) -> None:
        self.alp_instance = alp_instance
        self.build_start_time: float = 0.0
        self.build_time: float | None = None
        self.log_file_path: str | None = None
        self.solver_stats: dict | None = None
        self._added_mccormick_envelopes = False

    def get_solver_information_from_external_solver_log(self):
        """Extracts solver information from the external solver log."""
        if self.log_file_path is None:
            return
        opt_model = self.alp_instance.solver.external_solver.opt_model
        self.solver_stats = opt_model.get_solver_log_information(
            self.log_file_path
        )

    def build_started(self):
        """Marks the start of the model buildup process."""
        self.build_start_time = time.time()

    def build_finished(self):
        """Marks the end of the model buildup process and calculates the total time taken."""
        self.build_time = time.time() - self.build_start_time

    @property
    def final_nr_variables(self):
        """Retrieves the final number of variables from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_final_nr_vars(), None)

    @property
    def final_nr_constraints(self):
        """Retrieves the final number of constraints from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_final_nr_constraints(), None)

    @property
    def solution_value(self):
        """Retrieves the solution from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_solution_value(), None)

    @property
    def solving_time(self):
        """Retrieves the solving time from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_solving_time(), None)

    @property
    def mip_gap(self):
        """Retrieves the MIP gap from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_mip_gap(), None)

    @property
    def nr_nodes(self):
        """Retrieves the node count from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_nr_nodes(), None)

    @property
    def presolved_nr_variables(self):
        """Retrieves the number of presolved variables from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_presolved_nr_vars(), None)

    @property
    def presolved_nr_constraints(self):
        """Retrieves the number of presolved constraints from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_presolved_nr_constraints(), None)

    @property
    def presolved_nr_nonzeros(self):
        """Retrieves the number of presolved nonzeros from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_presolved_nr_nonzeros(), None)

    @property
    def root_solving_time(self):
        """Retrieves the root relaxation solving time from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_root_solving_time(), None)

    @property
    def root_solution_value(self):
        """Retrieves the root relaxation value from the external solver if available."""
        if self.solver_stats is None:
            return None
        return self.solver_stats.get(lsf.stats_root_solution_value(), None)

    @property
    def stair_locatelli_domain_volume_polytope(self) -> float | None:
        """Returns bilinear domain volume over polytope from the Stair-Locatelli handler."""
        if self.alp_instance.stair_locatelli is not None:
            return (
                self.alp_instance.stair_locatelli.calculate_mean_bilinear_domain_volume(
                    is_polytope=True
                )
            )
        volume = slo.StairLocatelli.calculate_mean_bilinear_domain_volume_box(
            self.alp_instance.model_data, self._added_mccormick_envelopes
        )
        self._added_mccormick_envelopes = True
        return volume

    @property
    def stair_locatelli_domain_volume_polygon(self) -> float | None:
        """Returns bilinear domain volume over polygon from the Stair-Locatelli handler."""
        if self.alp_instance.stair_locatelli is not None:
            return (
                self.alp_instance.stair_locatelli.calculate_mean_bilinear_domain_volume(
                    is_polytope=False
                )
            )
        volume = slo.StairLocatelli.calculate_mean_bilinear_domain_volume_box(
            self.alp_instance.model_data, self._added_mccormick_envelopes
        )
        self._added_mccormick_envelopes = True
        return volume
