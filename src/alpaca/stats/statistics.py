# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import time

import alpaca.locatelli.stair_locatelli as slo
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf

if TYPE_CHECKING:
    from alpaca.main import Alpaca


class Statistics:  # pylint: disable=too-many-instance-attributes
    """Collects and manages statistics related to the optimization process."""

    def __init__(self, alp_instance: Alpaca) -> None:
        self.alp_instance = alp_instance
        self.instance_name: str = "unknown"
        self.config_name: str = "unknown"
        self.build_start_time: float = 0.0
        self.build_time: float | None = None
        self.log_file_path: str | None = None
        self.solver_stats: dict | None = None
        self.original_nr_variables: int = 0
        self.original_nr_constraints: int = 0
        self.original_nr_bilinear_expressions: int = 0
        self.original_nr_bilinear_binary_expressions: int = 0
        self.original_nr_mixed_binary_expressions: int = 0
        self.original_nr_multilinear_expressions: int = 0
        self.original_nr_one_dim_expressions: int = 0
        self.pwl_nr_variables: int = 0
        self.pwl_nr_constraints: int = 0
        self.pwl_nr_bilinear_expressions: int = 0
        self.pwl_nr_bilinear_binary_expressions: int = 0
        self.pwl_nr_mixed_binary_expressions: int = 0
        self.pwl_nr_multilinear_expressions: int = 0
        self.pwl_nr_one_dim_expressions: int = 0
        self.locatelli_domain_volume_polygon: int = 0
        self.locatelli_domain_volume_polytope: int = 0
        self.mpip_nr_instances: int = 0
        self.mpip_ratio: float = 0.0
        self.mpip_separation_nr_cuts: int = 0
        self._added_mccormick_envelopes = False

    def get_solver_information_from_external_solver_log(self):
        """Extracts solver information from the external solver log."""
        if self.log_file_path is None:
            return
        opt_model = self.alp_instance.solver.external_solver.opt_model
        self.solver_stats = opt_model.get_solver_log_information(self.log_file_path)

    def track_statistics_original_model(self):
        """Tracks statistics related to the original model before pwl transformations."""
        self.original_nr_variables = len(self.alp_instance.model_data.variables)
        self.original_nr_constraints = len(self.alp_instance.model_data.constraints)
        self.original_nr_bilinear_expressions = len(
            self.alp_instance.model_data.expressions.bilinear_expressions
        )
        self.original_nr_bilinear_binary_expressions = len(
            self.alp_instance.model_data.expressions.bilinear_binary_expressions
        )
        self.original_nr_mixed_binary_expressions = len(
            self.alp_instance.model_data.expressions.bilinear_binary_expressions
        )
        self.original_nr_multilinear_expressions = len(
            self.alp_instance.model_data.expressions.multilinear_expressions
        )
        self.original_nr_one_dim_expressions = len(
            self.alp_instance.model_data.expressions.one_dim_expressions
        )

    def track_statistics_pwl_model(self):
        """Tracks statistics related to the model after pwl transformations."""
        self.pwl_nr_variables = len(self.alp_instance.model_data.variables)
        self.pwl_nr_constraints = len(self.alp_instance.model_data.constraints)
        self.pwl_nr_bilinear_expressions = len(
            self.alp_instance.model_data.expressions.bilinear_expressions
        )
        self.pwl_nr_bilinear_binary_expressions = len(
            self.alp_instance.model_data.expressions.bilinear_binary_expressions
        )
        self.pwl_nr_mixed_binary_expressions = len(
            self.alp_instance.model_data.expressions.bilinear_binary_expressions
        )
        self.pwl_nr_multilinear_expressions = len(
            self.alp_instance.model_data.expressions.multilinear_expressions
        )
        self.pwl_nr_one_dim_expressions = len(
            self.alp_instance.model_data.expressions.one_dim_expressions
        )
        self._check_filters_pwl_model()

    def _check_filters_pwl_model(self):
        if (
            self.alp_instance.user_settings.filter_no_bilinear_expressions
            and self.pwl_nr_bilinear_expressions == 0
        ):
            raise ValueError(lsf.error_filter_no_bilinear_expressions())

    def track_statistics_stair_locatelli(self):
        """Tracks statistics related to the Stair-Locatelli handler."""
        self.locatelli_domain_volume_polytope = (
            self._stair_locatelli_domain_volume_polytope()
        )
        self.locatelli_domain_volume_polygon = (
            self._stair_locatelli_domain_volume_polygon()
        )

    def track_statistics_mpip(self):
        """Tracks statistics related to the MPIP handler."""
        mpip_handler = self.alp_instance.solver.mpip_separation_handler.mpip_handler
        self.mpip_nr_instances = len(mpip_handler.mpip_dict)
        self.mpip_ratio = round(
            sum(
                mpip.calculate_relation_ratio()
                for mpip in mpip_handler.mpip_dict.values()
            )
            / len(mpip_handler.mpip_dict),
            3,
        )
        self._check_filters_mpip()

    def _check_filters_mpip(self):
        if (
            self.alp_instance.user_settings.filter_no_mpip_instances
            and self.pwl_nr_bilinear_expressions == 0
        ):
            raise ValueError(lsf.error_filter_no_mpip_instances())

    def track_statistics_mpip_separation(self):
        """Tracks statistics related to the MPIP separation process."""
        mpip_handler = self.alp_instance.solver.mpip_separation_handler.mpip_handler
        self.mpip_separation_nr_cuts = sum(
            mpip.separator.nr_of_cuts for mpip in mpip_handler.mpip_dict.values()
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

    def _stair_locatelli_domain_volume_polytope(self) -> float | None:
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

    def _stair_locatelli_domain_volume_polygon(self) -> float | None:
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

    @property
    def result_row_print(self):
        """Formats the collected statistics into a CSV row for output."""
        return (
            f"{self.instance_name},"
            f"{self.config_name},"
            f"{self.solving_time},"
            f"{self.nr_nodes},"
            f"{self.solution_value},"
            f"{self.mip_gap},"
            f"{self.final_nr_variables},"
            f"{self.final_nr_constraints},"
            f"{self.presolved_nr_variables},"
            f"{self.presolved_nr_constraints},"
            f"{self.presolved_nr_nonzeros},"
            f"{self.root_solution_value},"
            f"{self.root_solving_time},"
            f"{self.build_time},"
            f"{self.original_nr_variables},"
            f"{self.original_nr_constraints},"
            f"{self.original_nr_bilinear_expressions},"
            f"{self.original_nr_bilinear_binary_expressions},"
            f"{self.original_nr_mixed_binary_expressions},"
            f"{self.original_nr_multilinear_expressions},"
            f"{self.original_nr_one_dim_expressions},"
            f"{self.pwl_nr_variables},"
            f"{self.pwl_nr_constraints},"
            f"{self.pwl_nr_bilinear_expressions},"
            f"{self.pwl_nr_bilinear_binary_expressions},"
            f"{self.pwl_nr_mixed_binary_expressions},"
            f"{self.pwl_nr_multilinear_expressions},"
            f"{self.pwl_nr_one_dim_expressions},"
            f"{self.locatelli_domain_volume_polygon},"
            f"{self.locatelli_domain_volume_polytope},"
            f"{self.mpip_nr_instances},"
            f"{self.mpip_ratio}"
        )
