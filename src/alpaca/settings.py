# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import json
import logging
import os
import time

from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class StaticSettings:
    """
    Class containing static settings.
    """

    project_name = lsf.project_name()
    # ===== Paths to (static) input files =====
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = base_path + "/data/"
    import_path = data_path + "/import/"
    export_path = data_path + "/export/"
    instances_path = import_path + "/instances/"
    test_files_path = import_path + "/test_instances/"
    config_file_path = import_path + "config.json"

    # ===== Logging settings ====
    log_console_level = logging.DEBUG
    log_file_level = logging.INFO
    log_rotation_type = "size"  # use "size", "time" or "none"

    # ===== Data settings =====
    infinity = 1e6
    feasibility_tolerance = 1e-3

    # ===== MPIP settings =====
    max_violation_relation = 1e-2
    min_cut_violation = 1e-2
    rounding_precision = 5
    maximum_mpip_implication_factor = 3
    minimum_mpip_size = 2
    maximum_mpip_size = 5
    mpip_interval_lp_time_limit = 1
    nr_of_blocks_plotted = 5


class UserSettings:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    Class containing user settings.
    """

    def __init__(self, config_dict):
        self.seed = int(config_dict.get("seed", 42))
        self.solver_time_limit = int(config_dict.get("solver_time_limit", 7200))
        self.solver_thread_limit = int(config_dict.get("solver_thread_limit", 4))
        self.osil_file_name = str(config_dict.get("osil_file_name", "st_e41"))
        self.number_of_breakpoints = int(config_dict.get("number_of_breakpoints", 5))
        self.pwl_method = str(config_dict.get("pwl_method", "multiple_choice"))
        self.approximation = int(config_dict.get("approximation", 0))
        self.external_solver = str(config_dict.get("external_solver", "scip"))
        self.bound_propagation_rounds = int(
            config_dict.get("bound_propagation_rounds", 3)
        )
        self.bound_propagation_obbt_time_limit = int(
            config_dict.get("bound_propagation_obbt_time_limit", 300)
        )
        self.feature_stair_locatelli_obbt_time_limit = int(
            config_dict.get("feature/stair_locatelli/obbt_time_limit", 1800)
        )
        self.feature_stair_locatelli_evaluation_grid_size = int(
            config_dict.get("feature/stair_locatelli/evaluation_grid_size", 100)
        )
        self.export_path = (
            StaticSettings.base_path
            + f"/data/export/{time.strftime('%Y-%m-%d_%H-%M-%S')}_"
            f"Result_{StaticSettings.project_name}/"
        )
        self.reformulate_multilinear_to_bilinear = int(
            config_dict.get("reformulate_multilinear_to_bilinear", 1)
        )
        self.bilinear_handling = int(
            config_dict.get("bilinear_handling", 0)
        )  # 0: mccormick, 1: reformulate to sum of squares, 2: piecewise constant, 3: nonlinear
        self.bound_propagation = int(
            config_dict.get("bound_propagation", 0)
        )  # 0: manual, 1: obbt
        self.feature_mpip_separation = int(
            config_dict.get("feature/mpip/separation", 0)
        )
        self.feature_mpip_mccormick = int(config_dict.get("feature/mpip/mccormick", 0))
        self.feature_mpip_stair = int(config_dict.get("feature/mpip/stair", 0))
        self.feature_mpip_stripe = int(config_dict.get("feature/mpip/stripe", 0))
        self.feature_mpip_useless_threshold = float(
            config_dict.get("feature/mpip/useless_threshold", 0.1)
        )
        self.feature_mpip_frequency = int(config_dict.get("feature/mpip/frequency", 10))
        self.feature_mpip = (
            self.feature_mpip_separation
            or self.feature_mpip_mccormick
            or self.feature_mpip_stair
            or self.feature_mpip_stripe
        )
        self.breakpoint_generation = int(
            config_dict.get("breakpoint_generation", 1)
        )  # 0: equal, 1: neural networks (nnbp)
        self.feature_nnbp_learning_rate = float(
            config_dict.get("feature/nnbp/learning_rate", 1e-7)
        )
        self.feature_nnbp_nr_of_samples = int(
            config_dict.get("feature/nnbp/nr_of_samples", 1000)
        )
        self.feature_nnbp_time_limit = int(
            config_dict.get("feature/nnbp/time_limit", 100)
        )
        self.feature_nnbp_queue_size = int(
            config_dict.get("feature/nnbp/queue_size", 5)
        )
        self.feature_nnbp_convergence_tol = float(
            config_dict.get("feature/nnbp/convergence_tol", 1e-2)
        )
        self.feature_stair_locatelli = int(
            config_dict.get("feature/stair_locatelli", 0)
        )  # 0: disabled, 1: locatelli, 2: stair locatelli

        self.feature_stair_locatelli_grid_size = int(
            config_dict.get("feature/stair_locatelli/grid_size", 10)
        )  # grid size for stair locatelli

    def save_to_json(self):
        """
        Function that saves the self-object as a dict to json
        """
        logger.info(lsf.info_save_settings_json())
        json_data = self.__dict__
        with open(self.export_path + "config.json", "w", encoding="utf8") as json_file:
            json.dump(json_data, json_file, indent=4)
