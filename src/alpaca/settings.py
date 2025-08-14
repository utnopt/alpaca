# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import json
import logging
import os
import time

from alpaca.utils.logger import logger


class StaticSettings:
    """
    Class containing static settings.
    """

    project_name = "bip-pwl"
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
    infinity = 1e4
    feasibility_tolerance = 1e-3

    # ===== MPIP settings =====
    max_violation_relation = 1e-2
    min_cut_violation = 1e-2
    rounding_precision = 5


class UserSettings:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    Class containing user settings.
    """

    def __init__(self, config_dict):
        self.solver_time_limit = int(config_dict.get("solver_time_limit", 7200))
        self.osil_file_name = str(config_dict.get("osil_file_name", "st_e41"))
        self.number_of_breakpoints = int(config_dict.get("number_of_breakpoints", 5))
        self.pwl_method = str(config_dict.get("pwl_method", "multiple-choice"))
        self.approximation = str(config_dict.get("approximation", "False")) == "True"
        self.external_solver = str(config_dict.get("external_solver", "scip"))
        self.bound_propagation_rounds = int(
            config_dict.get("bound_propagation_rounds", 3)
        )
        self.export_path = (
            StaticSettings.base_path
            + f"/data/export/{time.strftime('%Y-%m-%d_%H-%M-%S')}_"
            f"Result_{StaticSettings.project_name}/"
        )
        self.feature_mpip_separation = int(
            config_dict.get("feature/mpip/separation", 0)
        )
        self.feature_mpip_mccormick = int(config_dict.get("feature/mpip/mccormick", 0))
        self.feature_mpip_stair = int(config_dict.get("feature/mpip/stair", 0))
        self.feature_mpip_stripe = int(config_dict.get("feature/mpip/stripe", 0))
        self.feature_mpip = (
            self.feature_mpip_separation
            or self.feature_mpip_mccormick
            or self.feature_mpip_stair
            or self.feature_mpip_stripe
        )

    def save_to_json(self):
        """
        Function that saves the self-object as a dict to json
        """
        logger.info("The settings are saved as JSON-format to the export folder")
        json_data = self.__dict__
        with open(self.export_path + "config.json", "w", encoding="utf8") as json_file:
            json.dump(json_data, json_file, indent=4)
