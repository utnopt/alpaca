# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import json
import dataclasses
import logging
import os
import time

from alpaca.utils.logger import logger


@dataclasses.dataclass
class StaticSettings:
    """
    Class containing static settings.
    """

    project_name = "bip-pwl"
    # ===== Paths to (static) input files =====
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = base_path + "/data/"
    import_path = data_path + "/import/"
    instances_path = import_path + "/instances/"
    config_file_path = import_path + "config.json"

    # ===== Logging settings ====
    log_console_level = logging.DEBUG
    log_file_level = logging.INFO
    log_rotation_type = "size"  # use "size", "time" or "none"


class UserSettings:  # pylint: disable=too-few-public-methods
    """
    Class containing user settings.
    """

    def __init__(self, config_dict):
        self.solver_time_limit = int(config_dict.get("solver_time_limit", 3600))
        self.osil_file_name = str(config_dict.get("osil_file_name", "st_e41"))

        self.export_path = (
            StaticSettings.base_path
            + f"/data/export/{time.strftime('%Y-%m-%d_%H-%M-%S')}_"
              f"Result_{StaticSettings.project_name}/"
        )

    def save_to_json(self):
        """
        function that saves the self-object as a dict to json
        """
        logger.info("\tThe settings are saved as JSON-format to the export folder")
        json_data = self.__dict__
        with open(self.export_path + "config.json", "w", encoding="utf8") as json_file:
            json.dump(json_data, json_file, indent=4)
