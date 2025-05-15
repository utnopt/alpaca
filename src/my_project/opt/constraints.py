# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""

from my_project.read.process_data import Data
from my_project.settings import UserSettings
from my_project.utils.logger import logger


class Constraints:  # pylint: disable=too-few-public-methods
    """Constraints of optimization model."""

    def __init__(
        self,
        data: Data,
        opt_model,
        user_settings: UserSettings,
    ):
        self.opt_model = opt_model
        self.data = data
        self.user_settings = user_settings

    def add_constraints_to_model(self):
        """
        Add constraints to optimization model.
        """
        logger.info("""\tAdd constraints""")
        self._add_flow_conservation_constraints()

    def _add_flow_conservation_constraints(self):
        pass
