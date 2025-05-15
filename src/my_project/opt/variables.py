# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""

from my_project.read.process_data import Data
from my_project.settings import UserSettings
from my_project.utils.logger import logger


class Variables:  # pylint: disable=too-few-public-methods
    """Variables of optimization model."""

    def __init__(
        self,
        data: Data,
        opt_model,
        user_settings: UserSettings,
    ):
        self.opt_model = opt_model
        self.data = data
        self.user_settings = user_settings

    def add_variables_to_model(self):
        """
        Add variables to optimization model.
        """
        logger.info("""\tAdd variables""")
        self._add_node_variables()

    def _add_node_variables(self):
        pass
