# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""

from my_project.read.process_data import Data
from my_project.settings import UserSettings
from my_project.utils.logger import logger


class Solution:  # pylint: disable=too-few-public-methods
    """Variables of optimization model."""

    def __init__(
        self,
        data: Data,
        opt_model,
        user_settings: UserSettings,
    ):
        self.data = data
        self.user_settings = user_settings
        self.opt_model = opt_model
        self.solution = {}

    def create_solution_from_opt_model(self):
        """
        Build solution object from optimization model solution.
        """
        logger.info("Processing results..")
        self._build_solution_env()

    def _build_solution_env(self):
        self.solution = {}
