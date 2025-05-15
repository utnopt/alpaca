# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""

from my_project.opt.solution import Solution
from my_project.read.process_data import Data
from my_project.settings import UserSettings
from my_project.utils.logger import logger


class Report:  # pylint: disable=too-few-public-methods
    """
    Report object.
    """

    def __init__(self, opt_results: Solution, data: Data, settings: UserSettings):
        self.opt_results = opt_results
        self.data = data
        self.settings = settings

    def write_report(self):
        """
        Write report from solution.
        """
        logger.info("Writing report..")
        self._create_plots()

    def _create_plots(self):
        pass
