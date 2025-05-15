# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""

from my_project.settings import UserSettings
from my_project.utils.logger import logger


class Data:  # pylint: disable=too-few-public-methods
    """Data container"""

    def __init__(self, settings: UserSettings):
        self.settings = settings

    def read_and_preprocess_data(self):
        """
        Create a data container object
        """
        logger.info("Reading data..")
        self._data_read()
        self._data_clean()

    def _data_read(self):
        pass

    def _data_clean(self):
        pass
