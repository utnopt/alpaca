# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""
import json

import my_project.settings as s


def read_config_file() -> dict:
    """Read config data file to dictionary."""
    with open(s.StaticSettings.config_file_path, "r", encoding="utf-8") as file:
        return json.load(file)
