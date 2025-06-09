# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import json

import alpaca.settings as s


def read_config_file() -> dict:
    """Read config data file to dictionary."""
    with open(s.StaticSettings.config_file_path, "r", encoding="utf-8") as file:
        return json.load(file)
