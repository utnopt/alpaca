# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import json

import alpaca.settings as s
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


def read_config_file() -> dict:
    """Read config data file to dictionary."""
    with open(
        s.StaticSettings.config_file_path,
        lsf.file_mode_read(),
        encoding=lsf.file_encoding_utf8(),
    ) as file:
        return json.load(file)
