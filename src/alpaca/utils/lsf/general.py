# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
"""
@authors: kuen,
"""


class General:
    """Project, general, and file handling strings."""

    @classmethod
    def project_name(cls) -> str:
        return "alpaca"

    @classmethod
    def empty_string(cls) -> str:
        return ""

    @classmethod
    def path_separator(cls) -> str:
        return "/"

    @classmethod
    def json_file_suffix(cls) -> str:
        return ".json"

    # --- File Handling & Logging ---

    @classmethod
    def file_mode_read(cls) -> str:
        return "r"

    @classmethod
    def file_mode_write(cls) -> str:
        return "w"

    @classmethod
    def file_encoding_utf8(cls) -> str:
        return "utf-8"

    @classmethod
    def log_rotation_type_size(cls) -> str:
        return "size"

    @classmethod
    def log_rotation_type_time(cls) -> str:
        return "time"
