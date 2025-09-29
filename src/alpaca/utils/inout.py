# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
import pathlib

from alpaca.settings import StaticSettings, UserSettings
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


def config_console_logger(log_level=logging.INFO):
    """
    function that configures a logger:
    ->for the console
    """
    logging.basicConfig(
        level=log_level,  # settings.LOG_LEVEL_CONSOLE,
        format="%(levelname)-8s:   %(message)s",
    )


def config_file_logger(
    settings: UserSettings,
    log_folder_name: str = None,  # overwrite for scenario based logging
    log_file_name: str = None,
):
    """
    function that configures a logger:
    ->for a log-file
    """
    if log_folder_name is None:
        log_folder_name = settings.export_path
    if log_file_name is None:
        log_file_name = StaticSettings.project_name + ".log"

    for hdlr in logger.handlers:
        # remove previous file handler (in case we run multiple scenario files)
        if issubclass(type(hdlr), logging.FileHandler):
            try:
                # If you want to keep base FileHandler to default log file name uncomment:
                # if (
                #    hdlr.baseFilename.split("\\")[-1].split("/")[-1]
                #    != StaticSettings.project_name + ".log"
                # ):
                logger.removeHandler(hdlr)
            except Exception as ex:  # pylint: disable=broad-exception-caught
                logger.warning(
                    lsf.warning_could_not_remove_log_file_handler(
                        str(hdlr.baseFilename), ex
                    )
                )

    # Create folder if it doesn't exists:
    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name, exist_ok=True)

    if StaticSettings.log_rotation_type.lower() == lsf.log_rotation_type_size():
        handler = RotatingFileHandler(
            log_folder_name + log_file_name,
            maxBytes=5 * 1024 * 1024,  # store up 5 MB per file
            backupCount=5,  # keep up to 5 files
        )
    elif StaticSettings.log_rotation_type.lower() == lsf.log_rotation_type_time():
        handler = TimedRotatingFileHandler(
            log_folder_name + log_file_name,
            when="midnight",  # you can also use 'W0' for rotating each Monday
            interval=1,
            backupCount=30,  # keep for 30 days
        )
    else:  # ordinary file handler
        handler = logging.FileHandler(log_folder_name + log_file_name, "a")

    handler.setLevel(StaticSettings.log_file_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)-8s:   %(message)s")
    )
    logger.addHandler(handler)

    # Now work around gurobi's double printing + logging bug
    # class DevNull:
    #     "Deactivation of direct console prints"
    #
    #     def write(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
    #         pass
    #
    #     def flush(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
    #         pass
    #
    # sys.stdout = DevNull()


def create_folder_if_not_exists(path: str):
    """Create folder if it does not exist."""
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    if path.exists() and not path.is_dir():
        raise FileExistsError(lsf.error_path_exists(path.as_posix()))
    path.mkdir(parents=True, exist_ok=True)
