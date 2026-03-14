# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import logging
from logging.handlers import RotatingFileHandler
import json
import pathlib

from alpaca.utils.logger import logger
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf


def config_console_logger(log_level):
    """
    function that configures a logger:
    ->for the console
    """
    logging.basicConfig(
        level=log_level,  # settings.LOG_LEVEL_CONSOLE,
        format="%(levelname)-8s:   %(message)s",
    )


def get_log_file_path() -> str | None:
    """Return the file path of the log file used by the logger,
    or None if no file handler is found."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None


def config_file_logger(path: str, level: str = "INFO"):
    """
    function that configures a logger:
    ->for a log-file
    """
    log_file_path = pathlib.Path(path)
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

    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_file_path,
        maxBytes=5 * 1024 * 1024,  # store up 5 MB per file
        backupCount=5,  # keep up to 5 files
    )

    handler.setLevel(level)
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


def read_config_file(path: str) -> dict:
    """Read config data file to dictionary."""
    with open(
        path,
        lsf.file_mode_read(),
        encoding=lsf.file_encoding_utf8(),
    ) as file:
        return json.load(file)


def write_file(path: str, content: str) -> None:
    """Writes content to a file.

    Args:
        path: Output file path.
        content: Content to write.
    """
    with open(path, lsf.file_mode_write(), encoding=lsf.file_encoding_utf8()) as file:
        file.write(content)


def build_latex_table(
    col_spec: str,
    header: str,
    rows: list[str],
    caption: str,
    label: str,
) -> str:
    """Builds a complete LaTeX table environment.

    Args:
        col_spec: Column specification (e.g., "lrrr").
        header: Header row(s) content.
        rows: List of data row strings.
        caption: Table caption.
        label: Table label for referencing.

    Returns:
        Complete LaTeX table as a string.
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    lines.extend(rows)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)
