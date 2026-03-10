# -*- coding: utf-8 -*-

"""
Study runner module for executing single instance-config combinations.

This module provides functionality to run a single optimization instance
with a specific configuration and return the results. It supports early
abort for instances that have already failed with another configuration.
"""
import os
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any

import alpaca as alp


class RunStatus(Enum):
    """Enumeration of possible run statuses."""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class RunResult:
    """Container for the result of a single study run.

    Attributes:
        instance_name: Name of the instance (without path/extension).
        config_name: Name of the configuration (without path/extension).
        status: The status of the run (SUCCESS, ERROR, or SKIPPED).
        csv_row: The CSV-formatted result row, or None if failed/skipped.
        error_message: Error message if status is ERROR, else None.
    """

    instance_name: str
    config_name: str
    status: RunStatus
    csv_row: str | None = None
    error_message: str | None = None


def extract_name_from_path(path: str) -> str:
    """Extracts the base name from a file path without extension.

    Args:
        path: Full path to the file.

    Returns:
        The file name without path and extension.
    """
    return os.path.splitext(os.path.basename(path))[0]


# pylint: disable=too-many-locals
def run_single_combination(args: tuple[str, str, str | None, bool, Any, int]) -> RunResult:
    """Runs a single instance-config combination and returns the result.

    This function is designed to be called from a process pool. It checks
    if the instance has already failed before starting execution, and
    marks the instance as failed if an error occurs.

    Args:
        args: Tuple of (instance_path, config_path, log_dir, suppress_output,
              failed_instances_dict).
            - instance_path: Full path to the .osil instance file.
            - config_path: Full path to the .json configuration file.
            - log_dir: Directory for log files. If None, logging is disabled.
            - suppress_output: If True, suppresses stdout during execution.
            - failed_instances_dict: Shared dict tracking failed instances.

    Returns:
        RunResult containing the status and CSV row if successful.
    """
    instance_path, config_path, log_dir, suppress_output, failed_instances, nr_of_threads = args
    instance_name = extract_name_from_path(instance_path)
    config_name = extract_name_from_path(config_path)

    # Check if this instance has already failed with another config
    if failed_instances.get(instance_name, False):
        return RunResult(
            instance_name=instance_name,
            config_name=config_name,
            status=RunStatus.SKIPPED,
            error_message="Instance failed with another configuration",
        )

    original_stdout = sys.stdout
    devnull_file: Any = None

    try:
        if suppress_output:
            with open(os.devnull, "w", encoding="utf-8") as devnull_file:
                sys.stdout = devnull_file

        # Initialize and configure Alpaca
        alpaca = alp.read_model_from_osil(instance_path)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{instance_name}_{config_name}.log")
            alpaca.configure_logging(log_path)

        alpaca.customize_settings(config_path)
        alpaca.user_settings.solver_thread_limit = min(8, nr_of_threads)
        alpaca.build_pwl_relaxation_solver()
        alpaca.solve()

        csv_row = alpaca.statistics.result_row_print

        return RunResult(
            instance_name=instance_name,
            config_name=config_name,
            status=RunStatus.SUCCESS,
            csv_row=csv_row,
        )

    except Exception as exc:  # pylint: disable=broad-except
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        # Mark this instance as failed so other configs will be skipped
        failed_instances[instance_name] = True

        return RunResult(
            instance_name=instance_name,
            config_name=config_name,
            status=RunStatus.ERROR,
            error_message=error_msg,
        )

    finally:
        if suppress_output:
            sys.stdout = original_stdout
            if devnull_file is not None:
                devnull_file.close()
