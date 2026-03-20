# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import os
import sys
import time

import alpaca as alp

IMPORT_PATH = "../../data/import/"
LOGGING_PATH = "../../data/export/logs/"
CONFIG_PATH = IMPORT_PATH + "/example_configs/"
CONFIG_FILE = CONFIG_PATH + "locatelli_gurobi.json"
INSTANCES_PATH = IMPORT_PATH + "instances/"
INSTANCE_FILE = INSTANCES_PATH + "pooling_adhya1pq.osil"


def run_optimization(
    instance_file=INSTANCE_FILE, logging_path=LOGGING_PATH, config_file=CONFIG_FILE
) -> alp.Alpaca:
    """
    Function that executes the whole optimization process.
    """
    alpaca = alp.read_model_from_osil(instance_file)
    alpaca.configure_logging(f"{logging_path}{time.strftime('%Y%m%d-%H%M%S')}.log")
    alpaca.customize_settings(config_file)
    alpaca.build_pwl_relaxation_solver()
    alpaca.solve()
    return alpaca


if __name__ == "__main__":
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        run_optimization()
