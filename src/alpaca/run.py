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
LOGGING_PATH = "../../export/data/logs/"
CONFIG_FILE = IMPORT_PATH + "config.json"
INSTANCES_PATH = IMPORT_PATH + "instances/"
INSTANCE_FILE = INSTANCES_PATH + "alkyl.osil"


def run_optimization():
    """
    Function that executes the whole optimization process.
    """
    alpaca = alp.read_model_from_osil(INSTANCE_FILE)
    alpaca.configure_logging(f"{LOGGING_PATH}{time.strftime('%Y%m%d-%H%M%S')}.log")
    alpaca.customize_settings(CONFIG_FILE)
    alpaca.build_pwl_relaxation_solver()
    alpaca.solve()


if __name__ == "__main__":
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        run_optimization()
