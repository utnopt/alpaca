# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import os
import sys
import time

import alpaca as alp


def run_optimization():
    """
    Function that executes the whole optimization process.
    """
    alpaca = alp.read_model_from_osil("../../test/test_instances/alkyl.osil")
    alpaca.configure_logging(
        f"../../data/export/logs/{time.strftime('%Y%m%d-%H%M%S')}.log"
    )
    alpaca.customize_settings("../../data/import/config.json")
    alpaca.build_pwl_relaxation_solver()
    alpaca.solve()


if __name__ == "__main__":
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        run_optimization()
