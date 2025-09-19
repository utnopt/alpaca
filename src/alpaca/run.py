# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import os
import sys
import traceback

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.mpip.mpip_handler as mph
import alpaca.mpip.separation.mpip_separationhandler as msh
import alpaca.settings as s
from alpaca.utils import inout as ut_io, data_reading as ut_dr
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


def run_optimization():
    """
    Function that executes the whole optimization process.
    1) get the settings
    2) read the data
    """
    try:
        ut_io.config_console_logger()
        config_dict = ut_dr.read_config_file()
        user_settings = s.UserSettings(config_dict)

        ut_io.config_file_logger(user_settings)
        user_settings.save_to_json()

        model_data = mda.ModelData(user_settings)

        external_solver = mm.MIPModel(
            model_data, user_settings, user_settings.external_solver
        )

        solver = slv.Solver(external_solver, user_settings)

        if user_settings.feature_mpip:
            if user_settings.pwl_method == lsf.pwl_method_none():
                logger.warning(lsf.warning_mpip_features_disabled_for_pwl_method_none())
            else:
                mpip_handler = mph.MPIPHandler(model_data)
                mpip_separation_handler = msh.MPIPSeparationHandler(
                    mpip_handler, external_solver.opt_model
                )
                solver.mpip_separation_handler = mpip_separation_handler
        runtime = solver.solve_instance()

        logger.info(lsf.info_optimization_finished(runtime))
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.error(lsf.error_exception_occurred(ex))
        logger.warning("%s", traceback.format_exc())


if __name__ == "__main__":
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        run_optimization()
