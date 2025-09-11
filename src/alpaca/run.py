# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import traceback

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.mpip.mpiphandler as mph
import alpaca.mpip.separation.mpip_separationhandler as msh
import alpaca.settings as s
from alpaca.utils import inout as ut_io, datareading as ut_dr
from alpaca.utils.logger import logger


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
            mpip_handler = mph.MPIPHandler(model_data)
            mpip_separation_handler = msh.MPIPSeparationHandler(
                mpip_handler, external_solver.opt_model
            )
            solver.mpip_separation_handler = mpip_separation_handler
        runtime = solver.solve_instance()

        logger.info(
            "Optimization finished successfully. Runtime: %.2f seconds", runtime
        )
        return {"status": "success"}
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.error("Error occurred while running optimization: %s", ex)
        logger.warning("%s", traceback.format_exc())
        return {"status": "error", "errorMessages": str(ex)}


if __name__ == "__main__":
    run_optimization()
