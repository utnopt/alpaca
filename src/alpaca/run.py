# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import traceback

import alpaca.model_data.model_data as mda
import alpaca.model_scip.model_scip as msc
import alpaca.solver.solver as slv
from alpaca.mpip import mpiphandler as mph, separationhandler as mps
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

        scip_model = msc.ModelScip(model_data, user_settings)

        solver = slv.Solver(scip_model, user_settings)

        if user_settings.feature_mpip:
            mpip_handler = mph.MPIPHandler(
                [
                    model_data.expressions.nonlinear_expressions[expr_key]
                    for expr_key in model_data.expressions.first_level_nonlinear_expression_keys
                ],
                model_data.expressions.bilinear_expressions,
                model_data.expressions.multilinear_expressions,
            )
            mpip_separation_handler = mps.SeparationHandler(
                mpip_handler, scip_model.opt_model
            )
            solver.mpip_separation_handler = mpip_separation_handler

        solver.solve_instance()

        logger.info("Optimization finished successfully.")
        return {"status": "success"}
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.error("Error occurred while running optimization: %s", ex)
        logger.warning("%s", traceback.format_exc())
        return {"status": "error", "errorMessages": str(ex)}


if __name__ == "__main__":
    run_optimization()
