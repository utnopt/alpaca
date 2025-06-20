# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import traceback

import alpaca.model_data.model_data as mda
import alpaca.mpip.mpiphandler as mph
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
        model_data.build_model_from_osil_data()

        mpip_handler = mph.MPIPHandler(model_data.first_level_nonlinear_expressions)
        mpip_handler.find_mpip_instances_in_nonlinear_expression()

        logger.info("Optimization finished successfully.")
        return {"status": "success"}
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.error("Error occurred while running optimization: %s", ex)
        logger.warning("%s", traceback.format_exc())
        return {"status": "error", "errorMessages": str(ex)}


if __name__ == "__main__":
    run_optimization()
