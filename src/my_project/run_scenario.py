# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""
import traceback

from my_project.opt import model
from my_project.opt import solution
from my_project.read import process_data
from my_project.report import report
import my_project.settings as s
from my_project.utils import inout as ut_io, datareading as ut_dr
from my_project.utils.logger import logger


def run_optimization():
    """
    function that executes the whole optimization process.
    1) get the settings
    2) read the data
    3) create the optimization model
    4) solve the optimization model
    5) process the optimization results
    6) create a report from the optimization results
    """
    try:
        ut_io.config_console_logger()
        config_dict = ut_dr.read_config_file()
        user_settings = s.UserSettings(config_dict)

        ut_io.config_file_logger(user_settings)
        user_settings.save_to_json()

        data = process_data.Data(user_settings)
        data.read_and_preprocess_data()

        opt_model = model.Model(data, user_settings)
        opt_model.build_optimization_model()

        opt_model.set_parameters_and_optimize()

        opt_results = solution.Solution(data, opt_model, user_settings)
        opt_results.create_solution_from_opt_model()

        opt_report = report.Report(opt_results, data, user_settings)
        opt_report.write_report()

        logger.info("Optimization finished successfully.")
        return opt_results
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.error("Error occurred while running optimization: %s", ex)
        logger.warning("%s", traceback.format_exc())
        return {"status": "error", "errorMessages": str(ex)}


if __name__ == "__main__":
    run_optimization()
