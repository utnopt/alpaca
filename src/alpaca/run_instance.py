# -*- coding: utf-8 -*-
"""
This script is a lightweight wrapper to run a single optimization instance.
It takes a single .osil filename as a command-line argument.
"""
import sys

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import model_scip as msc, model_gurobi as mgu
import alpaca.solver.solver as slv
import alpaca.mpip.mpiphandler as mph
from alpaca.mpip.separation import (
    separationhandler_scip as ses,
    separationhandler_gurobi as seg,
)
import alpaca.settings as s
from alpaca.utils import inout as ut_io
from alpaca.utils.logger import logger


def run_single_optimization(osil_file_name):
    """
    Function to execute a single optimization run for a given .osil file.
    It returns the filename and runtime in a parsable string.
    """
    try:
        # Load user settings, but use a base config without a specified file
        config_dict = {"osil_file_name": osil_file_name.strip(".osil"),
                       "external_solver": "gurobi",
                       "number_of_breakpoints": 20}
        user_settings = s.UserSettings(config_dict)

        ut_io.config_console_logger()
        ut_io.config_file_logger(user_settings)
        user_settings.save_to_json()

        # Instantiate model data and the external solver
        model_data = mda.ModelData(user_settings)
        external_solver = (
            msc.ModelScip(model_data, user_settings)
            if user_settings.external_solver == "scip"
            else mgu.ModelGurobi(model_data, user_settings)
        )

        # Initialize the solver
        solver = slv.Solver(external_solver, user_settings)

        # Handle MPIP features if enabled
        if user_settings.feature_mpip:
            mpip_handler = mph.MPIPHandler(
                [
                    model_data.expressions.nonlinear_expressions[expr_key]
                    for expr_key in model_data.expressions.first_level_nonlinear_expression_keys
                ],
                model_data.expressions.bilinear_expressions,
                model_data.expressions.multilinear_expressions,
            )
            if user_settings.external_solver == "scip":
                mpip_separation_handler = ses.SeparationHandler(
                    mpip_handler, external_solver.opt_model
                )
            else:
                mpip_separation_handler = seg.SeparationHandler(
                    mpip_handler, external_solver.opt_model
                )
            solver.mpip_separation_handler = mpip_separation_handler

        # Solve the instance and capture the runtime
        runtime = solver.solve_instance()

        # Log and print the result in a machine-readable format for the shell script
        logger.info("Optimization finished successfully. Runtime: %.2f seconds", runtime)
        print(f"{osil_file_name},{runtime}")
        return {"status": "success", "runtime": runtime}

    except Exception as ex:  # pylint: disable=broad-exception-caught
        # Log and print an error message if the optimization fails
        logger.error(
            "Error occurred while running optimization for %s: %s", osil_file_name, ex
        )
        print(f"{osil_file_name},ERROR: {str(ex)}")
        return {"status": "error", "errorMessages": str(ex)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        of_name = sys.argv[1]
        run_single_optimization(of_name)
    else:
        logger.error("Error: No .osil filename provided as a command-line argument.")
