# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code, too-many-locals
"""
This script is a lightweight wrapper to run a single optimization instance.
It takes a .osil file and several configuration settings as command-line arguments.
"""
import sys
import os
import argparse

# Import necessary modules from your existing project structure
import alpaca.model_data.model_data as mda
from alpaca.external_solvers import model_scip as msc, model_gurobi as mgu
import alpaca.solver.solver as slv
import alpaca.mpip.mpiphandler as mph
from alpaca.mpip.separation import (
    separationhandler_scip as ses,
    separationhandler_gurobi as seg,
)
import alpaca.settings as s
from alpaca.utils import inout as ut_io, datareading as ut_dr
from alpaca.utils.logger import logger


def run_single_optimization(args):
    """
    Function to execute a single optimization run for a given .osil file
    and specific settings.
    It returns the results in a parsable CSV string.
    """
    osil_full_path = args.file
    osil_file_name = "unknown"
    original_instances_path = s.StaticSettings.instances_path

    try:
        # Get the directory and filename from the full path
        osil_dir = os.path.dirname(osil_full_path)
        osil_file_name_with_ext = os.path.basename(osil_full_path)
        osil_file_name = os.path.splitext(osil_file_name_with_ext)[0]

        # Temporarily set the instance path to the directory of the current file
        s.StaticSettings.instances_path = osil_dir + "/"

        # Load base user settings from the default config file
        config_dict = ut_dr.read_config_file()
        user_settings = s.UserSettings(config_dict)

        # --- Override settings based on command-line arguments ---
        user_settings.osil_file_name = osil_file_name
        user_settings.number_of_breakpoints = args.breakpoints
        user_settings.feature_mpip = int(args.mpip_stripe)
        user_settings.feature_mpip_separation = int(args.mpip_stripe)
        seed_value = int(args.seed_value)
        # --- End of overrides ---

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
        mpip_handler = None

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
            if not mpip_handler.mpip_dict:
                logger.warning(
                    "MPIP feature enabled, but no MPIP instances found in the model."
                )
            else:
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
        solver.external_solver.opt_model.setIntParam(
            "randomization/randomseedshift", seed_value
        )
        runtime = solver.solve_instance()
        gap = round(solver.external_solver.opt_model.MIPGap, 4)
        nr_nodes = solver.external_solver.opt_model.getNNodes()
        nr_cuts = (
            0
            if not user_settings.feature_mpip
            else sum(
                mpip.separator.nr_of_cuts for mpip in mpip_handler.mpip_dict.values()
            )
        )

        # Log and print the result in the specified CSV format for the shell script
        logger.info(
            "Optimization finished successfully. Runtime: %.2f seconds", runtime
        )
        # Format: number_of_breakpoints,test_case,osil_file_name,runtime,gap
        print(
            f"{args.breakpoints},{args.test_case},{osil_file_name},"
            f"{runtime},{gap},{nr_nodes},{nr_cuts},{seed_value}"
        )

    except Exception as ex:  # pylint: disable=broad-exception-caught
        # Log and print an error message if the optimization fails
        logger.error(
            "Error occurred while running optimization for %s: %s", osil_file_name, ex
        )
        print(f"{args.breakpoints},{args.test_case},{osil_file_name},ERROR,{str(ex)}")

    finally:
        # Restore the original instances path to avoid side effects
        s.StaticSettings.instances_path = original_instances_path


if __name__ == "__main__":
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Run a single optimization instance with specific settings."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Full path to the .osil file."
    )
    parser.add_argument(
        "--breakpoints", type=int, required=True, help="Number of breakpoints to use."
    )
    parser.add_argument(
        "--test_case",
        type=str,
        required=True,
        help="Name of the test case (e.g., MPIP, Standard).",
    )
    parser.add_argument(
        "--mpip_stripe",
        type=int,
        required=True,
        choices=[0, 1],
        help="Enable/disable feature_mpip_stripe (1 or 0).",
    )
    parser.add_argument(
        "--seed_value",
        type=int,
        required=True,
        help="Seed value.",
    )

    # If arguments are provided, run the optimization
    if len(sys.argv) > 1:
        parsed_args = parser.parse_args()
        run_single_optimization(parsed_args)
    else:
        parser.print_help()
