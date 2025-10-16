# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code, too-many-locals
"""
This script is a lightweight wrapper to run a single optimization instance.
It takes a .osil file and several configuration settings as command-line arguments.
"""
import logging
import sys
import os
import argparse
import signal

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.mpip.mpip_handler as mph
import alpaca.mpip.separation.mpip_separationhandler as msh
import alpaca.settings as s
from alpaca.utils import inout as ut_io
from alpaca.utils.logger import logger
import alpaca.utils.error_handling as erh
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


def run_single_optimization(args) -> tuple[str, float, float, int, int, float]:
    # pylint: disable=too-many-statements
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
        config_dict = {
            "osil_file_name": osil_file_name,
            "number_of_breakpoints": args.breakpoints,
            "pwl_method": args.pwl_method,
            "feature/mpip/separation": int(args.test_case.lower() == "mpip"),
            "seed": int(args.seed_value),
            "external_solver": "scip",
            "solver_time_limit": 7200,
            "reformulate_multilinear_to_bilinear": 1,
            "bilinear_handling": 1,
            "breakpoint_generation": 1,
            "feature/nnbp/time_limit": 300,
            "bound_propagation": 1,
            "bound_propagation_time_limit": 300,
        }
        user_settings = s.UserSettings(config_dict)
        ut_io.config_console_logger(log_level=logging.ERROR)
        ut_io.config_file_logger(user_settings)
        user_settings.save_to_json()
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, erh.timeout_handler)
            signal.alarm(3600)
        try:
            model_data = mda.ModelData(user_settings)
        finally:
            # Disable the alarm once the operation is complete or has failed.
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

        external_solver = mm.MIPModel(
            model_data,
            nonlinear=user_settings.pwl_method == lsf.pwl_method_none(),
            bilinear=user_settings.bilinear_handling == 3,
        )

        solver = slv.Solver(external_solver)

        if user_settings.feature_mpip:
            mpip_handler = mph.MPIPHandler(model_data)
            if len(mpip_handler.mpip_dict) == 0:
                raise ValueError("No MPIP structures found.")
            mpip_separation_handler = msh.MPIPSeparationHandler(
                mpip_handler, external_solver.opt_model
            )
            solver.mpip_separation_handler = mpip_separation_handler
        seed_value = int(args.seed_value)
        solver.external_solver.opt_model.set_seed(seed_value)
        runtime = solver.solve_instance()
        mip_gap = solver.external_solver.opt_model.get_mip_gap()
        nr_cuts = (
            0
            if not user_settings.feature_mpip
            else sum(
                mpip.separator.nr_of_cuts for mpip in mpip_handler.mpip_dict.values()
            )
        )
        nr_applied_cuts = solver.external_solver.opt_model.get_nr_of_applied_cuts()
        mpip_ratio = (
            0
            if not user_settings.feature_mpip
            else round(
                sum(
                    mpip.calculate_relation_ratio()
                    for mpip in mpip_handler.mpip_dict.values()
                )
                / len(mpip_handler.mpip_dict),
                3,
            )
        )
        return osil_file_name, runtime, mip_gap, nr_cuts, nr_applied_cuts, mpip_ratio

    except Exception as ex:  # pylint: disable=broad-exception-caught
        # Log and print an error message if the optimization fails
        logger.error(
            "Error occurred while running optimization for %s: %s", osil_file_name, ex
        )
        return osil_file_name + "_ERROR", -1.0, -1.0, -1, -1, -1.0

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
        "--seed_value",
        type=int,
        required=True,
        help="Seed value.",
    )
    parser.add_argument(
        "--pwl_method",
        type=str,
        required=True,
        help="Pwl method.",
    )

    if len(sys.argv) > 1:
        parsed_args = parser.parse_args()
        original_stdout = sys.stdout

        try:
            # Redirect stdout to suppress unwanted prints from the library
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                sys.stdout = devnull
                results = run_single_optimization(parsed_args)
        finally:
            # Restore the original standard output
            sys.stdout = original_stdout

        # Print the results as a single CSV line. The calling shell script
        # will handle directing this to the results file.
        print(
            f"{parsed_args.test_case},{parsed_args.pwl_method},"
            f"{parsed_args.breakpoints},{parsed_args.seed_value},"
            + ",".join(map(str, results))
        )
    else:
        parser.print_help()
