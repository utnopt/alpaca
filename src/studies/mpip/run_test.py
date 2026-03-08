# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code, too-many-locals, too-many-branches
"""
@authors: kuen,
"""
import time
import sys
import os
import argparse

import alpaca as alp
from alpaca.utils.logger import logger


def run_single_optimization(args) -> tuple[str, float, float, int, int, float]:
    # pylint: disable=too-many-statements
    """
    Function to execute a single optimization run for a given .osil file
    and specific settings.
    It returns the results in a parsable CSV string.
    """
    osil_full_path = args.file
    osil_file_name = "unknown"

    try:
        osil_file_name_with_ext = os.path.basename(osil_full_path)
        osil_file_name = os.path.splitext(osil_file_name_with_ext)[0]
        config_dict = {
            "number_of_breakpoints": args.breakpoints,
            "pwl_method": args.pwl_method,
            "feature/mpip/bar": int(args.test_case.lower() == "mpip"),
            "seed": int(args.seed_value),
            "feature/mpip/frequency": 50,
            "external_solver": "gurobi",
            "solver_time_limit": 18000,
            "reformulate_multilinear_to_bilinear": 1,
            "bilinear_handling": 2,
            "breakpoint_generation": 0,
            "feature/nnbp/time_limit": 3600,
            "bound_propagation": 0,
            "bound_propagation_time_limit": 3600,
            "allow_infinite_bounds": 0,
        }
        alpaca = alp.read_model_from_osil(osil_full_path)
        alpaca.configure_logging(
            f"../../data/export/logs/{time.strftime('%Y%m%d-%H%M%S')}.log"
        )
        alpaca.customize_settings(config_dict)
        alpaca.build_pwl_relaxation_solver()
        alpaca.solver.external_solver.opt_model.hide_output()
        mpip_handler = None
        if alpaca.user_settings.feature_mpip:
            mpip_handler = alpaca.solver.mpip_separation_handler.mpip_handler
            if len(mpip_handler.mpip_dict) == 0:
                raise ValueError("No MPIP structures found.")
        alpaca.solver.external_solver.opt_model.set_seed(int(args.seed_value))
        alpaca.solve()
        opt_model = alpaca.solver.external_solver.opt_model
        mip_gap = opt_model.get_mip_gap()
        nr_cuts = (
            0
            if not alpaca.user_settings.feature_mpip
            else sum(
                mpip.separator.nr_of_cuts for mpip in mpip_handler.mpip_dict.values()
            )
        )
        nr_applied_cuts = opt_model.get_nr_of_applied_cuts()
        mpip_ratio = (
            0
            if not alpaca.user_settings.feature_mpip
            else round(
                sum(
                    mpip.calculate_relation_ratio()
                    for mpip in mpip_handler.mpip_dict.values()
                )
                / len(mpip_handler.mpip_dict),
                3,
            )
        )
        return (
            osil_file_name,
            alpaca.statistics.solving_time,
            mip_gap,
            nr_cuts,
            nr_applied_cuts,
            mpip_ratio,
        )

    except Exception as ex:  # pylint: disable=broad-exception-caught
        # Log and print an error message if the optimization fails
        logger.error(
            "Error occurred while running optimization for %s: %s", osil_file_name, ex
        )
        return osil_file_name + "_ERROR", -1.0, -1.0, -1, -1, -1.0


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
