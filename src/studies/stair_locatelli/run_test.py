# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import os
import sys
import argparse
import traceback
import time

import alpaca as alp
from alpaca.utils.logger import logger


def run_single_stair_locatelli_test(instance_full_path, stair_locatelli_setting):
    """
    Runs a single instance with a specific stair_locatelli setting.

    Args:
        instance_full_path (str): The full path to the .osil file.
        stair_locatelli_setting (int): The setting for the feature (0, 1, or 2).

    Returns:
        float or str: The objective bound value or 'ERROR' if it fails.
    """
    instance_name = os.path.splitext(os.path.basename(instance_full_path))[0]

    try:
        alpaca = alp.read_model_from_osil(instance_full_path)
        alpaca.configure_logging(
            f"../../data/export/logs/{time.strftime('%Y%m%d-%H%M%S')}.log"
        )
        config_dict = {
            "solver_time_limit": 100,
            "external_solver": "gurobi",
            "reformulate_multilinear_to_bilinear": 1,
            "bilinear_handling": 0,
            "pwl_method": "none",
            "feature/stair_locatelli": stair_locatelli_setting,
            "feature/stair_locatelli/grid_size": 5,
            "breakpoint_generation": 0,
            "bound_propagation": 1,
            "bound_propagation_time_limit": 3600,
            "feature/stair_locatelli/obbt_time_limit": 3600,
            "allow_infinite_bounds": 1,
        }
        alpaca.customize_settings(config_dict)
        alpaca.build_pwl_relaxation_solver()

        volume_improvement = 0.0
        if alpaca.user_settings.feature_stair_locatelli:
            volume_improvement = (
                alpaca.stair_locatelli.calculate_mean_bilinear_relaxation_volume_improvement()
            )
        opt_model = alpaca.solver.external_solver.opt_model
        opt_model.model.setParam("NodeLimit", 0)
        opt_model.model.setParam("Cuts", 0)
        opt_model.hide_output()
        alpaca.solve()
        return (
            alpaca.runtime,
            opt_model.model.ObjBound,
            volume_improvement,
        )

    except Exception:  # pylint: disable=broad-except
        logger.error("Error running instance %s", instance_name)
        logger.error(traceback.format_exc())
        return "ERROR", "ERROR", "ERROR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a stair_locatelli test for a given .osil instance."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Full path to the .osil file."
    )

    if len(sys.argv) > 1:
        parsed_args = parser.parse_args()
        instance_name_only = os.path.splitext(os.path.basename(parsed_args.file))[0]

        # Save the original standard output
        original_stdout = sys.stdout

        try:
            # Redirect stdout to suppress unwanted prints from the library
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                sys.stdout = devnull
                # Run for all three settings
                (runtime_without, obj_without, _) = run_single_stair_locatelli_test(
                    parsed_args.file, 0
                )
                (runtime_locatelli, obj_locatelli, volume_improvement_locatelli) = (
                    run_single_stair_locatelli_test(parsed_args.file, 1)
                )
                (
                    runtime_stair_locatelli,
                    obj_stair_locatelli,
                    volume_improvement_stair_locatelli,
                ) = run_single_stair_locatelli_test(parsed_args.file, 2)
                (
                    runtime_indicator_locatelli,
                    obj_indicator_locatelli,
                    volume_improvement_indicator_locatelli,
                ) = run_single_stair_locatelli_test(parsed_args.file, 3)
        finally:
            # Restore the original standard output
            sys.stdout = original_stdout

        # Print the results as a single CSV line. The calling shell script
        # will handle directing this to the results file.
        print(
            f"{instance_name_only},"
            f"{runtime_without},{obj_without},"
            f"{runtime_locatelli},{obj_locatelli},"
            f"{volume_improvement_locatelli},"
            f"{runtime_stair_locatelli},{obj_stair_locatelli},"
            f"{volume_improvement_stair_locatelli},"
            f"{runtime_indicator_locatelli},{obj_indicator_locatelli},"
            f"{volume_improvement_indicator_locatelli}"
        )

    else:
        parser.print_help()
