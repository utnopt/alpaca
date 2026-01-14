# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import os
import sys
import argparse
import traceback

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.locatelli.stair_locatelli as slo
import alpaca.settings as s
from alpaca.utils import inout as ut_io
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


def run_single_stair_locatelli_test(instance_full_path, stair_locatelli_setting):
    """
    Runs a single instance with a specific stair_locatelli setting.

    Args:
        instance_full_path (str): The full path to the .osil file.
        stair_locatelli_setting (int): The setting for the feature (0, 1, or 2).

    Returns:
        float or str: The objective bound value or 'ERROR' if it fails.
    """
    original_instances_path = s.StaticSettings.instances_path
    instance_dir = os.path.dirname(instance_full_path)
    instance_name = os.path.splitext(os.path.basename(instance_full_path))[0]

    try:
        # Temporarily set the instance path to the directory of the current file
        s.StaticSettings.instances_path = instance_dir + "/"

        ut_io.config_console_logger()
        logger.info(
            "Testing instance: %s with stair_locatelli=%d",
            instance_name,
            stair_locatelli_setting,
        )

        config_dict = {
            "solver_time_limit": 100,
            "osil_file_name": instance_name,
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
        }

        user_settings = s.UserSettings(config_dict)
        ut_io.config_file_logger(user_settings)
        user_settings.save_to_json()

        model_data = mda.ModelData(user_settings)
        volume_improvement = 0.0
        if user_settings.feature_stair_locatelli:
            stair_locatelli = slo.StairLocatelli(model_data)
            volume_improvement = (
                stair_locatelli.calculate_mean_bilinear_relaxation_volume_improvement()
            )

        external_solver = mm.MIPModel(
            model_data,
            nonlinear=user_settings.pwl_method == lsf.pwl_method_none(),
            bilinear=user_settings.bilinear_handling == 3,
        )

        external_solver.opt_model.model.setParam("NodeLimit", 0)
        external_solver.opt_model.model.setParam("Cuts", 0)

        solver = slv.Solver(external_solver)
        runtime = solver.solve_instance()

        logger.info(lsf.info_optimization_finished(runtime))
        return (
            runtime,
            external_solver.opt_model.model.ObjBound,
            volume_improvement,
        )

    except Exception:  # pylint: disable=broad-except
        logger.error("Error running instance %s", instance_name)
        logger.error(traceback.format_exc())
        return "ERROR", "ERROR", "ERROR"
    finally:
        # Restore original path to avoid side effects
        s.StaticSettings.instances_path = original_instances_path


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
