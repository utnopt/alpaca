# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import os
import sys
import traceback

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.locatelli.stair_locatelli as slo
import alpaca.settings as s
from alpaca.utils import inout as ut_io
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


def run_test():
    """Test stair locatelli on pooling instances"""
    for instance_name in os.listdir(s.StaticSettings.instances_path):
        if "pooling" in instance_name:
            instance_name = instance_name.strip(".osil")
            try:
                obj_without = run_test_for_instance(instance_name, 0)
                obj_locatelli = run_test_for_instance(instance_name, 1)
                obj_stair_locatelli = run_test_for_instance(instance_name, 2)
                logger.info("Instance: %s", instance_name)
                logger.info("Obj without locatelli: %f", obj_without)
                logger.info("Obj with locatelli: %f", obj_locatelli)
                logger.info("Obj with stair locatelli: %f", obj_stair_locatelli)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.error(traceback.format_exc())


def run_test_for_instance(instance_name, stair_locatelli):
    """Test stair locatelli on pooling instances"""
    ut_io.config_console_logger()
    logger.info("Testing instance: %s", instance_name)
    config_dict = {
        "solver_time_limit": 100,
        "osil_file_name": instance_name,
        "external_solver": "gurobi",
        "reformulate_multilinear_to_bilinear": 1,
        "bilinear_handling": 0,
        "pwl_method": "none",
        "feature/stair_locatelli": stair_locatelli,
        "feature/stair_locatelli/grid_size": 10,
        "bound_propagation": 0,
    }
    user_settings = s.UserSettings(config_dict)
    ut_io.config_file_logger(user_settings)
    user_settings.save_to_json()
    model_data = mda.ModelData(user_settings)
    if user_settings.feature_stair_locatelli:
        slo.StairLocatelli(model_data)
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
    return external_solver.opt_model.model.ObjBound


if __name__ == "__main__":
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        run_test()
