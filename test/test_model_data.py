# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pathlib
import pytest

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.mpip.mpip_handler as mph
import alpaca.mpip.separation.mpip_separationhandler as msh
import alpaca.settings as s


def run_model_test(instance_name, approximation, reformulate_multilinear):
    """Helper method to test model creation for a given instance."""
    base_dir = pathlib.Path(__file__).parent
    file_path = str(base_dir / "test_instances" / f"{instance_name}.osil")
    config_dict = {
        "approximation": approximation,
        "reformulate_multilinear": reformulate_multilinear,
        "allow_infinite_bounds": 1,
    }
    user_settings = s.UserSettings(config_dict)

    model_data = mda.ModelData(user_settings)
    model_data.read_model_from_osil_data(file_path)
    model_data.build_pwl_relaxation_model()

    external_solver = mm.MIPModel(model_data, nonlinear=False)
    solver = slv.Solver(external_solver)
    mpip_handler = mph.MPIPHandler(model_data)
    mpip_separation_handler = msh.MPIPSeparationHandler(
        mpip_handler, external_solver.opt_model
    )
    solver.mpip_separation_handler = mpip_separation_handler


@pytest.mark.parametrize(
    "instance_name", ["alkyl", "least", "chance", "chem", "st_glmp_kk92"]
)
@pytest.mark.parametrize("approximation", [0, 1])
@pytest.mark.parametrize("reformulate_multilinear", [0, 1])
def test_model_data(instance_name, approximation, reformulate_multilinear):
    """Test model data creation for various instances and configurations."""
    run_model_test(instance_name, approximation, reformulate_multilinear)
