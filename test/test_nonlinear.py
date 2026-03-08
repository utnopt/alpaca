# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pathlib
import pytest

import alpaca as alp


def run_nonlinear_test(instance_name):
    """Helper method to test nonlinear solve for a given instance."""
    solution_dict = {
        "alkyl": -1.76499965,
        "least": 14085.13985000,
        "chance": 29.89437816,
        "st_glmp_kk92": -12.0,
        "blend721": -13.5268,
    }
    base_dir = pathlib.Path(__file__).parent
    file_path = str(base_dir / "test_instances" / f"{instance_name}.osil")
    config_dict = {
        "solver_time_limit": 3600,
        "external_solver": "scip",
        "reformulate_multilinear_to_bilinear": 0,
        "bilinear_handling": 3,
        "allow_infinite_bounds": 1,
        "pwl_method": "none",
    }
    alpaca = alp.read_model_from_osil(file_path)
    alpaca.customize_settings(config_dict)
    alpaca.build_pwl_relaxation_solver()
    alpaca.solve()
    solution_value = alpaca.solver.external_solver.opt_model.get_objective_value()
    assert abs(solution_value - solution_dict[instance_name]) <= 0.05, (
        f"Expected solution {solution_dict[instance_name]} "
        f"but got {solution_value} for instance {instance_name}"
    )


@pytest.mark.parametrize(
    "instance_name", ["alkyl", "least", "chance", "st_glmp_kk92", "blend721"]
)
def test_nonlinear(instance_name):
    """Test original model solve instances."""
    run_nonlinear_test(instance_name)
