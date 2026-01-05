# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pytest
import numpy as np

from alpaca.model_data import variable as var, model_data as mde
from alpaca.expressions import one_dim_expression as ode
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.settings as s


@pytest.mark.parametrize("method", ["delta", "multiple_choice"])
@pytest.mark.parametrize(
    "one_dim_expression",
    [
        ode.SineExpression,
        ode.AbsExpression,
        ode.SquareExpression,
        ode.TangensHExpression
    ],
)
def test_approximation_accuracy(method, one_dim_expression):
    """Test the accuracy of PWL approximations for a sample function."""
    test_variable = var.Variable(name="x", lb=0.0, ub=10.0, var_type="C")
    test_variable.is_discretized = True
    settings = s.UserSettings({"number_of_breakpoints": 24, "pwl_method": method})
    model_data = mde.ModelData(settings=settings)
    model_data.variables = {"x": test_variable}
    test_expression = model_data.add_one_dim_expression(
        one_dim_expression, "test_expression", test_variable, 0
    )
    model_data.build_pwl_relaxation_model()
    external_solver = mm.MIPModel(model_data, nonlinear=False)
    external_solver.opt_model.hide_output()
    solver = slv.Solver(external_solver)
    checkpoints = np.linspace(test_variable.lb, test_variable.ub, num=10)
    for bp in checkpoints:
        true_value = test_expression.f(bp)
        solver.external_solver.opt_model.set_objective(
            test_expression.representative_variable.solver_variable
        )
        external_solver.opt_model.set_variable_lb(test_variable.solver_variable, bp)
        external_solver.opt_model.set_variable_ub(test_variable.solver_variable, bp)
        solver.external_solver.opt_model.optimize()
        lb = solver.external_solver.opt_model.get_val(
            test_expression.representative_variable.solver_variable
        )
        solver.external_solver.opt_model.set_objective(
            -test_expression.representative_variable.solver_variable
        )
        solver.external_solver.opt_model.optimize()
        ub = solver.external_solver.opt_model.get_val(
            test_expression.representative_variable.solver_variable
        )
        assert (
            ub >= true_value - s.StaticSettings.feasibility_tolerance
        ), f"Overestimation infeasible for {bp}"
        assert (
            lb <= true_value + s.StaticSettings.feasibility_tolerance
        ), f"Underestimation infeasible for {bp}"
        assert (
            abs(true_value - ub) < 0.1
        ), f"Overestimation too high at {bp} ({ub - true_value})"
        assert (
            abs(true_value - lb) < 0.1
        ), f"Underestimation too high at {bp} ({true_value - lb})"
