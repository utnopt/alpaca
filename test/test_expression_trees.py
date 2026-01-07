# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import math
import pytest
from bs4 import BeautifulSoup

from alpaca.model_data import variable as var, model_data as mde
from alpaca.model_buildup import expression_tree as etr
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.settings as s
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


SAMPLE_OSIL_DATA = [
    {
        "name": "nested_trig",
        "description": "sin(cos(x))",
        "xml": """
        <nl idx="-1">
            <sin>
                <cos>
                    <variable idx="0" coef="1"/>
                </cos>
            </sin>
        </nl>
        """,
        "variables": [{"name": "x_0", "idx": 0}],
        "inputs": {"x_0": 0.5},
        "expected_output": math.sin(math.cos(0.5)),
    },
    {
        "name": "power_times_var",
        "description": "(x^2) * y",
        "xml": """
        <nl idx="-1">
            <product>
                <power>
                    <variable idx="0" coef="1"/>
                    <number value="2"/>
                </power>
                <variable idx="1" coef="1"/>
            </product>
        </nl>
        """,
        "variables": [{"name": "x_0", "idx": 0}, {"name": "x_1", "idx": 1}],
        "inputs": {"x_0": 2.0, "x_1": 3.0},
        "expected_output": 2.0**2 * 3.0,
    },
    {
        "name": "alkyl_complex_sum",
        "description": "alkyl snippet: ((0.01*x3*x8 + x2) * -x5) + 0.98*x2",
        "xml": """
        <nl idx="1">
            <sum>
                <product>
                    <sum>
                        <product>
                            <variable idx="3" coef="1e-2"/>
                            <variable idx="8"/>
                        </product>
                        <variable idx="2"/>
                    </sum>
                    <variable idx="5" coef="-1"/>
                </product>
                <variable idx="2" coef=".98"/>
            </sum>
        </nl>
        """,
        "variables": [
            {"name": "x_2", "idx": 2},
            {"name": "x_3", "idx": 3},
            {"name": "x_5", "idx": 5},
            {"name": "x_8", "idx": 8},
        ],
        "inputs": {"x_2": 1.5, "x_3": 1.0, "x_5": 4.0, "x_8": 0.92},
        "expected_output": ((0.01 * 1.0 * 0.92 + 1.5) * -4.0) + 0.98 * 1.5,
    },
]


@pytest.mark.parametrize("data", SAMPLE_OSIL_DATA, ids=lambda d: d["name"])
def test_expression_decomposition_detailed(data):  # pylint: disable=too-many-locals
    """
    Detailed test for expression tree decomposition.
    """
    # 1. Setup
    settings = s.UserSettings({"pwl_method": "none", "bilinear_handling": 3})
    model_data = mde.ModelData(settings=settings)

    # Dictionary for easy lookup during variable indexing in OSiL-like tags
    for var_info in data["variables"]:
        new_variable = var.Variable(
            name=var_info["name"], lb=-10.0, ub=10.0, var_type="C"
        )
        model_data.add_variable(new_variable)

    soup = BeautifulSoup(data["xml"], "xml")
    for element in soup.find_all(string=True):
        if element.isspace():
            element.extract()

    nl_tag = soup.find("nl")

    # The actual tag used in OSiL for parsing is the child of <nl>
    root_expression_tag = nl_tag.find_next()

    expression_name = f"test_expr_{data['name']}"
    # add_nonlinear_expression creates a NonlinearExpression object internally
    model_data.add_nonlinear_expression(expression_name, root_expression_tag)

    # 2. Decompose
    expression_tree = etr.ExpressionTree(model_data)
    expression_tree.decompose()

    model_data.build_pwl_relaxation_model()
    external_solver = mm.MIPModel(model_data, nonlinear=True)
    external_solver.opt_model.hide_output()
    solver = slv.Solver(external_solver)
    result_variable = model_data.variables[
        lsf.representative_variable_name(
            model_data.expressions.first_level_nonlinear_expression_keys[0]
        )
    ].solver_variable
    solver.external_solver.opt_model.set_objective(result_variable)
    for var_name, checkpoint in data["inputs"].items():
        solver_variable = model_data.variables[var_name].solver_variable
        external_solver.opt_model.set_variable_lb(solver_variable, checkpoint)
        external_solver.opt_model.set_variable_ub(solver_variable, checkpoint)
    solver.external_solver.opt_model.optimize()
    result_value = solver.external_solver.opt_model.get_val(result_variable)
    expected_value = data["expected_output"]
    assert abs(result_value - expected_value) < 1e-4, (
        f"Expression '{data['description']}' evaluated to {result_value}, "
        f"expected {expected_value}"
    )
