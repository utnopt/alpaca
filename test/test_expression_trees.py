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
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf

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
        "test_vectors": [
            {"inputs": {"x_0": 0.5}, "expected_output": math.sin(math.cos(0.5))},
            {"inputs": {"x_0": -0.5}, "expected_output": math.sin(math.cos(-0.5))},
            {"inputs": {"x_0": 0.0}, "expected_output": math.sin(math.cos(0.0))},
            {"inputs": {"x_0": 3.14}, "expected_output": math.sin(math.cos(3.14))},
        ],
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
        "test_vectors": [
            {
                "inputs": {"x_0": 2.0, "x_1": 3.0},
                "expected_output": 2.0**2 * 3.0,
            },
            {
                "inputs": {"x_0": -2.0, "x_1": 3.0},
                "expected_output": (-2.0) ** 2 * 3.0,
            },
            {
                "inputs": {"x_0": 0.5, "x_1": -4.0},
                "expected_output": 0.5**2 * -4.0,
            },
        ],
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
        "test_vectors": [
            {
                "inputs": {"x_2": 1.5, "x_3": 1.0, "x_5": 4.0, "x_8": 0.92},
                "expected_output": ((0.01 * 1.0 * 0.92 + 1.5) * -4.0) + 0.98 * 1.5,
            },
            {
                "inputs": {"x_2": 0.0, "x_3": 10.0, "x_5": 2.0, "x_8": 0.5},
                "expected_output": ((0.01 * 10.0 * 0.5 + 0.0) * -2.0) + 0.98 * 0.0,
            },
        ],
    },
    {
        "name": "negative_product_coeff",
        "description": "Testing number * variable product structure: (-2 * x) * (y + 3) - 5*z",
        "xml": """
        <nl idx="-1">
            <sum>
                <product>
                    <product>
                        <number value="-2"/>
                        <variable idx="0"/>
                    </product>
                    <sum>
                        <variable idx="1"/>
                        <number value="3"/>
                    </sum>
                </product>
                <variable idx="2" coef="-5"/>
            </sum>
        </nl>
        """,
        "variables": [
            {"name": "x_0", "idx": 0},
            {"name": "x_1", "idx": 1},
            {"name": "x_2", "idx": 2},
        ],
        "test_vectors": [
            {
                "inputs": {"x_0": 1.0, "x_1": 2.0, "x_2": 1.0},
                "expected_output": (-2 * 1.0) * (2.0 + 3) - 5 * 1.0,  # -10 - 5 = -15
            },
            {
                "inputs": {"x_0": -1.0, "x_1": 0.0, "x_2": 0.0},
                "expected_output": (-2 * -1.0) * (0.0 + 3) - 5 * 0.0,  # 2 * 3 = 6
            },
            {
                "inputs": {"x_0": 0.5, "x_1": -5.0, "x_2": -2.0},
                "expected_output": (-2 * 0.5) * (-5.0 + 3)
                - 5 * -2.0,  # -1 * -2 + 10 = 12
            },
        ],
    },
    {
        "name": "complex_polynomial_fraction",
        "description": "2.5*x^3 - 4*x*y + 10",
        "xml": """
        <nl idx="-1">
            <sum>
                <product>
                    <number value="2.5"/>
                    <power>
                        <variable idx="0"/>
                        <number value="3"/>
                    </power>
                </product>
                <product>
                    <number value="-4"/>
                    <variable idx="0"/>
                    <variable idx="1"/>
                </product>
                <number value="10"/>
            </sum>
        </nl>
        """,
        "variables": [
            {"name": "x_0", "idx": 0},
            {"name": "x_1", "idx": 1},
        ],
        "test_vectors": [
            {
                "inputs": {"x_0": 1.0, "x_1": 1.0},
                "expected_output": 2.5 * (1**3) - 4 * 1 * 1 + 10,  # 8.5
            },
            {
                "inputs": {"x_0": 2.0, "x_1": 0.5},
                "expected_output": 2.5 * (2**3)
                - 4 * 2 * 0.5
                + 10,  # 2.5*8 - 4 + 10 = 20 - 4 + 10 = 26
            },
            {
                "inputs": {"x_0": -1.0, "x_1": -2.0},
                "expected_output": 2.5 * ((-1) ** 3)
                - 4 * (-1) * (-2)
                + 10,  # -2.5 - 8 + 10 = -0.5
            },
        ],
    },
]


@pytest.mark.parametrize("data", SAMPLE_OSIL_DATA, ids=lambda d: d["name"])
def test_expression_decomposition_detailed(data):  # pylint: disable=too-many-locals
    """
    Detailed test for expression tree decomposition with multiple test vectors per structure.
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

    # 3. Solve for each test vector
    # We build the solver once and re-optimize for different inputs
    external_solver = mm.MIPModel(model_data, nonlinear=True)
    external_solver.opt_model.hide_output()
    solver = slv.Solver(external_solver)

    result_variable = model_data.variables[
        lsf.representative_variable_name(
            model_data.expressions.first_level_nonlinear_expression_keys[0]
        )
    ].solver_variable
    for data_vector in data["test_vectors"]:
        solver.external_solver.opt_model.set_objective(result_variable)
        data_vector: dict[str, dict | float]
        inputs: dict = data_vector["inputs"]
        expected_val: float = data_vector["expected_output"]

        # Reset bounds for the current test vector inputs
        for var_name, checkpoint in inputs.items():
            solver_variable = model_data.variables[var_name].solver_variable
            external_solver.opt_model.set_variable_lb(solver_variable, checkpoint)
            external_solver.opt_model.set_variable_ub(solver_variable, checkpoint)

        solver.external_solver.opt_model.optimize()

        result_value = solver.external_solver.opt_model.get_val(result_variable)

        # Assertion with detailed failure message
        assert abs(result_value - expected_val) < 1e-4, (
            f"Expression '{data['description']}' with inputs {inputs} evaluated to {result_value}, "
            f"expected {expected_val}"
        )
