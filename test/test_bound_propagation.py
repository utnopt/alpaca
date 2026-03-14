# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pytest
from bs4 import BeautifulSoup

from alpaca.model_data import variable as var, model_data as mde
from alpaca.model_data import constraint as con
from alpaca.model_buildup import expression_tree as etr, bound_propagator as bpr
import alpaca.settings as s
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf

SAMPLE_BOUND_DATA = [
    {
        "name": "linear_constraint_simple",
        "description": "Propagate x + 2y = 10 with x,y in [0, 10]. y should tighten to [0, 5].",
        "settings": {"bound_propagation_rounds": 2},
        "variables": [
            {"name": "x_0", "lb": 0.0, "ub": 10.0},
            {"name": "x_1", "lb": 0.0, "ub": 10.0},
        ],
        "linear_constraints": [
            {
                "name": "c1",
                "coeffs": {"x_0": 1.0, "x_1": 2.0},
                "sense": "==",
                "rhs": 10.0,
            }
        ],
        "xml": None,
        "expected_bounds": {
            "x_0": (0.0, 10.0),
            "x_1": (0.0, 5.0),  # Max y = (10 - 0)/2 = 5
        },
    },
    {
        "name": "linear_constraint_negative_coeff",
        "description": "Propagate x - y <= 2 with x in [0, 5], y in [0, 5].",
        "settings": {"bound_propagation_rounds": 2},
        "variables": [
            {"name": "x_0", "lb": 0.0, "ub": 5.0},
            {"name": "x_1", "lb": 0.0, "ub": 5.0},
        ],
        "linear_constraints": [
            {
                "name": "c1",
                "coeffs": {"x_0": 1.0, "x_1": -1.0},
                "sense": "<=",
                "rhs": 2.0,
            }
        ],
        "xml": None,
        # x - y <= 2  =>  x <= 2 + y. Max x = 2 + 5 = 7 (capped at 5).
        #             =>  -y <= 2 - x => y >= x - 2. Min y = 0 - 2 = -2 (capped at 0).
        # No tightening expected with these specific bounds, let's try stricter constraint.
        # x - y <= -3 (x <= y - 3).
        # x in [0, 5], y in [0, 5].
        # Max x = 5 - 3 = 2. New x in [0, 2].
        # Min y = x + 3 = 0 + 3 = 3. New y in [3, 5].
        # Let's verify this scenario instead:
        "override_constraints_for_test": [
            {
                "name": "c_strict",
                "coeffs": {"x_0": 1.0, "x_1": -1.0},
                "sense": "<=",
                "rhs": -3.0,
            }
        ],
        "expected_bounds": {"x_0": (0.0, 2.0), "x_1": (3.0, 5.0)},
    },
    {
        "name": "expression_square_onedim",
        "description": "y = x^2, x in [-2, 3]. y should be [0, 9].",
        "settings": {"bound_propagation_rounds": 1},
        "variables": [
            {"name": "x_0", "lb": -2.0, "ub": 3.0},
        ],
        "xml": """
        <nl idx="-1">
            <power>
                <variable idx="0"/>
                <number value="2"/>
            </power>
        </nl>
        """,
        "check_root_variable": True,
        "expected_root_bounds": (0.0, 9.0),
    },
    {
        "name": "expression_product_multilinear",
        "description": "z = x*y, x in [2, 5], y in [-3, -1]. z should be [-15, -2].",
        "settings": {"bound_propagation_rounds": 1},
        "variables": [
            {"name": "x_0", "lb": 2.0, "ub": 5.0},
            {"name": "x_1", "lb": -3.0, "ub": -1.0},
        ],
        "xml": """
        <nl idx="-1">
            <product>
                <variable idx="0"/>
                <variable idx="1"/>
            </product>
        </nl>
        """,
        "check_root_variable": True,
        "expected_root_bounds": (-15.0, -2.0),
    },
    {
        "name": "expression_linear_structure",
        "description": "z = 2*x + 3*y + 5. x in [0,1], "
        "y in [1,2]. z in [2*0+3*1+5, 2*1+3*2+5] = [8, 13].",
        "settings": {"bound_propagation_rounds": 1},
        "variables": [
            {"name": "x_0", "lb": 0.0, "ub": 1.0},
            {"name": "x_1", "lb": 1.0, "ub": 2.0},
        ],
        "xml": """
        <nl idx="-1">
            <sum>
                <product>
                    <number value="2"/>
                    <variable idx="0"/>
                </product>
                <product>
                    <number value="3"/>
                    <variable idx="1"/>
                </product>
                <number value="5"/>
            </sum>
        </nl>
        """,
        "check_root_variable": True,
        "expected_root_bounds": (8.0, 13.0),
    },
    {
        "name": "combined_linear_and_expression",
        "description": "Constraint x = y. Expression z = x*y. x in [1, 2], y in [0, 10]. "
        "Constraint propagates y->[1,2]. Expr z->[1,4].",
        "settings": {"bound_propagation_rounds": 3},
        "variables": [
            {"name": "x_0", "lb": 1.0, "ub": 2.0},
            {"name": "x_1", "lb": 0.0, "ub": 10.0},
        ],
        "linear_constraints": [
            {
                "name": "c1",
                "coeffs": {"x_0": 1.0, "x_1": -1.0},
                "sense": "==",
                "rhs": 0.0,
            }
        ],
        "xml": """
        <nl idx="-1">
            <product>
                <variable idx="0"/>
                <variable idx="1"/>
            </product>
        </nl>
        """,
        "check_root_variable": True,
        "expected_root_bounds": (1.0, 4.0),
        "expected_bounds": {"x_1": (1.0, 2.0)},
    },
]


@pytest.mark.parametrize("data", SAMPLE_BOUND_DATA, ids=lambda d: d["name"])
def test_bound_propagation(data):  # pylint: disable=too-many-locals, too-many-branches
    """
    Test for bound propagation logic covering linear constraints and expressions.
    Excludes OBBT.
    """
    # 1. Setup
    user_settings = {"pwl_method": "none"}
    if "settings" in data:
        user_settings.update(data["settings"])

    settings = s.UserSettings(user_settings)
    model_data = mde.ModelData(settings=settings)

    # Add Variables
    for var_info in data["variables"]:
        new_variable = var.Variable(
            name=var_info["name"], lb=var_info["lb"], ub=var_info["ub"], var_type="C"
        )
        model_data.add_variable(new_variable)

    # Add Linear Constraints
    constraints_to_use = data.get(
        "override_constraints_for_test", data.get("linear_constraints", [])
    )

    for con_info in constraints_to_use:
        # Map coeff names to Variable objects
        var_list = []
        for v_name, coeff in con_info["coeffs"].items():
            var_list.append((coeff, model_data.variables[v_name]))
        sense = con_info["sense"]
        if sense == "==":
            sense_const = lsf.constraint_eq()
        elif sense == "<=":
            sense_const = lsf.constraint_leq()
        elif sense == ">=":
            sense_const = lsf.constraint_geq()
        else:
            sense_const = sense

        new_constraint = con.LinearConstraint(
            name=con_info["name"],
            variables=var_list,
            con_type=sense_const,
            rhs=con_info["rhs"],
        )
        model_data.add_constraint(new_constraint)

    # Add Expressions (if any)
    root_var_name = None
    if data["xml"]:
        soup = BeautifulSoup(data["xml"], "xml")
        for element in soup.find_all(string=True):
            if element.isspace():
                element.extract()

        nl_tag = soup.find("nl")
        root_expression_tag = nl_tag.find_next()

        expression_name = f"test_expr_{data['name']}"
        model_data.add_nonlinear_expression(expression_name, root_expression_tag)

        # Decompose to generate the expression objects that bound propagation works on
        expression_tree = etr.ExpressionTree(model_data)
        expression_tree.decompose()

        # Identify the root variable for result checking
        root_var_name = lsf.representative_variable_name(
            model_data.expressions.first_level_nonlinear_expression_keys[0]
        )

    # 2. Propagate Bounds
    propagator = bpr.BoundPropagator(model_data)
    propagator.propagate_bounds()

    # 3. Assertions

    # Check explicitly defined variable bounds
    if "expected_bounds" in data:
        for var_name, (exp_lb, exp_ub) in data["expected_bounds"].items():
            actual_var = model_data.variables[var_name]
            assert (
                actual_var.lb >= exp_lb - 1e-5
            ), f"Variable {var_name} LB expected {exp_lb}, got {actual_var.lb}"
            assert (
                actual_var.ub <= exp_ub + 1e-5
            ), f"Variable {var_name} UB expected {exp_ub}, got {actual_var.ub}"

    # Check root variable of the expression tree
    if data.get("check_root_variable") and root_var_name:
        actual_root = model_data.variables[root_var_name]
        exp_lb, exp_ub = data["expected_root_bounds"]

        assert (
            actual_root.lb >= exp_lb - 1e-5
        ), f"Root expression LB expected {exp_lb}, got {actual_root.lb}"
        assert (
            actual_root.ub <= exp_ub + 1e-5
        ), f"Root expression UB expected {exp_ub}, got {actual_root.ub}"
