"""
Unit tests for Piecewise Linear (PWL) and Constant (PWC) relaxations.

This module tests the creation of PWL formulations, including domain
representation (e.g., Multiple Choice, Delta) and the coupling of functions to
these domains for both approximations and relaxations on convex and non-convex
functions.
"""

# pylint: disable=protected-access

import unittest
import math
from types import SimpleNamespace

# Import the actual classes needed for the tests
from alpaca.model_data.variable import Variable
from alpaca.expressions.one_dim_expression import SquareExpression, SineExpression
from alpaca.expressions.multilinear_expression import MultilinearExpression
from alpaca.pwl.multiple_choice_method import MultipleChoiceMethod
from alpaca.pwl.delta_method import DeltaMethod
from alpaca.model_buildup.pwl_handler import PWLHandler
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


# Mock settings to avoid dependency on the full settings module
class MockUserSettings:
    """A mock user settings class for testing purposes."""

    def __init__(self, approximation=False, pwl_method="multiple-choice"):
        """
        Initializes mock settings for PWL tests.

        Args:
            approximation (bool): Whether to use approximation.
            pwl_method (str): The PWL method to use.
        """
        self.approximation = approximation
        self.pwl_method = pwl_method
        self.bilinear_handling = 2
        self.reformulate_multilinear_to_bilinear = False


# Mock ModelData to isolate the PWL logic
class MockModelData:
    """A mock model data class to provide a testing environment for PWL."""

    def __init__(self, approximation=False, pwl_method="multiple-choice"):
        """
        Initializes mock model data.

        Args:
            approximation (bool): Whether to use approximation.
            pwl_method (str): The PWL method to use.
        """
        self.settings = MockUserSettings(approximation, pwl_method)
        self.variables = {}
        self.constraints = {}
        # Keep track of expressions for the handler test
        self.expressions = SimpleNamespace(
            one_dim_expressions={}, multilinear_expressions={}, bilinear_expressions={}
        )

    def add_variable(self, var):
        """Adds a variable to the mock model data."""
        self.variables[var.name] = var

    def add_constraint(self, con):
        """Adds a constraint to the mock model data."""
        self.constraints[con.name] = con


class TestPWLRelaxations(unittest.TestCase):
    """Test cases for piecewise linear and constant relaxations."""

    def setUp(self):
        """Set up common variables and expressions for tests."""
        self.model_data_approx_mc = MockModelData(
            approximation=True, pwl_method="multiple_choice"
        )
        self.model_data_relax_mc = MockModelData(
            approximation=False, pwl_method="multiple_choice"
        )
        self.model_data_approx_delta = MockModelData(
            approximation=True, pwl_method="delta"
        )
        self.model_data_relax_delta = MockModelData(
            approximation=False, pwl_method="delta"
        )

    def test_multiple_choice_domain_representation(self):
        """Test Multiple Choice method for a variable's domain representation."""
        x_var = Variable("x", lb=0, ub=10)
        x_var.breakpoints = [0, 3, 7, 10]  # 3 intervals
        mc_method = MultipleChoiceMethod(x_var)

        # 3 binary variables (one for each interval)
        self.assertEqual(len(mc_method.pwl_variables_binary), 3)
        self.assertEqual(
            mc_method.pwl_variables_binary[0].name,
            lsf.var_name_pwl_multiple_choice_binary("x", 0),
        )

        # 3 continuous variables
        self.assertEqual(len(mc_method.pwl_variables_continuous), 3)
        self.assertEqual(
            mc_method.pwl_variables_continuous[0].name,
            lsf.var_name_pwl_multiple_choice_continuous("x", 0),
        )

        # 1 SOS, 1 link, 3 LB, 3 UB = 8 constraints
        self.assertEqual(len(mc_method.pwl_constraints), 8)
        con_sum_cont = next(
            c
            for c in mc_method.pwl_constraints
            if c.name == lsf.con_name_pwl_multiple_choice_variable_link_continuous("x")
        )
        vars_in_con = {var.name: coeff for coeff, var in con_sum_cont.variables}
        self.assertAlmostEqual(vars_in_con["x"], -1.0)
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_continuous("x", 0)], 1.0
        )

    def test_delta_domain_representation(self):
        """Test Delta method for a variable's domain representation."""
        x_var = Variable("x", lb=0, ub=10)
        x_var.breakpoints = [0, 3, 7, 10]  # 4 breakpoints, 3 intervals
        delta_method = DeltaMethod(x_var)

        # n-2 binary variables
        self.assertEqual(len(delta_method.pwl_variables_binary), 2)
        self.assertEqual(
            delta_method.pwl_variables_binary[0].name,
            lsf.var_name_pwl_delta_binary("x", 0),
        )

        # n-1 continuous variables
        self.assertEqual(len(delta_method.pwl_variables_continuous), 3)
        self.assertEqual(
            delta_method.pwl_variables_continuous[0].name,
            lsf.var_name_pwl_delta_continuous("x", 0),
        )
        self.assertEqual(delta_method.pwl_variables_continuous[0].ub, 1.0)

        # 1 link, 2*(n-2) interval constraints = 1 + 4 = 5 constraints
        self.assertEqual(len(delta_method.pwl_constraints), 5)
        link_con = next(c for c in delta_method.pwl_constraints if "varlink" in c.name)
        vars_in_con = {var.name: coeff for coeff, var in link_con.variables}
        self.assertAlmostEqual(link_con.rhs, -x_var.lb)
        self.assertAlmostEqual(vars_in_con["x"], -1.0)
        # Coeff is (bp[i+1] - bp[i])
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_delta_continuous("x", 0)], 3.0
        )
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_delta_continuous("x", 1)], 4.0
        )

    def test_pwl_approximation_one_dim_mc(self):
        """Test the PWL approximation for y = x^2 with Multiple Choice."""
        x_var = Variable("x", lb=0, ub=4)
        x_var.breakpoints = [0, 2, 4]
        y_var = Variable("y")

        x_var.pwl = MultipleChoiceMethod(x_var)
        sq_expr = SquareExpression("sq1", self.model_data_approx_mc, x_var, 0, y_var)
        constraints = x_var.pwl.couple_domain_to_function_value_one_dim(
            sq_expr, approximation=True
        )
        self.assertEqual(len(constraints), 1)

        approx_con = constraints[0]
        self.assertEqual(
            approx_con.name, lsf.con_name_pwl_multiple_choice_approximation("sq1")
        )
        vars_in_con = {var.name: coeff for coeff, var in approx_con.variables}
        self.assertAlmostEqual(vars_in_con["y"], -1.0)
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_continuous("x", 0)], 2.0
        )  # slope for [0,2]
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_binary("x", 0)], 0.0
        )  # intercept for [0,2]
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_continuous("x", 1)], 6.0
        )  # slope for [2,4]
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_binary("x", 1)], -8.0
        )  # intercept for [2,4]

    def test_pwl_approximation_one_dim_delta(self):
        """Test the PWL approximation for y = x^2 with Delta Method."""
        x_var = Variable("x", lb=0, ub=4)
        x_var.breakpoints = [0, 2, 4]
        y_var = Variable("y")

        x_var.pwl = DeltaMethod(x_var)
        sq_expr = SquareExpression("sq1", self.model_data_approx_delta, x_var, 0, y_var)
        constraints = x_var.pwl.couple_domain_to_function_value_one_dim(
            sq_expr, approximation=True
        )
        self.assertEqual(len(constraints), 1)
        approx_con = constraints[0]
        self.assertEqual(approx_con.name, lsf.con_name_pwl_delta_approximation("sq1"))
        self.assertAlmostEqual(approx_con.rhs, -sq_expr.f(x_var.breakpoints[0]))

        vars_in_con = {var.name: coeff for coeff, var in approx_con.variables}
        self.assertAlmostEqual(vars_in_con["y"], -1.0)
        # Coeff is f(bp[i+1]) - f(bp[i])
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_delta_continuous("x", 0)], 4.0
        )  # f(2)-f(0)
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_delta_continuous("x", 1)], 12.0
        )  # f(4)-f(2)

    def test_pwl_relaxation_one_dim_convex_mc(self):
        """Test PWL relaxation for a convex function y = x^2 with Multiple Choice."""
        x_var = Variable("x", lb=0, ub=4)
        x_var.breakpoints = [0, 2, 4]
        y_var = Variable("y")

        x_var.pwl = MultipleChoiceMethod(x_var)
        sq_expr = SquareExpression("sq1", self.model_data_relax_mc, x_var, 0, y_var)
        constraints = x_var.pwl.couple_domain_to_function_value_one_dim(
            sq_expr, approximation=False
        )
        self.assertEqual(len(constraints), 2)
        under_con = next(c for c in constraints if "under" in c.name)
        over_con = next(c for c in constraints if "over" in c.name)

        # For a convex function, max_dev is 0 at breakpoints, min_dev is negative between.
        # For x^2 on [0,2], min_dev=-1, max_dev=0. Intercept=0.
        # For x^2 on [2,4], min_dev=-1, max_dev=0. Intercept=-8.
        under_vars = {var.name: coeff for coeff, var in under_con.variables}
        self.assertAlmostEqual(
            under_vars[lsf.var_name_pwl_multiple_choice_binary("x", 0)], -1.0
        )  # 0 + (-1)
        self.assertAlmostEqual(
            under_vars[lsf.var_name_pwl_multiple_choice_binary("x", 1)], -9.0
        )  # -8 + (-1)

        over_vars = {var.name: coeff for coeff, var in over_con.variables}
        self.assertAlmostEqual(
            over_vars[lsf.var_name_pwl_multiple_choice_binary("x", 0)], 0.0
        )  # 0 + 0
        self.assertAlmostEqual(
            over_vars[lsf.var_name_pwl_multiple_choice_binary("x", 1)], -8.0
        )  # -8 + 0

    def test_pwl_relaxation_one_dim_convex_delta(self):
        """Test PWL relaxation for a convex function y = x^2 with Delta Method."""
        x_var = Variable("x", lb=0, ub=4)
        x_var.breakpoints = [0, 2, 4]
        y_var = Variable("y")

        x_var.pwl = DeltaMethod(x_var)
        sq_expr = SquareExpression("sq1", self.model_data_relax_delta, x_var, 0, y_var)
        constraints = x_var.pwl.couple_domain_to_function_value_one_dim(
            sq_expr, approximation=False
        )
        self.assertEqual(len(constraints), 2)
        under_con = next(c for c in constraints if "under" in c.name)
        over_con = next(c for c in constraints if "over" in c.name)

        # For x^2 on [0,2], min_dev0=-1, max_dev0=0.
        # For x^2 on [2,4], min_dev1=-1, max_dev1=0.
        # Under: rhs = -f(bp0)-min_dev0 = -0-(-1) = 1.
        # Coeff for z0 is min_dev1-min_dev0 = -1 - (-1) = 0.
        self.assertAlmostEqual(under_con.rhs, 1.0)
        under_vars = {var.name: coeff for coeff, var in under_con.variables}
        self.assertNotIn(lsf.var_name_pwl_delta_binary("x", 1), under_vars)

        # Over: rhs = -f(bp0)-max_dev0 = -0-0 = 0. Coeff for z0 is max_dev1-max_dev0 = 0 - 0 = 0.
        self.assertAlmostEqual(over_con.rhs, 0.0)
        over_vars = {var.name: coeff for coeff, var in over_con.variables}
        self.assertNotIn(lsf.var_name_pwl_delta_binary("x", 1), over_vars)

    def test_pwl_relaxation_one_dim_nonconvex_mc(self):
        """Test PWL relaxation for non-convex y = sin(x) with Multiple Choice."""
        x_var = Variable("x", lb=0, ub=math.pi)
        x_var.breakpoints = [0, math.pi / 2, math.pi]
        y_var = Variable("y")

        x_var.pwl = MultipleChoiceMethod(x_var)
        sin_expr = SineExpression("sin1", self.model_data_relax_mc, x_var, 0, y_var)
        constraints = x_var.pwl.couple_domain_to_function_value_one_dim(
            sin_expr, approximation=False
        )
        self.assertEqual(len(constraints), 2)
        under_con = next(c for c in constraints if "under" in c.name)
        over_con = next(c for c in constraints if "over" in c.name)

        # Seg 1 [0, pi/2]: min_dev=0, max_dev≈0.2105, intercept=0
        # Seg 2 [pi/2, pi]: min_dev=0, max_dev≈0.2105, intercept=2
        over_vars = {var.name: coeff for coeff, var in over_con.variables}
        self.assertAlmostEqual(
            over_vars[lsf.var_name_pwl_multiple_choice_binary("x", 0)],
            0 + 0.2105,
            places=4,
        )
        self.assertAlmostEqual(
            over_vars[lsf.var_name_pwl_multiple_choice_binary("x", 1)],
            2 + 0.2105,
            places=4,
        )

        under_vars = {var.name: coeff for coeff, var in under_con.variables}
        self.assertAlmostEqual(
            under_vars[lsf.var_name_pwl_multiple_choice_binary("x", 0)], 0.0
        )
        self.assertAlmostEqual(
            under_vars[lsf.var_name_pwl_multiple_choice_binary("x", 1)], 2.0
        )

    def test_pwl_relaxation_one_dim_nonconvex_delta(self):
        """Test PWL relaxation for non-convex y = sin(x) with Delta Method."""
        x_var = Variable("x", lb=0, ub=math.pi)
        x_var.breakpoints = [0, math.pi / 2, math.pi]
        y_var = Variable("y")

        x_var.pwl = DeltaMethod(x_var)
        sin_expr = SineExpression("sin1", self.model_data_relax_delta, x_var, 0, y_var)
        constraints = x_var.pwl.couple_domain_to_function_value_one_dim(
            sin_expr, approximation=False
        )
        self.assertEqual(len(constraints), 2)
        under_con = next(c for c in constraints if "under" in c.name)
        over_con = next(c for c in constraints if "over" in c.name)

        # min_dev0=0, max_dev0≈0.2105. min_dev1=0, max_dev1≈0.2105.
        # Under: rhs = -f(0)-min_dev0 = 0. Coeff z0 = min1-min0 = 0.
        self.assertAlmostEqual(under_con.rhs, 0.0)
        # Over: rhs = -f(0)-max_dev0 = -0.2105. Coeff z0 = max1-max0 = 0.
        self.assertAlmostEqual(over_con.rhs, -0.2105, places=4)

    def test_pwc_approximation_multilinear(self):
        """Test the piecewise constant approximation for z = x*y."""
        x_var = Variable("x", lb=0, ub=2)
        x_var.breakpoints = [0, 1, 2]
        y_var = Variable("y", lb=0, ub=4)
        y_var.breakpoints = [0, 2, 4]
        z_var = Variable("z", lb=0, ub=8)
        z_var.breakpoints = [0, 1.5, 5, 8]

        ml_expr = MultilinearExpression(
            "ml1", self.model_data_approx_mc, [x_var, y_var], 0, z_var
        )
        ml_expr.representative_variable.is_discretized = True
        ml_expr.extract_mpip_relation(approximation=True)
        self.assertEqual(ml_expr.piecewise_constant_relation[(0, 0)], (0,))
        self.assertEqual(ml_expr.piecewise_constant_relation[(0, 1)], (0,))
        self.assertEqual(ml_expr.piecewise_constant_relation[(1, 0)], (0,))
        self.assertEqual(ml_expr.piecewise_constant_relation[(1, 1)], (1,))

    def test_pwc_relaxation_multilinear_full(self):
        """Test the generated PWC relaxation constraint."""
        x_var = Variable("x")
        x_var.breakpoints = [0, 1]
        x_var.pwl = MultipleChoiceMethod(x_var)
        y_var = Variable("y")
        y_var.breakpoints = [0, 1]
        y_var.pwl = MultipleChoiceMethod(y_var)
        z_var = Variable("z")
        z_var.breakpoints = [0, 1]
        z_var.pwl = MultipleChoiceMethod(z_var)

        ml_expr = MultilinearExpression(
            "ml1", self.model_data_relax_mc, [x_var, y_var], 0, z_var
        )
        ml_expr.piecewise_constant_relation = {(0, 0): (0,)}

        constraints = z_var.pwl.apply_pwc_relaxation(ml_expr, approximation=False)
        self.assertEqual(len(constraints), 1)
        con = constraints[0]

        # Expected constraint: -z_bp_0 + x_bp_0 + y_bp_0 <= 1
        self.assertEqual(
            con.name, lsf.con_name_pwc_multiple_choice_multilinear("ml1", [0, 0])
        )
        self.assertAlmostEqual(con.rhs, 1.0)
        vars_in_con = {v.name: c for c, v in con.variables}
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_binary("z", 0)], -1.0
        )
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_binary("x", 0)], 1.0
        )
        self.assertAlmostEqual(
            vars_in_con[lsf.var_name_pwl_multiple_choice_binary("y", 0)], 1.0
        )

    def test_pwl_handler_method_selection(self):
        """Test that PWLHandler selects the correct PWL method from settings."""
        # Test Multiple Choice
        model_mc = MockModelData(pwl_method="multiple_choice")
        x_mc = Variable("x")
        x_mc.is_discretized = True
        model_mc.add_variable(x_mc)
        handler_mc = PWLHandler(model_mc)
        handler_mc._discretize_variable_domains()
        self.assertIsInstance(model_mc.variables["x"].pwl, MultipleChoiceMethod)

        # Test Delta Method
        model_delta = MockModelData(pwl_method="delta")
        x_delta = Variable("x")
        x_delta.is_discretized = True
        model_delta.add_variable(x_delta)
        handler_delta = PWLHandler(model_delta)
        handler_delta._discretize_variable_domains()
        self.assertIsInstance(model_delta.variables["x"].pwl, DeltaMethod)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
