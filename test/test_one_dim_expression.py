# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""
Comprehensive unit tests for OneDimExpression classes
"""
import unittest
import math
from unittest.mock import MagicMock

import alpaca.settings as s
from alpaca.model_data import variable as var
from alpaca.expressions import one_dim_expression as ode


class TestOneDimExpression(unittest.TestCase):
    """Unit test for OneDimExpression classes."""

    def setUp(self):
        """Set up test environment."""
        self.config_dict = {
            "osil_file_name": "alkyl",
            "pwl_method": "multiple-choice",
            "pwl_breakpoints": 5,
        }
        self.user_settings = s.UserSettings(self.config_dict)

        # Create mock model_data with proper setup
        self.model_data = MagicMock()
        self.model_data.settings = self.user_settings
        self.model_data.variables = {}
        self.model_data.constraints = MagicMock()

        # Create variable with breakpoints
        self.var_x = var.Variable("x_1", lb=1.0, ub=5.0)
        self.var_x.breakpoints = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.var_x.pwl_variables_binary = [
            var.Variable(f"x_1_bp_{i}", var_type="B") for i in range(4)
        ]
        self.var_x.pwl_variables_continuous = [
            var.Variable(f"x_1_c_{i}", var_type="C") for i in range(4)
        ]
        self.model_data.variables["x_1"] = self.var_x

        # Add occurring_in attribute if not present
        if not hasattr(self.var_x, "occurring_in"):
            self.var_x.occurring_in = []

    def test_square_expression_initialization(self):
        """Test SquareExpression initialization."""
        square_expr = ode.SquareExpression(
            "test_square", self.model_data, self.var_x, 1
        )

        self.assertEqual(square_expr.name, "test_square")
        self.assertEqual(square_expr.variable, self.var_x)
        self.assertEqual(square_expr.level, 1)
        self.assertIn("square", self.var_x.occurring_in)

        # Check representative variable
        self.assertIsNotNone(square_expr.representative_variable)
        self.assertEqual(square_expr.representative_variable.name, "r_test_square")
        self.assertEqual(
            square_expr.representative_variable.lb, -s.StaticSettings.infinity
        )
        self.assertEqual(
            square_expr.representative_variable.ub, s.StaticSettings.infinity
        )

    def test_square_expression_reference_points(self):
        """Test reference points calculation for SquareExpression."""
        square_expr = ode.SquareExpression(
            "test_square", self.model_data, self.var_x, 1
        )

        # Test multiple-choice method
        ref_points = square_expr._get_reference_points_multiple_choice()
        expected_points = [x**2 for x in self.var_x.breakpoints]
        self.assertEqual(ref_points, expected_points)

    def test_square_expression_bound_propagation(self):
        """Test bound propagation for SquareExpression."""
        square_expr = ode.SquareExpression(
            "test_square", self.model_data, self.var_x, 1
        )
        square_expr.propagate_variable_bounds()

        # Check bounds (x^2 where x ∈ [1,5])
        self.assertEqual(square_expr.representative_variable.lb, 1.0)
        self.assertEqual(square_expr.representative_variable.ub, 25.0)

        # Test with negative bounds
        var_neg = var.Variable("x_neg", lb=-2.0, ub=3.0)
        square_neg = ode.SquareExpression("test_neg", self.model_data, var_neg, 1)
        square_neg.propagate_variable_bounds()
        self.assertEqual(square_neg.representative_variable.lb, 0.0)  # x^2 min is 0
        self.assertEqual(square_neg.representative_variable.ub, 9.0)

    def test_exponential_expression_functionality(self):
        """Test ExponentialExpression functionality."""
        exp_expr = ode.ExponentialExpression("test_exp", self.model_data, self.var_x, 1)

        # Check initialization
        self.assertEqual(exp_expr.variable, self.var_x)
        self.assertIn("exp", self.var_x.occurring_in)

        # Test reference points
        ref_points = exp_expr._get_reference_points_multiple_choice()
        expected_points = [math.exp(x) for x in self.var_x.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation
        exp_expr.propagate_variable_bounds()
        self.assertAlmostEqual(exp_expr.representative_variable.lb, math.exp(1.0))
        self.assertAlmostEqual(
            exp_expr.representative_variable.ub,
            min(s.StaticSettings.infinity, math.exp(5.0)),
        )

    def test_ln_expression_functionality(self):
        """Test LnExpression functionality."""
        # Create variable with positive bounds for ln
        var_ln = var.Variable("x_ln", lb=0.1, ub=10.0)
        var_ln.breakpoints = [0.1, 1.0, 5.0, 10.0]

        ln_expr = ode.LnExpression("test_ln", self.model_data, var_ln, 1)

        # Check initialization
        self.assertEqual(ln_expr.variable, var_ln)
        self.assertIn("ln", var_ln.occurring_in)

        # Test reference points
        ref_points = ln_expr._get_reference_points_multiple_choice()
        expected_points = [math.log(x) for x in var_ln.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation
        ln_expr.propagate_variable_bounds()
        self.assertAlmostEqual(ln_expr.representative_variable.lb, math.log(0.1))
        self.assertAlmostEqual(ln_expr.representative_variable.ub, math.log(10.0))

        # Test invalid bounds
        var_invalid = var.Variable("x_invalid", lb=-1.0, ub=1.0)
        with self.assertRaises(AssertionError):
            ln_invalid = ode.LnExpression(
                "test_invalid", self.model_data, var_invalid, 1
            )
            ln_invalid.propagate_variable_bounds()

    def test_square_root_expression_functionality(self):
        """Test SquareRootExpression functionality."""
        # Create variable with non-negative bounds
        var_sqrt = var.Variable("x_sqrt", lb=0.0, ub=16.0)
        var_sqrt.breakpoints = [0.0, 4.0, 9.0, 16.0]

        sqrt_expr = ode.SquareRootExpression("test_sqrt", self.model_data, var_sqrt, 1)

        # Check initialization
        self.assertEqual(sqrt_expr.variable, var_sqrt)
        self.assertIn("sqrt", var_sqrt.occurring_in)

        # Test reference points
        ref_points = sqrt_expr._get_reference_points_multiple_choice()
        expected_points = [math.sqrt(x) for x in var_sqrt.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation
        sqrt_expr.propagate_variable_bounds()
        self.assertAlmostEqual(sqrt_expr.representative_variable.lb, 0.0)
        self.assertAlmostEqual(sqrt_expr.representative_variable.ub, 4.0)

        # Test invalid bounds
        var_invalid = var.Variable("x_invalid", lb=-1.0, ub=1.0)
        with self.assertRaises(AssertionError):
            root_invalid = ode.SquareRootExpression(
                "test_invalid", self.model_data, var_invalid, 1
            )
            root_invalid.propagate_variable_bounds()

    def test_sine_expression_functionality(self):
        """Test SineExpression functionality."""
        sine_expr = ode.SineExpression("test_sin", self.model_data, self.var_x, 1)

        # Check initialization
        self.assertEqual(sine_expr.variable, self.var_x)
        self.assertIn("sin", self.var_x.occurring_in)

        # Test reference points
        ref_points = sine_expr._get_reference_points_multiple_choice()
        expected_points = [math.sin(x) for x in self.var_x.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation (always between -1 and 1)
        sine_expr.propagate_variable_bounds()
        self.assertEqual(sine_expr.representative_variable.lb, -1.0)
        self.assertEqual(sine_expr.representative_variable.ub, 1.0)

    def test_cosine_expression_functionality(self):
        """Test CosineExpression functionality."""
        cosine_expr = ode.CosineExpression("test_cos", self.model_data, self.var_x, 1)

        # Check initialization
        self.assertEqual(cosine_expr.variable, self.var_x)
        self.assertIn("cos", self.var_x.occurring_in)

        # Test reference points
        ref_points = cosine_expr._get_reference_points_multiple_choice()
        expected_points = [math.cos(x) for x in self.var_x.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation (always between -1 and 1)
        cosine_expr.propagate_variable_bounds()
        self.assertEqual(cosine_expr.representative_variable.lb, -1.0)
        self.assertEqual(cosine_expr.representative_variable.ub, 1.0)

    def test_log_expression_functionality(self):
        """Test LogExpression (base-10) functionality."""
        # Create variable with positive bounds
        var_log = var.Variable("x_log", lb=0.1, ub=100.0)
        var_log.breakpoints = [0.1, 1.0, 10.0, 100.0]

        log_expr = ode.LogExpression("test_log", self.model_data, var_log, 1)

        # Check initialization
        self.assertEqual(log_expr.variable, var_log)
        self.assertIn("log10", var_log.occurring_in)

        # Test reference points
        ref_points = log_expr._get_reference_points_multiple_choice()
        expected_points = [math.log10(x) for x in var_log.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation
        log_expr.propagate_variable_bounds()
        self.assertAlmostEqual(log_expr.representative_variable.lb, math.log10(0.1))
        self.assertAlmostEqual(log_expr.representative_variable.ub, math.log10(100.0))

        # Test invalid bounds
        var_invalid = var.Variable("x_invalid", lb=-1.0, ub=1.0)
        with self.assertRaises(AssertionError):
            log_invalid = ode.LogExpression(
                "test_invalid", self.model_data, var_invalid, 1
            )
            log_invalid.propagate_variable_bounds()

    def test_abs_expression_functionality(self):
        """Test AbsExpression functionality."""
        # Test with positive variable
        abs_pos = ode.AbsExpression("test_abs_pos", self.model_data, self.var_x, 1)
        abs_pos.propagate_variable_bounds()
        self.assertEqual(abs_pos.representative_variable.lb, 1.0)
        self.assertEqual(abs_pos.representative_variable.ub, 5.0)

        # Test with negative variable
        var_neg = var.Variable("x_neg", lb=-3.0, ub=-1.0)
        abs_neg = ode.AbsExpression("test_abs_neg", self.model_data, var_neg, 1)
        abs_neg.propagate_variable_bounds()
        self.assertEqual(abs_neg.representative_variable.lb, 1.0)
        self.assertEqual(abs_neg.representative_variable.ub, 3.0)

        # Test with mixed variable
        var_mixed = var.Variable("x_mixed", lb=-2.0, ub=3.0)
        abs_mixed = ode.AbsExpression("test_abs_mixed", self.model_data, var_mixed, 1)
        abs_mixed.propagate_variable_bounds()
        self.assertEqual(abs_mixed.representative_variable.lb, 0.0)
        self.assertEqual(abs_mixed.representative_variable.ub, 3.0)

    def test_tanh_expression_functionality(self):
        """Test TangensHExpression functionality."""
        tanh_expr = ode.TangensHExpression("test_tanh", self.model_data, self.var_x, 1)

        # Check initialization
        self.assertEqual(tanh_expr.variable, self.var_x)
        self.assertIn("tanh", self.var_x.occurring_in)

        # Test reference points
        ref_points = tanh_expr._get_reference_points_multiple_choice()
        expected_points = [math.tanh(x) for x in self.var_x.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation
        tanh_expr.propagate_variable_bounds()
        self.assertAlmostEqual(tanh_expr.representative_variable.lb, math.tanh(1.0))
        self.assertAlmostEqual(tanh_expr.representative_variable.ub, math.tanh(5.0))

    def test_inverse_expression_functionality(self):
        """Test InverseExpression functionality."""
        # Create variable with positive bounds
        var_pos = var.Variable("x_pos", lb=1.0, ub=5.0)
        var_pos.breakpoints = [1.0, 2.0, 3.0, 4.0, 5.0]

        inv_expr = ode.InverseExpression("test_inv", self.model_data, var_pos, 1)

        # Check initialization
        self.assertEqual(inv_expr.variable, var_pos)
        self.assertIn(
            "inverse", var_pos.occurring_in
        )
        # Test reference points
        ref_points = inv_expr._get_reference_points_multiple_choice()
        expected_points = [1 / x for x in var_pos.breakpoints]
        self.assertEqual(ref_points, expected_points)

        # Test bound propagation
        inv_expr.propagate_variable_bounds()
        self.assertAlmostEqual(inv_expr.representative_variable.lb, 1 / 5.0)
        self.assertAlmostEqual(inv_expr.representative_variable.ub, 1 / 1.0)

        # Test with negative bounds
        var_neg = var.Variable("x_neg", lb=-5.0, ub=-1.0)
        inv_neg = ode.InverseExpression("test_inv_neg", self.model_data, var_neg, 1)
        inv_neg.propagate_variable_bounds()
        self.assertAlmostEqual(inv_neg.representative_variable.lb, -1 / 1.0)
        self.assertAlmostEqual(inv_neg.representative_variable.ub, -1 / 5.0)

        # Test invalid bounds (spanning zero)
        var_invalid = var.Variable("x_invalid", lb=-1.0, ub=1.0)
        with self.assertRaises(AssertionError):
            inverse_invalid = ode.InverseExpression(
                "test_invalid", self.model_data, var_invalid, 1
            )
            inverse_invalid.propagate_variable_bounds()

    def test_piecewise_linear_approximation(self):
        """Test piecewise linear approximation application."""
        square_expr = ode.SquareExpression(
            "test_square", self.model_data, self.var_x, 1
        )

        # Call the approximation method
        square_expr.apply_piecewise_linear_approximation()

        # Verify constraints were added
        self.model_data.constraints.update.assert_called_once()

        # Get the added constraints
        constraints_dict = self.model_data.constraints.update.call_args[0][0]
        self.assertIn("mc_test_square", constraints_dict)

        constraint = constraints_dict["mc_test_square"]
        self.assertEqual(constraint.con_type, "==")

        # Should have: 1 (rep var) + 4 (continuous) + 4 (binary) variables
        self.assertEqual(len(constraint.variables), 9)

    def test_different_pwl_methods(self):
        """Test behavior with different piecewise linear methods."""
        # Test multiple-choice method
        self.model_data.settings.pwl_method = "multiple-choice"
        square_mc = ode.SquareExpression("test_mc", self.model_data, self.var_x, 1)
        square_mc.apply_piecewise_linear_approximation()


if __name__ == "__main__":
    unittest.main()
