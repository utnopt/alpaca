# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
Unit tests for BilinearExpression class
"""
import unittest
from unittest.mock import MagicMock

import alpaca.settings as s
from alpaca.model_data import variable as var
from alpaca.expressions import bilinear_expression as ble


class TestBilinearExpression(unittest.TestCase):
    """Unit test for BilinearExpression class."""

    def setUp(self):
        """Set up test environment."""
        self.config_dict = {"osil_file_name": "alkyl"}
        self.user_settings = s.UserSettings(self.config_dict)

        # Create mock model_data with proper mock setup
        self.model_data = MagicMock()
        self.model_data.variables = {}

        # Create variables with occurring_in attribute
        self.var_x = var.Variable("x_1", lb=1.0, ub=5.0)
        self.var_y = var.Variable("x_2", lb=2.0, ub=7.0)
        self.var_x.occurring_in = []
        self.var_y.occurring_in = []

        self.model_data.variables["x_1"] = self.var_x
        self.model_data.variables["x_2"] = self.var_y

    def test_initialization(self):
        """Test initialization of BilinearExpression."""
        bilinear_expr = ble.BilinearExpression(
            "test_bilinear", self.model_data, (self.var_x, self.var_y), 1
        )

        # Check basic attributes
        self.assertEqual(bilinear_expr.name, "test_bilinear")
        self.assertEqual(bilinear_expr.first_var, self.var_x)
        self.assertEqual(bilinear_expr.second_var, self.var_y)
        self.assertEqual(bilinear_expr.level, 1)

        # Check representative variable
        self.assertIsNotNone(bilinear_expr.representative_variable)
        self.assertEqual(bilinear_expr.representative_variable.name, "r_test_bilinear")
        self.assertEqual(
            bilinear_expr.representative_variable.lb, -s.StaticSettings.infinity
        )
        self.assertEqual(
            bilinear_expr.representative_variable.ub, s.StaticSettings.infinity
        )

        # Check nonlinearity tracking
        self.assertIn("bilinear", self.var_x.occurring_in)
        self.assertIn("bilinear", self.var_y.occurring_in)
        self.assertIn("bilinear", bilinear_expr.representative_variable.occurring_in)

    def test_bound_propagation_positive_variables(self):
        """Test bound propagation with positive variable bounds."""
        bilinear_expr = ble.BilinearExpression(
            "test_bilinear", self.model_data, (self.var_x, self.var_y), 1
        )

        # Call bound propagation
        bilinear_expr.propagate_variable_bounds()

        # Calculate expected bounds
        combinations = [
            self.var_x.lb * self.var_y.lb,
            self.var_x.lb * self.var_y.ub,
            self.var_x.ub * self.var_y.lb,
            self.var_x.ub * self.var_y.ub,
        ]
        expected_lb = min(combinations)  # 1*2 = 2
        expected_ub = max(combinations)  # 5*7 = 35

        # Check both bounds are updated
        self.assertEqual(bilinear_expr.representative_variable.lb, expected_lb)
        self.assertEqual(bilinear_expr.representative_variable.ub, expected_ub)

    def test_bound_propagation_mixed_bounds(self):
        """Test bound propagation with variables that have negative bounds."""
        # Create variables with negative bounds
        var_a = var.Variable("a", lb=-3.0, ub=-1.0)
        var_b = var.Variable("b", lb=2.0, ub=4.0)
        var_a.occurring_in = []
        var_b.occurring_in = []
        self.model_data.variables["a"] = var_a
        self.model_data.variables["b"] = var_b

        bilinear_expr = ble.BilinearExpression(
            "test_mixed", self.model_data, (var_a, var_b), 1
        )

        bilinear_expr.propagate_variable_bounds()

        # Calculate expected bounds for (-3.0, -1.0) × (2.0, 4.0)
        combinations = [
            -3.0 * 2.0,  # -6
            -3.0 * 4.0,  # -12
            -1.0 * 2.0,  # -2
            -1.0 * 4.0,  # -4
        ]
        expected_lb = min(combinations)  # -12
        expected_ub = max(combinations)  # -2

        self.assertEqual(bilinear_expr.representative_variable.lb, expected_lb)
        self.assertEqual(bilinear_expr.representative_variable.ub, expected_ub)

    def test_bound_propagation_zero_included(self):
        """Test bound propagation when zero is included in variable bounds."""
        var_a = var.Variable("a", lb=-2.0, ub=3.0)
        var_b = var.Variable("b", lb=-1.0, ub=4.0)
        var_a.occurring_in = []
        var_b.occurring_in = []
        self.model_data.variables["a"] = var_a
        self.model_data.variables["b"] = var_b

        bilinear_expr = ble.BilinearExpression(
            "test_zero", self.model_data, (var_a, var_b), 1
        )

        bilinear_expr.propagate_variable_bounds()

        # Calculate expected bounds for (-2.0, 3.0) × (-1.0, 4.0)
        combinations = [
            -2.0 * -1.0,  # 2
            -2.0 * 4.0,  # -8
            3.0 * -1.0,  # -3
            3.0 * 4.0,  # 12
        ]
        expected_lb = min(combinations)  # -8
        expected_ub = max(combinations)  # 12

        self.assertEqual(bilinear_expr.representative_variable.lb, expected_lb)
        self.assertEqual(bilinear_expr.representative_variable.ub, expected_ub)

    def test_initialization_with_invalid_variables(self):
        """Test initialization with invalid variable inputs."""
        with self.assertRaises(ValueError):
            ble.BilinearExpression("invalid", self.model_data, (self.var_x,), 1)

        with self.assertRaises(ValueError):
            ble.BilinearExpression("invalid", self.model_data, "not_a_tuple", 1)


if __name__ == "__main__":
    unittest.main()
