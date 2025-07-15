# -*- coding: utf-8 -*-
"""
Unit tests for MultilinearExpression class
"""
import unittest
from unittest.mock import MagicMock
import itertools
from operator import mul
from functools import reduce

import alpaca.settings as s
from alpaca.model_data import variable as var
from alpaca.expressions import multilinear_expression as mle


class TestMultilinearExpression(unittest.TestCase):
    """Unit test for MultilinearExpression class."""

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
        self.var_z = var.Variable("x_3", lb=-3.0, ub=4.0)

        # Initialize occurring_in sets
        self.var_x.occurring_in = []
        self.var_y.occurring_in = []
        self.var_z.occurring_in = []

        self.model_data.variables["x_1"] = self.var_x
        self.model_data.variables["x_2"] = self.var_y
        self.model_data.variables["x_3"] = self.var_z

    def test_initialization(self):
        """Test initialization of MultilinearExpression."""
        variables = [self.var_x, self.var_y, self.var_z]
        multilinear_expr = mle.MultilinearExpression(
            "test_multilinear", self.model_data, variables, 1
        )

        # Check basic attributes
        self.assertEqual(multilinear_expr.name, "test_multilinear")
        self.assertEqual(multilinear_expr.variables, variables)
        self.assertEqual(multilinear_expr.level, 1)

        # Check representative variable
        self.assertIsNotNone(multilinear_expr.representative_variable)
        self.assertEqual(
            multilinear_expr.representative_variable.name, "r_test_multilinear"
        )
        self.assertEqual(
            multilinear_expr.representative_variable.lb, -s.StaticSettings.infinity
        )
        self.assertEqual(
            multilinear_expr.representative_variable.ub, s.StaticSettings.infinity
        )

        # Check nonlinearity tracking
        for variable in variables:
            self.assertIn("multilinear", variable.occurring_in)
        self.assertIn(
            "multilinear", multilinear_expr.representative_variable.occurring_in
        )

    def test_bound_propagation_positive_variables(self):
        """Test bound propagation with positive variable bounds."""
        variables = [
            self.var_x,
            self.var_y,
        ]  # Just use two positive variables for simplicity
        multilinear_expr = mle.MultilinearExpression(
            "test_positive", self.model_data, variables, 1
        )

        # Call bound propagation
        multilinear_expr.propagate_variable_bounds()

        # Calculate expected bounds (product of all combinations of bounds)
        bounds = [(variable.lb, variable.ub) for variable in variables]
        products = list(reduce(mul, product) for product in itertools.product(*bounds))
        expected_lb = min(products)
        expected_ub = max(products)

        # Check both bounds are updated
        self.assertEqual(multilinear_expr.representative_variable.lb, expected_lb)
        self.assertEqual(multilinear_expr.representative_variable.ub, expected_ub)

    def test_bound_propagation_mixed_bounds(self):
        """Test bound propagation with variables that have mixed bounds."""
        variables = [self.var_x, self.var_z]  # One positive, one with negative bounds
        multilinear_expr = mle.MultilinearExpression(
            "test_mixed", self.model_data, variables, 1
        )

        multilinear_expr.propagate_variable_bounds()

        # Calculate expected bounds
        bounds = [(variable.lb, variable.ub) for variable in variables]
        products = list(reduce(mul, product) for product in itertools.product(*bounds))
        expected_lb = min(products)
        expected_ub = max(products)

        self.assertEqual(multilinear_expr.representative_variable.lb, expected_lb)
        self.assertEqual(multilinear_expr.representative_variable.ub, expected_ub)

    def test_bound_propagation_three_variables(self):
        """Test bound propagation with three variables."""
        variables = [self.var_x, self.var_y, self.var_z]
        multilinear_expr = mle.MultilinearExpression(
            "test_three_vars", self.model_data, variables, 1
        )

        multilinear_expr.propagate_variable_bounds()

        # Calculate expected bounds
        bounds = [(variable.lb, variable.ub) for variable in variables]
        products = list(reduce(mul, product) for product in itertools.product(*bounds))
        expected_lb = max(-s.StaticSettings.infinity, min(products))
        expected_ub = min(s.StaticSettings.infinity, max(products))

        self.assertEqual(multilinear_expr.representative_variable.lb, expected_lb)
        self.assertEqual(multilinear_expr.representative_variable.ub, expected_ub)

    def test_initialization_with_single_variable(self):
        """Test initialization with single variable (should work)."""
        multilinear_expr = mle.MultilinearExpression(
            "test_single", self.model_data, [self.var_x], 1
        )

        self.assertEqual(len(multilinear_expr.variables), 1)
        self.assertEqual(multilinear_expr.variables[0], self.var_x)


if __name__ == "__main__":
    unittest.main()
