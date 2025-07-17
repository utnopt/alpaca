# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code, no-member
"""
Unit tests for LinearExpression class
"""
import unittest
from unittest.mock import MagicMock

import alpaca.settings as s
from alpaca.model_data import variable as var
from alpaca.expressions import linear_expression as lie


class TestLinearExpression(unittest.TestCase):
    """Unit test for LinearExpression class."""

    def setUp(self):
        """Set up test environment."""
        self.config_dict = {"osil_file_name": "alkyl"}
        self.user_settings = s.UserSettings(self.config_dict)

        # Create mock model_data with proper mock for constraints
        self.model_data = MagicMock()
        self.model_data.variables = {}
        self.model_data.constraints = MagicMock()

        # Configure the mock add_constraint method to actually add to the dictionary
        def mock_add_constraint(constraint):
            self.model_data.constraints[constraint.name] = constraint
            return constraint

        self.model_data.add_constraint.side_effect = mock_add_constraint

        # Configure the mock add_variable method to actually add to the dictionary
        def mock_add_variable(variable):
            self.model_data.variables[variable.name] = variable
            return variable

        self.model_data.add_variable.side_effect = mock_add_variable

        # Create variables
        self.var_x = var.Variable("x_1", lb=1.0, ub=5.0)
        self.var_y = var.Variable("x_2", lb=2.0, ub=7.0)
        self.model_data.variables["x_1"] = self.var_x
        self.model_data.variables["x_2"] = self.var_y

    def test_initialization(self):
        """Test initialization of LinearExpression."""
        linear_expr = lie.LinearExpression("test_expr", self.model_data, 1)

        self.assertEqual(linear_expr.name, "test_expr")
        self.assertEqual(linear_expr.variables, [])
        self.assertEqual(linear_expr.constant, 0.0)
        self.assertEqual(linear_expr.level, 1)
        self.assertIsNotNone(linear_expr.representative_variable)
        self.assertEqual(linear_expr.representative_variable.name, "r_test_expr")
        self.assertEqual(
            linear_expr.representative_variable.lb, -s.StaticSettings.infinity
        )
        self.assertEqual(
            linear_expr.representative_variable.ub, s.StaticSettings.infinity
        )

    def test_add_constraint(self):
        """Test adding constraint from linear expression."""
        linear_expr = lie.LinearExpression("test_expr", self.model_data, 1)
        linear_expr.variables = [(2.0, self.var_x), (3.0, self.var_y)]
        linear_expr.constant = 5.0

        # Call the method
        linear_expr.add_constraint_from_linear_expression()

        # Verify constraint was added
        self.model_data.constraints.__setitem__.assert_called_once()

        # Get the constraint arguments
        args = self.model_data.constraints.__setitem__.call_args[0]
        self.assertEqual(len(args), 2)

        # Check constraint name and type
        constraint_name, constraint = args
        self.assertEqual(constraint_name, "c_test_expr")
        self.assertEqual(constraint.con_type, "==")
        self.assertEqual(constraint.rhs, -5.0)

        # Check variables in constraint
        self.assertEqual(len(constraint.variables), 3)
        self.assertEqual(constraint.variables[0], (2.0, self.var_x))
        self.assertEqual(constraint.variables[1], (3.0, self.var_y))
        self.assertEqual(
            constraint.variables[2], (-1.0, linear_expr.representative_variable)
        )

    def test_bound_propagation_positive_coefficients(self):
        """Test bound propagation with positive coefficients."""
        linear_expr = lie.LinearExpression("test_expr", self.model_data, 1)
        linear_expr.variables = [(2.0, self.var_x), (3.0, self.var_y)]
        linear_expr.constant = 5.0

        linear_expr.propagate_variable_bounds()

        # Expected bounds calculations:
        # Lower bound: 5 + (2*1) + (3*2) = 5 + 2 + 6 = 13
        # Upper bound: 5 + (2*5) + (3*7) = 5 + 10 + 21 = 36
        self.assertEqual(linear_expr.representative_variable.lb, 13.0)
        self.assertEqual(linear_expr.representative_variable.ub, 36.0)

    def test_bound_propagation_mixed_coefficients(self):
        """Test bound propagation with mixed positive/negative coefficients."""
        linear_expr = lie.LinearExpression("test_expr", self.model_data, 1)
        linear_expr.variables = [(2.0, self.var_x), (-3.0, self.var_y)]
        linear_expr.constant = 5.0

        linear_expr.propagate_variable_bounds()

        # Expected bounds calculations:
        # Lower bound: 5 + min(2*1, 2*5) + min(-3*2, -3*7) = 5 + 2 + (-21) = -14
        # Upper bound: 5 + max(2*1, 2*5) + max(-3*2, -3*7) = 5 + 10 + (-6) = 9
        self.assertEqual(linear_expr.representative_variable.lb, -14.0)
        self.assertEqual(linear_expr.representative_variable.ub, 9.0)

    def test_empty_expression(self):
        """Test behavior with empty expression (no variables)."""
        linear_expr = lie.LinearExpression("empty_expr", self.model_data, 1)

        # Should still create a constraint with just the constant
        linear_expr.constant = 10.0
        linear_expr.add_constraint_from_linear_expression()

        args = self.model_data.constraints.__setitem__.call_args[0]
        constraint_name, constraint = args

        self.assertEqual(constraint_name, "c_empty_expr")
        self.assertEqual(constraint.rhs, -10.0)
        self.assertEqual(
            len(constraint.variables), 1
        )  # Just the representative variable
        self.assertEqual(
            constraint.variables[0], (-1.0, linear_expr.representative_variable)
        )


if __name__ == "__main__":
    unittest.main()
