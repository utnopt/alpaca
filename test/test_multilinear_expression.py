# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
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
from alpaca.expressions import bilinear_expression as ble


class TestMultilinearExpression(unittest.TestCase):
    """Unit test for MultilinearExpression class."""

    def setUp(self):
        """Set up test environment."""
        self.config_dict = {"osil_file_name": "alkyl"}
        self.user_settings = s.UserSettings(self.config_dict)

        # Create mock model_data with proper mock setup
        self.model_data = MagicMock()
        self.model_data.variables = {}
        # Ensure model_data.constraints is a dictionary for testing updates
        self.model_data.constraints = {}

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

        # Create variables with occurring_in attribute
        self.var_x = var.Variable("x_1", lb=1.0, ub=5.0)
        self.var_y = var.Variable("x_2", lb=2.0, ub=7.0)
        self.var_z = var.Variable("x_3", lb=-3.0, ub=4.0)
        self.var_w = var.Variable("x_4", lb=0.5, ub=2.0)

        # Initialize occurring_in sets
        self.var_x.occurring_in = []
        self.var_y.occurring_in = []
        self.var_z.occurring_in = []
        self.var_w.occurring_in = []

        self.model_data.variables["x_1"] = self.var_x
        self.model_data.variables["x_2"] = self.var_y
        self.model_data.variables["x_3"] = self.var_z
        self.model_data.variables["x_4"] = self.var_w

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

    def test_bound_propagation_three_variables(self):
        """Test bound propagation with three variables."""
        variables = [self.var_x, self.var_y, self.var_z]
        multilinear_expr = mle.MultilinearExpression(
            "test_three_vars", self.model_data, variables, 1
        )

        multilinear_expr.propagate_variable_bounds()

        # Calculate expected bounds
        bounds = [(v.lb, v.ub) for v in variables]
        products = [reduce(mul, p) for p in itertools.product(*bounds)]
        expected_lb = min(products)  # 5 * 7 * -3 = -105
        expected_ub = max(products)  # 5 * 7 * 4 = 140

        self.assertEqual(multilinear_expr.representative_variable.lb, expected_lb)
        self.assertEqual(multilinear_expr.representative_variable.ub, expected_ub)

    def test_reformulate_to_bilinear_base_case(self):
        """Test reformulation for 3 variables (base case)."""
        variables = [self.var_x, self.var_y, self.var_z]
        multilinear_expr = mle.MultilinearExpression(
            "test_mle_3", self.model_data, variables, 1
        )

        # Mock the creation of bilinear expressions
        mock_sub_be = ble.BilinearExpression(
            "mb_test_mle_3_sub", self.model_data, (self.var_y, self.var_z), 2
        )
        mock_main_be = ble.BilinearExpression(
            "mb_test_mle_3",
            self.model_data,
            (self.var_x, mock_sub_be.representative_variable),
            1,
        )
        self.model_data.add_bilinear_expression = MagicMock(
            side_effect=[mock_sub_be, mock_main_be]
        )

        multilinear_expr.reformulate_to_bilinear_expressions()

        # Check that add_bilinear_expression was called twice
        self.assertEqual(self.model_data.add_bilinear_expression.call_count, 2)

        # Check the call for the sub-expression (y*z)
        call_1 = self.model_data.add_bilinear_expression.call_args_list[0]
        self.assertEqual(call_1.args[0].name, "mb_test_mle_3_sub")
        self.assertEqual(call_1.args[0].first_var, self.var_y)
        self.assertEqual(call_1.args[0].second_var, self.var_z)

        # Check the call for the main expression (x * sub_representative)
        call_2 = self.model_data.add_bilinear_expression.call_args_list[1]
        self.assertEqual(call_2.args[0].name, "mb_test_mle_3")
        self.assertEqual(call_2.args[0].first_var, self.var_x)
        self.assertEqual(call_2.args[0].second_var, mock_sub_be.representative_variable)

    def test_reformulate_to_bilinear_recursive_case(self):
        """Test reformulation for 4 variables (recursive case)."""
        variables = [self.var_x, self.var_y, self.var_z, self.var_w]
        multilinear_expr = mle.MultilinearExpression(
            "test_mle_4", self.model_data, variables, 1
        )

        # Mock the creation of sub-expressions
        mock_sub_me = mle.MultilinearExpression(
            "mb_test_mle_4_sub",
            self.model_data,
            [self.var_y, self.var_z, self.var_w],
            2,
        )
        mock_sub_me.reformulate_to_bilinear_expressions = (
            MagicMock()
        )  # Mock the recursive call
        mock_main_be = ble.BilinearExpression(
            "mb_test_mle_4",
            self.model_data,
            (self.var_x, mock_sub_me.representative_variable),
            1,
        )

        self.model_data.add_multilinear_expression = MagicMock(return_value=mock_sub_me)
        self.model_data.add_bilinear_expression = MagicMock(return_value=mock_main_be)

        multilinear_expr.reformulate_to_bilinear_expressions()

        # Check that a sub-multilinear expression was created for (y*z*w)
        self.model_data.add_multilinear_expression.assert_called_once()
        call_me = self.model_data.add_multilinear_expression.call_args[0][0]
        self.assertEqual(call_me.name, "mb_test_mle_4_sub")
        self.assertEqual(call_me.variables, [self.var_y, self.var_z, self.var_w])

        # Check that the recursive call was made on the sub-expression
        mock_sub_me.reformulate_to_bilinear_expressions.assert_called_once()

        # Check that the final bilinear expression was created (x * sub_representative)
        self.model_data.add_bilinear_expression.assert_called_once()
        call_be = self.model_data.add_bilinear_expression.call_args[0][0]
        self.assertEqual(call_be.name, "mb_test_mle_4")
        self.assertEqual(call_be.first_var, self.var_x)
        self.assertEqual(call_be.second_var, mock_sub_me.representative_variable)

    def test_apply_piecewise_constant_relaxation_with_approximation(
        self,
    ):
        """Test apply_piecewise_constant_relaxation with approximation."""
        # Use only 2 variables for simplicity
        variables = [self.var_x, self.var_y]
        self.var_x.breakpoints = [1.0, 3.0, 5.0]
        self.var_y.breakpoints = [2.0, 4.0, 7.0]
        self.var_x.pwl_variables_binary = [MagicMock(), MagicMock()]
        self.var_y.pwl_variables_binary = [MagicMock(), MagicMock()]

        rep_var = var.Variable("r_test_mle_approx", lb=0.0, ub=100.0)
        rep_var.breakpoints = [0.0, 10.0, 20.0, 30.0, 40.0]
        rep_var.pwl_variables_binary = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        multilinear_expr = mle.MultilinearExpression(
            "test_mle_approx", self.model_data, variables, 1, rep_var
        )
        self.model_data.constraints = {}

        multilinear_expr.apply_piecewise_constant_relaxation(approximation=True)

        # Expected mid-values for x: 2, 4
        # Expected mid-values for y: 3, 5.5
        # (ix, iy) | mid_x | mid_y | implied_val | implied_idx
        # (0, 0)   | 2     | 3     | 6           | 0
        # (0, 1)   | 2     | 5.5   | 11          | 1
        # (1, 0)   | 4     | 3     | 12          | 1
        # (1, 1)   | 4     | 5.5   | 22          | 2

        expected_relation = {
            (0, 0): (0,),
            (0, 1): (1,),
            (1, 0): (1,),
            (1, 1): (2,),
        }
        self.assertEqual(
            multilinear_expr.piecewise_constant_relation, expected_relation
        )
        self.assertEqual(len(self.model_data.constraints), 4)

        # Check one constraint, e.g., for (ix, iy) = (1, 1)
        constraint_name = "mc_test_mle_approx_1_1"
        self.assertIn(constraint_name, self.model_data.constraints)
        c_11 = self.model_data.constraints[constraint_name]
        self.assertEqual(c_11.rhs, len(variables) - 1.0)  # 2 - 1 = 1

        expected_vars = {
            (-1.0, rep_var.pwl_variables_binary[2]),  # implied_idx = 2
            (1.0, self.var_x.pwl_variables_binary[1]),
            (1.0, self.var_y.pwl_variables_binary[1]),
        }
        self.assertEqual(set(c_11.variables), expected_vars)


if __name__ == "__main__":
    unittest.main()
