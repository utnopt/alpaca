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
import bisect

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

    def test_apply_piecewise_constant_approximation(
        self,
    ):  # pylint: disable=too-many-locals
        """Test apply_piecewise_constant_approximation for MultilinearExpression."""
        # Set up breakpoints for variables
        self.var_x.breakpoints = [1.0, 3.0, 5.0]  # Two intervals
        self.var_y.breakpoints = [2.0, 4.0, 7.0]  # Two intervals
        self.var_z.breakpoints = [-3.0, 0.0, 4.0]  # Two intervals

        # Mock binary variables for PWL
        self.var_x.pwl_variables_binary = [MagicMock(), MagicMock()]
        self.var_y.pwl_variables_binary = [MagicMock(), MagicMock()]
        self.var_z.pwl_variables_binary = [MagicMock(), MagicMock()]

        # Initialize representative variable with breakpoints
        rep_var = var.Variable("r_test_multilinear_pwl", lb=-100.0, ub=100.0)
        rep_var.breakpoints = [-50.0, 0.0, 10.0, 20.0, 30.0, 40.0]
        rep_var.pwl_variables_binary = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        variables = [self.var_x, self.var_y, self.var_z]
        multilinear_expr = mle.MultilinearExpression(
            "test_multilinear_pwl", self.model_data, variables, 1, rep_var
        )

        # Ensure model_data.constraints is a dictionary to check updates
        self.model_data.constraints = {}

        multilinear_expr.apply_piecewise_constant_approximation()

        # Expected mid-values:
        # x: (1+3)/2=2, (3+5)/2=4
        # y: (2+4)/2=3, (4+7)/2=5.5
        # z: (-3+0)/2=-1.5, (0+4)/2=2

        # There will be 2*2*2 = 8 combinations
        # Let's calculate some expected implied_values and indices
        # rep_var.breakpoints = [-50.0, 0.0, 10.0, 20.0, 30.0, 40.0]
        # len = 6, len - 2 = 4. So implied_index will be min(bisect_left_result, 4).

        expected_piecewise_constant_relation = {}
        expected_constraints_count = 0

        x_mid_values = [
            (self.var_x.breakpoints[i + 1] + bp) / 2
            for i, bp in enumerate(self.var_x.breakpoints[:-1])
        ]
        y_mid_values = [
            (self.var_y.breakpoints[i + 1] + bp) / 2
            for i, bp in enumerate(self.var_y.breakpoints[:-1])
        ]
        z_mid_values = [
            (self.var_z.breakpoints[i + 1] + bp) / 2
            for i, bp in enumerate(self.var_z.breakpoints[:-1])
        ]

        all_mid_values_with_indices = [
            [(val, i) for i, val in enumerate(x_mid_values)],
            [(val, i) for i, val in enumerate(y_mid_values)],
            [(val, i) for i, val in enumerate(z_mid_values)],
        ]
        for combination_with_indices in itertools.product(*all_mid_values_with_indices):
            implied_value = 1.0
            current_variable_indices = []
            for mid_value, index in combination_with_indices:
                implied_value *= mid_value
                current_variable_indices.append(index)

            implied_index = min(
                bisect.bisect_left(rep_var.breakpoints, implied_value) - 1,
                len(rep_var.breakpoints) - 2,
            )
            expected_piecewise_constant_relation[tuple(current_variable_indices)] = (
                implied_index,
            )
            expected_constraints_count += 1

            # Verify a few specific constraints
            constraint_name = (
                f"mc_{multilinear_expr.name}_"
                f"{'_'.join(map(str, current_variable_indices))}"
            )
            self.assertIn(constraint_name, self.model_data.constraints)
            c = self.model_data.constraints[constraint_name]
            self.assertEqual(c.name, constraint_name)
            self.assertEqual(c.con_type, "<=")
            self.assertEqual(c.rhs, len(variables) - 1)

            expected_vars = [(-1.0, rep_var.pwl_variables_binary[implied_index])]
            for var_idx, index_in_combination in enumerate(current_variable_indices):
                expected_vars.append(
                    (
                        1.0,
                        multilinear_expr.variables[var_idx].pwl_variables_binary[
                            index_in_combination
                        ],
                    )
                )
            self.assertEqual(set(c.variables), set(expected_vars))

        self.assertEqual(
            multilinear_expr.piecewise_constant_relation,
            expected_piecewise_constant_relation,
        )
        self.assertEqual(len(self.model_data.constraints), expected_constraints_count)


if __name__ == "__main__":
    unittest.main()
