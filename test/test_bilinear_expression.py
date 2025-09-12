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
from alpaca.expressions import linear_expression as lie
from alpaca.expressions import one_dim_expression as ode


class TestBilinearExpression(unittest.TestCase):
    """Unit test for BilinearExpression class."""

    def setUp(self):
        """Set up test environment."""
        self.config_dict = {
            "osil_file_name": "alkyl",
            "reformulate_multilinear": False,
        }
        self.user_settings = s.UserSettings(self.config_dict)

        # Create mock model_data with proper mock setup
        self.model_data = MagicMock()
        self.model_data.variables = {}
        # Ensure model_data.constraints is a dictionary for testing updates
        self.model_data.constraints = {}
        self.model_data.expressions = MagicMock()
        self.model_data.expressions.linear_expressions = {}
        self.model_data.expressions.one_dim_expressions = {}

        # Configure the mock add_constraint method to actually add to the dictionary
        def mock_add_constraint(constraint):
            self.model_data.constraints[constraint.name] = constraint
            return constraint

        def mock_add_variable(variable):
            self.model_data.variables[variable.name] = variable
            return variable

        self.model_data.add_constraint.side_effect = mock_add_constraint
        self.model_data.add_variable.side_effect = mock_add_variable

        self.var_x = var.Variable("x_1", lb=1.0, ub=5.0)
        self.var_y = var.Variable("x_2", lb=2.0, ub=7.0)
        self.var_x.occurring_in = []
        self.var_y.occurring_in = []

        self.model_data.variables["x_1"] = self.var_x
        self.model_data.variables["x_2"] = self.var_y

    def test_initialization(self):
        """Test initialization of BilinearExpression."""
        bilinear_expr = ble.BilinearExpression(
            "test_bilinear",
            self.model_data,
            [self.var_x, self.var_y],
            1,
        )

        # Check basic attributes
        self.assertEqual(bilinear_expr.name, "test_bilinear")
        self.assertEqual(bilinear_expr.variables[0], self.var_x)
        self.assertEqual(bilinear_expr.variables[1], self.var_y)
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
        self.assertIn("multilinear2", self.var_x.occurring_in)
        self.assertIn("multilinear2", self.var_y.occurring_in)

    def test_bound_propagation_positive_variables(self):
        """Test bound propagation with positive variable bounds."""
        bilinear_expr = ble.BilinearExpression(
            "test_bilinear",
            self.model_data,
            [self.var_x, self.var_y],
            1,
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
            "test_mixed", self.model_data, [var_a, var_b], 1
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
            "test_zero", self.model_data, [var_a, var_b], 1
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

    def test_apply_piecewise_constant_relaxation(self):
        """Test apply_piecewise_constant_relaxation for BilinearExpression."""
        # Set up breakpoints for variables
        self.var_x.breakpoints = [1.0, 3.0, 5.0]  # Two intervals
        self.var_y.breakpoints = [2.0, 4.0, 7.0]  # Two intervals

        # Mock binary variables for PWL
        self.var_x.pwl_variables_binary = [MagicMock(), MagicMock()]
        self.var_y.pwl_variables_binary = [MagicMock(), MagicMock()]

        # Initialize representative variable with breakpoints
        rep_var = var.Variable("r_test_bilinear_pwl", lb=0.0, ub=100.0)
        rep_var.breakpoints = [0.0, 10.0, 20.0, 30.0, 40.0]
        rep_var.is_discretized = True
        rep_var.pwl_variables_binary = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        bilinear_expr = ble.BilinearExpression(
            "test_bilinear_pwl",
            self.model_data,
            [self.var_x, self.var_y],
            1,
            rep_var,
        )
        self.model_data.constraints = {}

        # Test without approximation
        bilinear_expr.apply_piecewise_constant_relaxation(approximation=False)
        bilinear_expr.extract_mpip_relation(approximation=False)

        # Expected implied bounds and indices
        # (i, j) | x_bounds  | y_bounds  | implied_bounds | implied_indices
        # (0, 0) | [1, 3]    | [2, 4]    | [2, 12]        | (0, 1) -> range(0, 2) -> (0, 1)
        # (0, 1) | [1, 3]    | [4, 7]    | [4, 21]        | (0, 2) -> range(0, 3) -> (0, 1, 2)
        # (1, 0) | [3, 5]    | [2, 4]    | [6, 20]        | (0, 1) -> range(0, 2) -> (0, 1)
        # (1, 1) | [3, 5]    | [4, 7]    | [12, 35]       | (1, 3) -> range(1, 4) -> (1, 2, 3)

        expected_relation = {
            (0, 0): (0, 1),
            (0, 1): (0, 1, 2),
            (1, 0): (0, 1),
            (1, 1): (1, 2, 3),
        }
        self.assertEqual(bilinear_expr.piecewise_constant_relation, expected_relation)
        self.assertEqual(len(self.model_data.constraints), 4)

    def test_apply_piecewise_constant_relaxation_with_approximation(self):
        """Test apply_piecewise_constant_relaxation with approximation."""
        # Set up breakpoints for variables
        self.var_x.breakpoints = [1.0, 3.0, 5.0]
        self.var_y.breakpoints = [2.0, 4.0, 7.0]

        # Mock binary variables for PWL
        self.var_x.pwl_variables_binary = [MagicMock(), MagicMock()]
        self.var_y.pwl_variables_binary = [MagicMock(), MagicMock()]

        # Initialize representative variable with breakpoints
        rep_var = var.Variable("r_test_bilinear_approx", lb=0.0, ub=100.0)
        rep_var.breakpoints = [0.0, 10.0, 20.0, 30.0, 40.0]
        rep_var.is_discretized = True
        rep_var.pwl_variables_binary = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        bilinear_expr = ble.BilinearExpression(
            "test_bilinear_approx",
            self.model_data,
            [self.var_x, self.var_y],
            1,
            rep_var,
        )
        self.model_data.constraints = {}

        # Test with approximation
        bilinear_expr.apply_piecewise_constant_relaxation(approximation=True)
        bilinear_expr.extract_mpip_relation(approximation=True)

        # Expected mid-values for x: (1+3)/2=2, (3+5)/2=4
        # Expected mid-values for y: (2+4)/2=3, (4+7)/2=5.5
        # (i, j) | mid_x | mid_y | implied_value | implied_index
        # (0, 0) | 2     | 3     | 6             | 0
        # (0, 1) | 2     | 5.5   | 11            | 1
        # (1, 0) | 4     | 3     | 12            | 1
        # (1, 1) | 4     | 5.5   | 22            | 2

        expected_relation = {
            (0, 0): (0,),
            (0, 1): (1,),
            (1, 0): (1,),
            (1, 1): (2,),
        }
        self.assertEqual(bilinear_expr.piecewise_constant_relation, expected_relation)
        self.assertEqual(len(self.model_data.constraints), 4)

    def test_reformulate_to_sum_of_squares(self):
        """Test reformulation of bilinear expression to sum of squares."""
        bilinear_expr = ble.BilinearExpression(
            "test_bilinear", self.model_data, [self.var_x, self.var_y], 1
        )

        # Mock the expression creation methods
        mock_master_le = lie.LinearExpression(
            "le_test_bilinear_master", self.model_data, 1
        )
        mock_sub_le = lie.LinearExpression("le_test_bilinear_sub", self.model_data, 3)
        mock_square_x = ode.SquareExpression(
            "fvs_test_bilinear", self.model_data, self.var_x, 2
        )
        mock_square_y = ode.SquareExpression(
            "svs_test_bilinear", self.model_data, self.var_y, 2
        )
        mock_square_h = ode.SquareExpression(
            "hvs_test_bilinear", self.model_data, mock_sub_le.representative_variable, 2
        )

        self.model_data.add_linear_expression = MagicMock(
            side_effect=[mock_master_le, mock_sub_le]
        )
        self.model_data.add_one_dim_expression = MagicMock(
            side_effect=[mock_square_x, mock_square_y, mock_square_h]
        )

        # Call the method
        bilinear_expr.reformulate_to_sum_of_squares()

        # Check calls to create expressions
        self.assertEqual(self.model_data.add_linear_expression.call_count, 2)
        self.assertEqual(self.model_data.add_one_dim_expression.call_count, 3)

        # Check sub linear expression: p = x - y
        self.assertIn((1.0, self.var_x), mock_sub_le.variables)
        self.assertIn((-1.0, self.var_y), mock_sub_le.variables)

        # Check master linear expression: z = 0.5 * (x^2 + y^2 - p^2)
        self.assertIn(
            (0.5, mock_square_x.representative_variable), mock_master_le.variables
        )
        self.assertIn(
            (0.5, mock_square_y.representative_variable), mock_master_le.variables
        )
        self.assertIn(
            (-0.5, mock_square_h.representative_variable), mock_master_le.variables
        )


if __name__ == "__main__":
    unittest.main()
