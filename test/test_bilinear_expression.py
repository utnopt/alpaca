# -*- coding: utf-8 -*-
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

        self.assertEqual(bilinear_expr.name, "test_bilinear")
        self.assertEqual(bilinear_expr.variables[0], self.var_x)
        self.assertEqual(bilinear_expr.variables[1], self.var_y)
        self.assertEqual(bilinear_expr.level, 1)
        self.assertIsNotNone(bilinear_expr.representative_variable)
        self.assertEqual(bilinear_expr.representative_variable.name, "r_test_bilinear")
        self.assertIn("multilinear2", self.var_x.occurring_in)
        self.assertIn("multilinear2", self.var_y.occurring_in)

    def test_bound_propagation(self):
        """Test bound propagation with various variable bounds."""
        test_cases = [
            # Case 1: All positive bounds
            {"name": "pos", "x": (1.0, 5.0), "y": (2.0, 7.0), "exp_z": (2.0, 35.0)},
            # Case 2: Mixed positive and negative bounds
            {"name": "mix", "x": (-3.0, -1.0), "y": (2.0, 4.0), "exp_z": (-12.0, -2.0)},
            # Case 3: Bounds including zero
            {"name": "zero", "x": (-2.0, 3.0), "y": (-1.0, 4.0), "exp_z": (-8.0, 12.0)},
        ]

        for case in test_cases:
            with self.subTest(case["name"]):
                var_a = var.Variable("a", lb=case["x"][0], ub=case["x"][1])
                var_b = var.Variable("b", lb=case["y"][0], ub=case["y"][1])
                expr = ble.BilinearExpression(
                    f"test_{case['name']}", self.model_data, [var_a, var_b], 1
                )
                expr.propagate_variable_bounds()
                self.assertAlmostEqual(
                    expr.representative_variable.lb, case["exp_z"][0]
                )
                self.assertAlmostEqual(
                    expr.representative_variable.ub, case["exp_z"][1]
                )

    def test_add_mccormick_envelope_minimal_example(self):
        """Test adding McCormick envelope with a simple, verifiable example."""
        # Arrange a minimal example: x in [1, 2], y in [3, 5]
        x = var.Variable("x", lb=1.0, ub=2.0)
        y = var.Variable("y", lb=3.0, ub=5.0)
        bilinear_expr = ble.BilinearExpression(
            "test_mccormick", self.model_data, [x, y], 1
        )
        z = bilinear_expr.representative_variable
        self.model_data.constraints.clear()

        # Act
        bilinear_expr.add_mccormick_envelope()

        # Assert: Check that the four McCormick constraints are created correctly
        self.assertEqual(len(self.model_data.constraints), 4)

        # Expected constraints from z = x*y, x in [1,2], y in [3,5]
        # mc1: z >= x.lb*y + y.ub*x - x.lb*y.ub  => z >= 1*y + 5*x - 5
        # mc2: z >= x.ub*y + y.lb*x - x.ub*y.lb  => z >= 2*y + 3*x - 6
        # mc3: z <= x.lb*y + y.lb*x - x.lb*y.lb  => z <= 1*y + 3*x - 3
        # mc4: z <= x.ub*y + y.ub*x - x.ub*y.ub  => z <= 2*y + 5*x - 10
        expected = {
            f"mc1_{bilinear_expr.name}": {
                "vars": {(1, z), (-1, y), (-5, x)},
                "rhs": -5,
            },
            f"mc2_{bilinear_expr.name}": {
                "vars": {(1, z), (-2, y), (-3, x)},
                "rhs": -6,
            },
            f"mc3_{bilinear_expr.name}": {
                "vars": {(1, z), (-1, y), (-3, x)},
                "rhs": -3,
            },
            f"mc4_{bilinear_expr.name}": {
                "vars": {(1, z), (-2, y), (-5, x)},
                "rhs": -10,
            },
        }

        for name, const in self.model_data.constraints.items():
            self.assertIn(name, expected)
            self.assertEqual(set(const.variables), expected[name]["vars"])
            self.assertAlmostEqual(const.rhs, expected[name]["rhs"])

    def _setup_reformulation_mocks(self):
        """Helper to set up mocks for reformulation tests."""
        created_linear_exprs, created_one_dim_exprs = [], []

        def le_factory(*_args, **_kwargs):
            mock = MagicMock(
                spec=lie.LinearExpression,
                variables=[],
                representative_variable=MagicMock(spec=var.Variable),
            )
            created_linear_exprs.append(mock)
            return mock

        def ode_factory(*_args, **_kwargs):
            mock = MagicMock(
                spec=ode.SquareExpression,
                representative_variable=MagicMock(spec=var.Variable),
            )
            created_one_dim_exprs.append(mock)
            return mock

        self.model_data.add_linear_expression.side_effect = le_factory
        self.model_data.add_one_dim_expression.side_effect = ode_factory
        return created_linear_exprs, created_one_dim_exprs

    def test_reformulate_to_sum_of_squares(self):
        """Test reformulation of a continuous bilinear expression to sum of squares."""
        bilinear_expr = ble.BilinearExpression(
            "test_bilinear", self.model_data, [self.var_x, self.var_y], 1
        )
        created_linear_exprs, created_one_dim_exprs = self._setup_reformulation_mocks()

        # Act
        bilinear_expr.reformulate_to_sum_of_squares()

        # Assert
        self.assertEqual(self.model_data.add_linear_expression.call_count, 2)
        self.assertEqual(self.model_data.add_one_dim_expression.call_count, 3)

        self.assertEqual(len(created_linear_exprs), 2)
        master_le, sub_le = created_linear_exprs[0], created_linear_exprs[1]
        self.assertEqual(len(created_one_dim_exprs), 3)
        sq_x, sq_y, sq_h = (
            created_one_dim_exprs[0],
            created_one_dim_exprs[1],
            created_one_dim_exprs[2],
        )

        # Assert sub-expression p = x - y
        self.assertEqual(set(sub_le.variables), {(1.0, self.var_x), (-1.0, self.var_y)})

        # Assert master-expression z = 0.5 * (x^2 + y^2 - p^2)
        expected_master_vars = {
            (0.5, sq_x.representative_variable),
            (0.5, sq_y.representative_variable),
            (-0.5, sq_h.representative_variable),
        }
        self.assertEqual(set(master_le.variables), expected_master_vars)

    def test_reformulate_to_sum_of_squares_with_binary_var(self):
        """Test sum of squares reformulation where one variable is binary."""
        var_b = var.Variable("b", var_type="B", lb=0.0, ub=1.0)
        bilinear_expr = ble.BilinearExpression(
            "test_binary", self.model_data, [var_b, self.var_y], 1
        )
        created_linear_exprs, created_one_dim_exprs = self._setup_reformulation_mocks()

        # Act
        bilinear_expr.reformulate_to_sum_of_squares()

        # Assert
        self.assertEqual(self.model_data.add_linear_expression.call_count, 2)
        self.assertEqual(self.model_data.add_one_dim_expression.call_count, 2)

        self.assertEqual(len(created_linear_exprs), 2)
        master_le, sub_le = created_linear_exprs[0], created_linear_exprs[1]
        self.assertEqual(len(created_one_dim_exprs), 2)
        sq_y, sq_h = created_one_dim_exprs[0], created_one_dim_exprs[1]

        # Assert sub-expression p = b - y
        self.assertEqual(set(sub_le.variables), {(1.0, var_b), (-1.0, self.var_y)})

        # Assert master-expression z = 0.5 * (b + y^2 - p^2)
        expected_master_vars = {
            (0.5, var_b),  # Direct use of binary variable
            (0.5, sq_y.representative_variable),
            (-0.5, sq_h.representative_variable),
        }
        self.assertEqual(set(master_le.variables), expected_master_vars)

    def test_apply_piecewise_constant_relaxation(self):
        """Test apply_piecewise_constant_relaxation with and without approximation."""
        self.var_x.breakpoints = [1.0, 3.0, 5.0]
        self.var_y.breakpoints = [2.0, 4.0, 7.0]
        self.var_x.pwl_variables_binary = [MagicMock(), MagicMock()]
        self.var_y.pwl_variables_binary = [MagicMock(), MagicMock()]
        rep_var = var.Variable("r_test_pwcr", lb=0.0, ub=100.0)
        rep_var.breakpoints = [0.0, 10.0, 20.0, 30.0, 40.0]
        rep_var.is_discretized = True
        rep_var.pwl_variables_binary = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        test_cases = [
            {
                "name": "no_approximation",
                "approximation": False,
                "expected": {
                    (0, 0): (0, 1),
                    (0, 1): (0, 1, 2),
                    (1, 0): (0, 1),
                    (1, 1): (1, 2, 3),
                },
            },
            {
                "name": "with_approximation",
                "approximation": True,
                "expected": {
                    (0, 0): (0,),
                    (0, 1): (1,),
                    (1, 0): (1,),
                    (1, 1): (2,),
                },
            },
        ]
        for case in test_cases:
            with self.subTest(case["name"]):
                expr = ble.BilinearExpression(
                    f"test_{case['name']}",
                    self.model_data,
                    [self.var_x, self.var_y],
                    1,
                    rep_var,
                )
                self.model_data.constraints = {}
                expr.apply_piecewise_constant_relaxation(case["approximation"])
                expr.extract_mpip_relation(case["approximation"])
                self.assertEqual(expr.piecewise_constant_relation, case["expected"])
                self.assertEqual(len(self.model_data.constraints), 4)


if __name__ == "__main__":
    unittest.main()
