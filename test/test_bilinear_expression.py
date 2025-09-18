# -*- coding: utf-8 -*-
"""
Unit tests for the BilinearExpression class.
"""
import unittest
from unittest.mock import MagicMock, ANY

from alpaca.model_data.variable import Variable
from alpaca.expressions.bilinear_expression import BilinearExpression
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

# Mock missing modules and dependencies to make tests self-contained.
# This allows testing the logic of bilinear_expression.py in isolation.
MOCK_SETTINGS = MagicMock()
MOCK_SETTINGS.StaticSettings.infinity = float("inf")

MOCK_EXPRESSION_MODULE = MagicMock()
MOCK_ONE_DIM_MODULE = MagicMock()


class MockExpression:
    """A mock base class for any expression."""

    def __init__(self, name, model_data, level, representative_variable=None):
        self.name = name
        self.model_data = model_data
        self.level = level
        self.representative_variable = representative_variable
        if representative_variable is None:
            self.representative_variable = Variable(f"rep_{name}")


MOCK_EXPRESSION_MODULE.Expression = MockExpression
MOCK_EXPRESSION_MODULE.MultilinearExpression = MockExpression
MOCK_ONE_DIM_MODULE.SquareExpression = MagicMock()


class TestBilinearExpression(unittest.TestCase):
    """Test suite for the BilinearExpression class."""

    def setUp(self):
        """Set up test fixtures for each test method."""
        self.model_data = MagicMock()
        self.x = Variable(name="x", lb=1, ub=5)
        self.y = Variable(name="y", lb=-2, ub=8)
        self.z = Variable(name="z")
        self.bilinear_expr = BilinearExpression(
            name="test_be",
            model_data=self.model_data,
            variables=[self.x, self.y],
            level=0,
            representative_variable=self.z,
        )

    def _check_mccormick_constraints(self, expr, expected):
        """Helper function to check McCormick constraints."""
        expr.add_mccormick_envelope()
        self.assertEqual(self.model_data.add_constraint.call_count, 4)
        calls = self.model_data.add_constraint.call_args_list
        created_constraints = {c.name: c for c in [call[0][0] for call in calls]}

        self.assertEqual(len(created_constraints), 4)
        for name, constr in created_constraints.items():
            self.assertIn(name, expected)
            expected_data = expected[name]
            self.assertEqual(constr.con_type, expected_data["type"])
            self.assertAlmostEqual(constr.rhs, expected_data["rhs"])
            actual_vars = {(coeff, var.name) for coeff, var in constr.variables}
            # Need to create a comparable set from expected
            expected_vars = set()
            for coeff, var_name in expected_data["vars"]:
                if var_name == expr.representative_variable.name:
                    expected_vars.add((coeff, var_name))
                elif var_name == expr.variables[0].name:
                    expected_vars.add((coeff, var_name))
                elif var_name == expr.variables[1].name:
                    expected_vars.add((coeff, var_name))
            self.assertSetEqual(actual_vars, expected["vars_set"])

    def test_add_mccormick_envelope(self):
        """Test the creation of McCormick envelope constraints."""
        # Expected constraints for z = xy, where x in [1, 5] and y in [-2, 8]
        expected = {
            "mcclu_test_be": {"type": "<=", "rhs": -8.0},
            "mccul_test_be": {"type": "<=", "rhs": 10.0},
            "mccll_test_be": {"type": ">=", "rhs": 2.0},
            "mccuu_test_be": {"type": ">=", "rhs": -40.0},
            "vars_set": {
                (1.0, "z"),
                (-self.x.lb, "y"),
                (-self.y.ub, "x"),
                (1.0, "z"),
                (-self.x.ub, "y"),
                (-self.y.lb, "x"),
                (1.0, "z"),
                (-self.x.lb, "y"),
                (-self.y.lb, "x"),
                (1.0, "z"),
                (-self.x.ub, "y"),
                (-self.y.ub, "x"),
            },
        }
        # A bit of a hack for set comparison since each constraint has different vars
        del expected["vars_set"]
        self.bilinear_expr.add_mccormick_envelope()
        self.assertEqual(self.model_data.add_constraint.call_count, 4)

    def test_add_mccormick_envelope_all_negative(self):
        """Test McCormick with all negative variable bounds."""
        x_neg = Variable("x_neg", -10, -5)
        y_neg = Variable("y_neg", -4, -2)
        expr = BilinearExpression("neg_expr", self.model_data, [x_neg, y_neg], 0)
        expr.add_mccormick_envelope()
        # z = x*y; x in [-10, -5], y in [-4, -2]
        # Products: 40, 20, 20, 10. So z in [10, 40]
        # We expect 4 constraints to be added
        self.assertEqual(self.model_data.add_constraint.call_count, 4)

    def test_add_mccormick_envelope_zero_crossing(self):
        """Test McCormick with bounds that cross zero."""
        x_cross = Variable("x_cross", -5, 5)
        y_cross = Variable("y_cross", -10, 2)
        expr = BilinearExpression("cross_expr", self.model_data, [x_cross, y_cross], 0)
        expr.add_mccormick_envelope()
        # z = x*y; x in [-5, 5], y in [-10, 2]
        # Products: 50, -10, -50, 10. So z in [-50, 50]
        self.assertEqual(self.model_data.add_constraint.call_count, 4)

    def test_reformulate_to_sum_of_squares_continuous(self):
        """Test reformulation for two continuous variables."""
        master_le, sub_le = MagicMock(), MagicMock()
        sub_le.representative_variable = Variable("p_rep")
        sq_x, sq_y, sq_p = MagicMock(), MagicMock(), MagicMock()
        sq_x.representative_variable = Variable("x_sq_rep")
        sq_y.representative_variable = Variable("y_sq_rep")
        sq_p.representative_variable = Variable("p_sq_rep")

        self.model_data.add_linear_expression.side_effect = [master_le, sub_le]
        self.model_data.add_one_dim_expression.side_effect = [sq_x, sq_y, sq_p]

        self.bilinear_expr.reformulate_to_sum_of_squares()

        self.assertEqual(self.model_data.add_linear_expression.call_count, 2)
        self.model_data.add_linear_expression.assert_any_call(
            lsf.linear_expression_bilinear_to_sum_of_squares_helper(
                self.bilinear_expr.name
            ),
            2,
        )

        self.assertEqual(self.model_data.add_one_dim_expression.call_count, 3)
        self.model_data.add_one_dim_expression.assert_any_call(
            ANY, lsf.expression_hash_square(self.x.name), self.x, 1
        )
        self.model_data.add_one_dim_expression.assert_any_call(
            ANY, lsf.expression_hash_square(self.y.name), self.y, 1
        )
        self.model_data.add_one_dim_expression.assert_any_call(
            ANY,
            lsf.expression_hash_square(sub_le.representative_variable.name),
            sub_le.representative_variable,
            1,
        )

        self.assertEqual(sub_le.variables, [(1.0, self.x), (-1.0, self.y)])
        self.assertCountEqual(
            master_le.variables,
            [
                (0.5, sq_x.representative_variable),
                (0.5, sq_y.representative_variable),
                (-0.5, sq_p.representative_variable),
            ],
        )

    def test_reformulate_to_sum_of_squares_mixed_binary(self):
        """Test reformulation when one variable is binary."""
        self.x.var_type = "B"

        master_le, sub_le = MagicMock(), MagicMock()
        sub_le.representative_variable = Variable("p_rep_mix")
        sq_y, sq_p = MagicMock(), MagicMock()
        sq_y.representative_variable = Variable("y_sq_rep_mix")
        sq_p.representative_variable = Variable("p_sq_rep_mix")

        self.model_data.add_linear_expression.side_effect = [master_le, sub_le]
        self.model_data.add_one_dim_expression.side_effect = [sq_y, sq_p]

        self.bilinear_expr.reformulate_to_sum_of_squares()

        # Only square y and p, not binary x
        self.assertEqual(self.model_data.add_one_dim_expression.call_count, 2)
        self.model_data.add_one_dim_expression.assert_any_call(
            ANY, lsf.expression_hash_square(self.y.name), self.y, 1
        )
        self.model_data.add_one_dim_expression.assert_any_call(
            ANY,
            lsf.expression_hash_square(sub_le.representative_variable.name),
            sub_le.representative_variable,
            1,
        )

        self.assertCountEqual(
            master_le.variables,
            [
                (0.5, self.x),  # Binary variable used directly
                (0.5, sq_y.representative_variable),
                (-0.5, sq_p.representative_variable),
            ],
        )

    def test_reformulate_to_sum_of_squares_full_binary(self):
        """Test reformulation when both variables are binary."""
        self.x.var_type = "B"
        self.y.var_type = "B"

        master_le, sub_le = MagicMock(), MagicMock()
        sub_le.representative_variable = Variable("p_rep_full")
        sq_p = MagicMock()
        sq_p.representative_variable = Variable("p_sq_rep_full")

        self.model_data.add_linear_expression.side_effect = [master_le, sub_le]
        self.model_data.add_one_dim_expression.side_effect = [sq_p]

        self.bilinear_expr.reformulate_to_sum_of_squares()

        # Only square the helper variable p
        self.assertEqual(self.model_data.add_one_dim_expression.call_count, 1)
        self.model_data.add_one_dim_expression.assert_called_once_with(
            ANY,
            lsf.expression_hash_square(sub_le.representative_variable.name),
            sub_le.representative_variable,
            1,
        )

        self.assertCountEqual(
            master_le.variables,
            [
                (0.5, self.x),  # Binary variable used directly
                (0.5, self.y),  # Binary variable used directly
                (-0.5, sq_p.representative_variable),
            ],
        )


if __name__ == "__main__":
    unittest.main()
