# -*- coding: utf-8 -*-
"""
Unit tests for the MultilinearExpression class.
"""
# pylint: disable=protected-access
import unittest
from unittest.mock import MagicMock, call, patch

from alpaca.model_data.variable import Variable
from alpaca.expressions.multilinear_expression import MultilinearExpression
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

# Mock missing modules and dependencies to make tests self-contained.
MOCK_SETTINGS = MagicMock()
MOCK_SETTINGS.StaticSettings.infinity = float("inf")

MOCK_EXPRESSION_MODULE = MagicMock()


class MockExpression:
    """A mock base class for any expression."""

    def __init__(self, name, model_data, level, representative_variable=None):
        self.name = name
        self.model_data = model_data
        self.level = level
        if representative_variable is None:
            representative_variable = Variable(f"rep_{name}")
        self.representative_variable = representative_variable
        self.variables = []


MOCK_EXPRESSION_MODULE.Expression = MockExpression


class TestMultilinearExpression(unittest.TestCase):
    """Test suite for the MultilinearExpression class."""

    def setUp(self):
        """Set up test fixtures for each test method."""
        self.model_data = MagicMock()
        self.v = Variable(name="v", lb=0.1, ub=0.5)
        self.w = Variable(name="w", lb=-1, ub=2)
        self.x = Variable(name="x", lb=1, ub=5)
        self.y = Variable(name="y", lb=-2, ub=8)
        self.z = Variable(name="z", lb=0, ub=4)
        self.rep_var = Variable(name="rep_var")

    def test_init(self):
        """Test the __init__ method for variable registration."""
        variables = [self.x, self.y, self.z]
        expr = MultilinearExpression(
            name="test_init_expr",
            model_data=self.model_data,
            variables=variables,
            level=0,
        )
        self.assertEqual(expr.level, 0)
        for var in variables:
            self.assertTrue(var.is_discretized)

    def test_get_implied_lb_and_ub(self):
        """Test the static method get_implied_lb_and_ub."""
        # Case 1: All positive ranges
        res = MultilinearExpression.get_implied_lb_and_ub([(1, 2), (3, 4)])
        self.assertEqual(res, (3, 8))

        # Case 2: Mixed positive and negative ranges
        res = MultilinearExpression.get_implied_lb_and_ub([(-2, 3), (1, 5)])
        self.assertEqual(res, (-10, 15))

        # Case 3: All negative ranges
        res = MultilinearExpression.get_implied_lb_and_ub([(-4, -2), (-3, -1)])
        self.assertEqual(res, (2, 12))

        # Case 4: Range includes zero
        res = MultilinearExpression.get_implied_lb_and_ub([(-5, 5), (-10, 2)])
        self.assertEqual(res, (-50, 50))

        # Case 5: Creative case with five variables
        res = MultilinearExpression.get_implied_lb_and_ub(
            [(-1, 1), (-2, 2), (-3, 3), (-0.5, 4), (1, 10)]
        )
        # Max positive product: 1*2*3*4*10 = 240
        # Max negative product: 1*2*3*(-0.5)*10 = -30, or -1*2*3*4*10 = -240
        self.assertEqual(res, (-240, 240))

    def test_reformulate_to_bilinear_trilinear(self):
        """Test reformulation of a 3-variable expression."""
        sub_bilinear = MagicMock()
        sub_bilinear.representative_variable = Variable("sub_rep")
        self.model_data.add_bilinear_expression.return_value = sub_bilinear

        expr = MultilinearExpression(
            "tri_expr", self.model_data, [self.x, self.y, self.z], 0, self.rep_var
        )
        expr.reformulate_to_bilinear_expressions()

        self.assertEqual(self.model_data.add_bilinear_expression.call_count, 2)
        expected_calls = [
            call(
                lsf.expression_hash_bilinear(self.y.name, self.z.name),
                [self.y, self.z],
                1,
            ),
            call(
                lsf.expression_hash_bilinear(
                    self.x.name, sub_bilinear.representative_variable.name
                ),
                [self.x, sub_bilinear.representative_variable],
                0,
                representative_variable=self.rep_var,
            ),
        ]
        self.model_data.add_bilinear_expression.assert_has_calls(
            expected_calls, any_order=True
        )

    @patch("alpaca.expressions.multilinear_expression.MultilinearExpression")
    def test_reformulate_to_bilinear_pentalinear(self, mock_mle_class):
        """Test recursive reformulation of a 5-variable expression."""
        mock_sub_mle = mock_mle_class.return_value
        mock_sub_mle.representative_variable = Variable("sub_mle_rep")

        expr = MultilinearExpression(
            "penta_expr",
            self.model_data,
            [self.v, self.w, self.x, self.y, self.z],
            0,
            self.rep_var,
        )
        expr.reformulate_to_bilinear_expressions()

        # Check that a sub-multilinear expression was created for (w*x*y*z)
        mock_mle_class.assert_called_once_with(
            lsf.expression_hash_multilinear(
                [self.w.name, self.x.name, self.y.name, self.z.name]
            ),
            self.model_data,
            [self.w, self.x, self.y, self.z],
            1,
        )
        # Check that the sub-expression's reformulation was called
        mock_sub_mle.reformulate_to_bilinear_expressions.assert_called_once()
        # Check that the final bilinear expression for v*(wxyz) was created
        self.model_data.add_bilinear_expression.assert_called_once_with(
            lsf.expression_hash_bilinear(
                self.v.name, mock_sub_mle.representative_variable.name
            ),
            [self.v, mock_sub_mle.representative_variable],
            0,
            representative_variable=self.rep_var,
        )

    def test_extract_mpip_relation_not_discretized(self):
        """Test that extract_mpip_relation exits early if rep_var is not discretized."""
        expr = MultilinearExpression(
            "early_exit", self.model_data, [self.x, self.y], 0, self.rep_var
        )
        self.rep_var.is_discretized = False
        expr.extract_mpip_relation()
        self.assertFalse(expr.piecewise_constant_relation)  # Should be empty dict

    def test_mpip_helpers(self):
        """Test helper methods for MPIP relation extraction."""
        self.x.breakpoints = [0, 1, 2.5, 5]  # Non-uniform
        self.y.breakpoints = [10, 15, 20]
        self.rep_var.breakpoints = [0, 15, 30, 45, 100]  # Non-uniform

        expr = MultilinearExpression(
            "mpip_expr", self.model_data, [self.x, self.y], 0, self.rep_var
        )

        mid_vals = expr.get_all_variables_mid_values_with_indices()
        self.assertEqual(
            mid_vals, [[(0.5, 0), (1.75, 1), (3.75, 2)], [(12.5, 0), (17.5, 1)]]
        )

        self.assertEqual(expr._get_implied_index_from_implied_value(-10), 0)
        self.assertEqual(expr._get_implied_index_from_implied_value(14.9), 0)
        self.assertEqual(
            expr._get_implied_index_from_implied_value(15), 0
        )  # Exactly on breakpoint
        self.assertEqual(expr._get_implied_index_from_implied_value(45), 2)
        self.assertEqual(
            expr._get_implied_index_from_implied_value(100), 3
        )  # Upper bound
        self.assertEqual(
            expr._get_implied_index_from_implied_value(200), 3
        )  # Beyond upper bound

    def test_extract_mpip_relation_3_vars(self):
        """Test the main MPIP relation extraction method with three variables."""
        self.x.breakpoints = [0, 1]
        self.y.breakpoints = [10, 12]
        self.z.breakpoints = [2, 3]
        self.rep_var.breakpoints = [0, 22, 40]
        self.rep_var.is_discretized = True

        expr = MultilinearExpression(
            "mpip_3var", self.model_data, [self.x, self.y, self.z], 0, self.rep_var
        )

        # Test with approximation = False (exact interval)
        expr.extract_mpip_relation(approximation=False)
        expected_exact = {
            # Only one combination possible for each var
            (0, 0, 0): (0, 1),  # x[0,1],y[10,12],z[2,3] -> prod[0, 36] -> indices(0,1)
        }
        self.assertDictEqual(expr.piecewise_constant_relation, expected_exact)

        # Test with approximation = True
        expr.extract_mpip_relation(approximation=True)
        expected_approx = {
            # Mid values (0.5, 11, 2.5) -> prod 13.75 -> index 0
            (0, 0, 0): (0,)
        }
        self.assertDictEqual(expr.piecewise_constant_relation, expected_approx)


if __name__ == "__main__":
    unittest.main()
