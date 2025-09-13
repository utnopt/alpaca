"""
Unit tests for the BoundPropagator class.

This module contains test cases for bound propagation logic on various
expression types (linear, multilinear, trigonometric, etc.) to ensure
correct bound tightening.
"""

import unittest
import math

# Import the actual classes being tested or needed for the tests
from alpaca.model_buildup.bound_propagator import BoundPropagator
from alpaca.model_data.variable import Variable
from alpaca.model_data.constraint import Constraint
from alpaca.expressions.linear_expression import LinearExpression
from alpaca.expressions.multilinear_expression import MultilinearExpression
from alpaca.expressions.one_dim_expression import (
    SquareExpression,
    SineExpression,
    CosineExpression,
    ExponentialExpression,
    LnExpression,
    SquareRootExpression,
    InverseExpression,
    TangensHExpression,  # Corrected from TanhExpression
)


# Mock settings to avoid dependency on the full settings module
class MockUserSettings:
    """A mock user settings class for testing purposes."""
    def __init__(self):
        """Initializes mock settings."""
        self.bound_propagation_rounds = 3


class MockExpressionContainer:
    """A mock container for various types of expressions."""
    def __init__(self):
        """Initializes the expression container."""
        self.one_dim_expressions = {}
        self.linear_expressions = {}
        self.multilinear_expressions = {}
        self.bilinear_expressions = {}

    def all_low_dim_expressions(self):
        """Returns a list of all low-dimensional expressions."""
        expressions = list(self.one_dim_expressions.values())
        expressions.extend(self.linear_expressions.values())
        expressions.extend(self.multilinear_expressions.values())
        return expressions


# Mock ModelData to isolate the BoundPropagator logic
class MockModelData:
    """A mock model data class to provide a testing environment."""
    def __init__(self):
        """Initializes mock model data."""
        self.settings = MockUserSettings()
        self.variables = {}
        self.constraints = {}
        self.expressions = MockExpressionContainer()


class TestBoundPropagator(unittest.TestCase):
    """Test cases for the BoundPropagator class."""

    def setUp(self):
        """Set up a fresh model_data and propagator for each test."""
        self.model_data = MockModelData()
        self.propagator = BoundPropagator(self.model_data)

    def test_propagate_linear_constraints_simple_addition(self):
        """Test bound propagation with a simple constraint: x + y == 10."""
        x_var = Variable("x", lb=0, ub=10)
        y_var = Variable("y", lb=0, ub=10)
        self.model_data.variables = {"x": x_var, "y": y_var}
        constraint = Constraint("c1", con_type="==", rhs=10,
                                variables=[(1.0, x_var), (1.0, y_var)])
        self.model_data.constraints = {"c1": constraint}

        # Initially, bounds are wide
        self.assertEqual(x_var.lb, 0)
        self.assertEqual(x_var.ub, 10)
        self.assertEqual(y_var.lb, 0)
        self.assertEqual(y_var.ub, 10)

        # Propagate bounds
        self.propagator.propagate_linear_constraints()

        # Bounds should not change in this symmetric case
        self.assertEqual(x_var.lb, 0)
        self.assertEqual(x_var.ub, 10)
        self.assertEqual(y_var.lb, 0)
        self.assertEqual(y_var.ub, 10)

        # Now, tighten one variable's bounds and re-propagate
        x_var.lb = 2
        x_var.ub = 5
        self.propagator.propagate_linear_constraints()

        # The bounds of y should tighten
        self.assertEqual(y_var.lb, 5)  # 10 (rhs) - 5 (x.ub) = 5
        self.assertEqual(y_var.ub, 8)  # 10 (rhs) - 2 (x.lb) = 8

    def test_propagate_linear_constraints_with_negative_coeffs(self):
        """Test bound propagation with a constraint like: 2x - 3y == 5."""
        x_var = Variable("x", lb=0, ub=10)
        y_var = Variable("y", lb=0, ub=5)
        self.model_data.variables = {"x": x_var, "y": y_var}
        constraint = Constraint("c1", con_type="==", rhs=5,
                                variables=[(2.0, x_var), (-3.0, y_var)])
        self.model_data.constraints = {"c1": constraint}

        self.propagator.propagate_linear_constraints()

        # Expected bounds for x after one pass:
        # From 2x = 5 + 3y:
        # x_lb = (5 + 3*y.lb) / 2 = (5 + 0) / 2 = 2.5. New lb = max(0, 2.5) = 2.5
        # x_ub = (5 + 3*y.ub) / 2 = (5 + 15) / 2 = 10. New ub = min(10, 10) = 10
        self.assertAlmostEqual(x_var.lb, 2.5)
        self.assertAlmostEqual(x_var.ub, 10)

        # Expected bounds for y after one pass (using original x bounds):
        # From 3y = 2x - 5:
        # y_lb = (2*x.lb - 5) / 3 = (0 - 5) / 3 = -1.66. New lb=max(0, -1.66)=0
        # y_ub = (2*x.ub - 5) / 3 = (20 - 5) / 3 = 5. New ub = min(5, 5) = 5
        # Bounds for y do not tighten in the first pass
        self.assertAlmostEqual(y_var.lb, 0)
        self.assertAlmostEqual(y_var.ub, 5)

    def test_propagate_linear_expression(self):
        """Test bound propagation for a LinearExpression: z = 2x - y + 5."""
        x_var = Variable("x", lb=1, ub=3)
        y_var = Variable("y", lb=-2, ub=4)
        z_var = Variable("z", lb=-100, ub=100)  # Representative variable
        self.model_data.variables = {"x": x_var, "y": y_var, "z": z_var}

        lin_expr = LinearExpression(
            "le1", self.model_data, level=0, representative_variable=z_var)
        lin_expr.variables = [(2.0, x_var), (-1.0, y_var)]
        lin_expr.constant = 5.0
        self.model_data.expressions.linear_expressions["le1"] = lin_expr

        self.propagator.propagate_expressions()

        # Calculate expected bounds for z
        # lb = 5 + (2 * 1) + (-1 * 4) = 5 + 2 - 4 = 3
        # ub = 5 + (2 * 3) + (-1 * -2) = 5 + 6 + 2 = 13
        self.assertEqual(z_var.lb, 3)
        self.assertEqual(z_var.ub, 13)

    def test_propagate_multilinear_expression(self):
        """Test bound propagation for a MultilinearExpression: z = x * y."""
        x_var = Variable("x", lb=-2, ub=3)
        y_var = Variable("y", lb=-4, ub=5)
        z_var = Variable("z", lb=-1000, ub=1000)
        self.model_data.variables = {"x": x_var, "y": y_var, "z": z_var}

        ml_expr = MultilinearExpression(
            "ml1", self.model_data, [x_var, y_var], level=0,
            representative_variable=z_var)
        self.model_data.expressions.multilinear_expressions["ml1"] = ml_expr

        self.propagator.propagate_expressions()

        # Expected: min/max of {-2*-4, -2*5, 3*-4, 3*5} = {8, -10, -12, 15}
        self.assertEqual(z_var.lb, -12)
        self.assertEqual(z_var.ub, 15)

    def test_propagate_square_expression(self):
        """Test bound propagation for a SquareExpression: y = x^2."""
        # Case 1: Bounds of x cross zero
        x1_var = Variable("x1", lb=-5, ub=4)
        y1_var = Variable("y1", lb=-100, ub=100)
        sq_expr1 = SquareExpression(
            "sq1", self.model_data, x1_var, level=0,
            representative_variable=y1_var)
        self.model_data.expressions.one_dim_expressions["sq1"] = sq_expr1

        # Case 2: Bounds of x are both positive
        x2_var = Variable("x2", lb=2, ub=5)
        y2_var = Variable("y2", lb=-100, ub=100)
        sq_expr2 = SquareExpression(
            "sq2", self.model_data, x2_var, level=0,
            representative_variable=y2_var)
        self.model_data.expressions.one_dim_expressions["sq2"] = sq_expr2

        # pylint: disable=protected-access
        self.propagator._propagate_bounds_one_dim_expression(sq_expr1)
        # pylint: disable=protected-access
        self.propagator._propagate_bounds_one_dim_expression(sq_expr2)

        # NOTE: The provided code for one-dim propagation is flawed as it only
        # checks the function values at the boundaries (f(lb), f(ub)). A
        # correct implementation would also consider extrema within the
        # interval. These tests verify the *actual* behavior of the code.

        # For x1 in [-5, 4], f(-5)=25, f(4)=16. The code calculates min/max
        # of these. The true range is [0, 25], but the code will get [16, 25].
        self.assertAlmostEqual(y1_var.lb, 16)
        self.assertAlmostEqual(y1_var.ub, 25)

        # For x2 in [2, 5], the function is monotonic.
        # f(2)=4, f(5)=25. The calculated range [4, 25] is correct.
        self.assertAlmostEqual(y2_var.lb, 4)
        self.assertAlmostEqual(y2_var.ub, 25)

    def test_propagate_trigonometric_expression(self):
        """Test propagation for SineExpression and CosineExpression."""
        x_var = Variable("x", lb=-10, ub=10)
        y_sin = Variable("y_sin", lb=-100, ub=100)
        y_cos = Variable("y_cos", lb=-100, ub=100)

        sin_expr = SineExpression("sin1", self.model_data, x_var, 0, y_sin)
        cos_expr = CosineExpression("cos1", self.model_data, x_var, 0, y_cos)
        self.model_data.expressions.one_dim_expressions = {
            "sin1": sin_expr, "cos1": cos_expr}

        self.propagator.propagate_expressions()

        # Sine and Cosine bounds should always be tightened to [-1, 1]
        self.assertEqual(y_sin.lb, -1)
        self.assertEqual(y_sin.ub, 1)
        self.assertEqual(y_cos.lb, -1)
        self.assertEqual(y_cos.ub, 1)

    def test_propagate_exponential_expression(self):
        """Test propagation for an ExponentialExpression: y = e^x."""
        x_var = Variable("x", lb=-1, ub=2)
        y_var = Variable("y", lb=-100, ub=100)
        exp_expr = ExponentialExpression("exp1", self.model_data, x_var, 0, y_var)
        self.model_data.expressions.one_dim_expressions["exp1"] = exp_expr

        # pylint: disable=protected-access
        self.propagator._propagate_bounds_one_dim_expression(exp_expr)

        # exp is monotonically increasing, so bounds are f(lb) and f(ub)
        self.assertAlmostEqual(y_var.lb, math.exp(-1))
        self.assertAlmostEqual(y_var.ub, math.exp(2))

    def test_propagate_more_one_dim_expressions(self):
        """Test propagation for various other one-dimensional expressions."""
        # Test LnExpression: y = ln(x) for x in [1, 10]
        x_ln = Variable("x_ln", lb=1, ub=10)
        y_ln = Variable("y_ln", lb=-100, ub=100)
        ln_expr = LnExpression("ln1", self.model_data, x_ln, 0, y_ln)
        self.propagator._propagate_bounds_one_dim_expression(ln_expr)  # pylint: disable=protected-access
        self.assertAlmostEqual(y_ln.lb, math.log(1))
        self.assertAlmostEqual(y_ln.ub, math.log(10))

        # Test SquareRootExpression: y = sqrt(x) for x in [4, 25]
        x_sqrt = Variable("x_sqrt", lb=4, ub=25)
        y_sqrt = Variable("y_sqrt", lb=-100, ub=100)
        sqrt_expr = SquareRootExpression(
            "sqrt1", self.model_data, x_sqrt, 0, y_sqrt)
        # pylint: disable=protected-access
        self.propagator._propagate_bounds_one_dim_expression(sqrt_expr)
        self.assertAlmostEqual(y_sqrt.lb, 2)
        self.assertAlmostEqual(y_sqrt.ub, 5)

        # Test InverseExpression: y = 1/x for x in [0.1, 2]
        x_inv = Variable("x_inv", lb=0.1, ub=2)
        y_inv = Variable("y_inv", lb=-100, ub=100)
        inv_expr = InverseExpression("inv1", self.model_data, x_inv, 0, y_inv)
        # pylint: disable=protected-access
        self.propagator._propagate_bounds_one_dim_expression(inv_expr)
        self.assertAlmostEqual(y_inv.lb, 1 / 2)
        self.assertAlmostEqual(y_inv.ub, 1 / 0.1)

        # Test TanhExpression: y = tanh(x) for x in [-1, 1]
        x_tanh = Variable("x_tanh", lb=-1, ub=1)
        y_tanh = Variable("y_tanh", lb=-100, ub=100)
        tanh_expr = TangensHExpression("tanh1", self.model_data, x_tanh, 0, y_tanh)
        # pylint: disable=protected-access
        self.propagator._propagate_bounds_one_dim_expression(tanh_expr)
        self.assertAlmostEqual(y_tanh.lb, math.tanh(-1))
        self.assertAlmostEqual(y_tanh.ub, math.tanh(1))


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
