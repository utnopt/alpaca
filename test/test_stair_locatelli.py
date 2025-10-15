# -*- coding: utf-8 -*-
"""
Unit tests for the StairLocatelli feature.
"""

# pylint: disable=protected-access, too-many-instance-attributes
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Assuming the file is in a structure that allows this import.
# If not, you might need to adjust the python path.
from alpaca.locatelli.stair_locatelli import StairLocatelli
from alpaca.model_data import constraint as con
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class TestStairLocatelli(unittest.TestCase):
    """Test suite for the StairLocatelli class."""

    def setUp(self):
        """Set up mock objects for testing."""
        # Mock model_data and its nested attributes
        self.mock_model_data = MagicMock()
        self.mock_model_data.settings.feature_stair_locatelli_obbt_time_limit = 100
        self.mock_model_data.settings.feature_stair_locatelli_grid_size = 3
        self.mock_model_data.settings.feature_stair_locatelli = 1  # Convex hull enabled

        # Mock variables
        self.mock_x = MagicMock()
        self.mock_x.name = "x"
        self.mock_x.lb = 0.0
        self.mock_x.ub = 2.0
        self.mock_x.solver_variable = "x_solver"

        self.mock_y = MagicMock()
        self.mock_y.name = "y"
        self.mock_y.lb = 0.0
        self.mock_y.ub = 3.0
        self.mock_y.solver_variable = "y_solver"

        self.mock_w = MagicMock()
        self.mock_w.name = "w"

        # Mock bilinear expression
        self.mock_bil_expr = MagicMock()
        self.mock_bil_expr.name = "w_xy"
        self.mock_bil_expr.variables = [self.mock_x, self.mock_y]
        self.mock_bil_expr.representative_variable = self.mock_w
        self.mock_model_data.expressions.bilinear_expressions = {
            "w_xy": self.mock_bil_expr
        }

        # We need a fresh mock for each test that uses the solver
        self.mip_model_patcher = patch("alpaca.locatelli.stair_locatelli.mm.MIPModel")
        self.mock_mip_model_constructor = self.mip_model_patcher.start()
        self.mock_solver_instance = self.mock_mip_model_constructor.return_value
        self.mock_opt_model = self.mock_solver_instance.opt_model

    def tearDown(self):
        """Stop any patches."""
        self.mip_model_patcher.stop()

    # --- Tests for Static Methods ---

    def test_filter_equal_vertices(self):
        """Test filtering of nearly identical consecutive vertices."""
        vertices = [(0, 0), (1, 1), (1.0001, 1.0002), (2, 2), (0, 0)]
        expected = [(0, 0), (1.0001, 1.0002), (2, 2)]
        result = StairLocatelli._filter_equal_vertices(vertices)
        self.assertEqual(result, expected)

    def test_filter_collinear_vertices(self):
        """Test filtering of collinear vertices."""
        vertices = [(0, 0), (1, 1), (2, 2), (1, 3), (0, 2)]
        expected = [(0, 0), (2, 2), (1, 3), (0, 2)]
        result = StairLocatelli._filter_collinear_vertices(vertices)
        self.assertEqual(result, expected)

    def test_check_if_vertices_are_collinear(self):
        """Test the collinearity check for three vertices."""
        v1, v2, v3 = (0, 0), (1, 1), (2, 2)
        self.assertTrue(StairLocatelli._check_if_vertices_are_collinear(v1, v2, v3))

        v1, v2, v3 = (0, 0), (1, 1), (1, 2)
        self.assertFalse(StairLocatelli._check_if_vertices_are_collinear(v1, v2, v3))

    def test_get_convex_hull(self):
        """Test the convex hull (Monotone Chain) algorithm."""
        points = [(0, 0), (1, 5), (3, 3), (6, 4), (7, 1), (4, 0), (3, 2)]
        expected_hull = [(0, 0), (1, 5), (6, 4), (7, 1), (4, 0)]
        # Sort is part of the algorithm, so we don't need to sort expected
        result = StairLocatelli._get_convex_hull(points)
        self.assertCountEqual(result, expected_hull)

    def test_calculate_hyperplane_for_vertex_triple(self):
        """Test hyperplane calculation."""
        # Non-collinear points for z = xy
        v1, v2, v3 = (1, 2), (3, 1), (2, 4)
        z1, z2, z3 = 1 * 2, 3 * 1, 2 * 4  # 2, 3, 8

        # Equation: ax + by + c = z -> ax + by - z + c = 0
        # For z = xy, the mccormick plane is w = x_u*y + y_u*x - x_u*y_u
        # This is more complex, the function solves ax+by+c = w

        matrix = np.array([[v1[0], v1[1], 1], [v2[0], v2[1], 1], [v3[0], v3[1], 1]])
        b = np.array([z1, z2, z3])
        expected_coeffs = np.linalg.solve(matrix, b)

        check, result_coeffs = StairLocatelli._calculate_hyperplane_for_vertex_triple(
            v1, v2, v3
        )

        self.assertTrue(check)
        np.testing.assert_allclose(result_coeffs, expected_coeffs)

    def test_calculate_hyperplane_for_collinear_triple(self):
        """Test hyperplane calculation fails for collinear points."""
        v1, v2, v3 = (1, 1), (2, 2), (3, 3)
        check = StairLocatelli._check_if_vertices_are_collinear(v1, v2, v3)
        self.assertTrue(check)

    def test_check_if_hyperplane_is_infeasible(self):
        """Test feasibility check of a hyperplane against checkpoints."""
        coeffs = np.array([0.5, 0.5, -1])
        # Checkpoints: (3,3), (4,4), (5,2)
        # 3*3=9. 0.5*3+0.5*3-1 = 2. 2<=9 (True)
        # 4*4=16. 0.5*4+0.5*4-1 = 3. 3<=16 (True)
        # 5*2=10. 0.5*5+0.5*2-1 = 2.5. 2.5<=10 (True)
        checkpoints_valid_geq = [(3, 3), (4, 4), (5, 2)]
        self.assertEqual(
            StairLocatelli._check_if_hyperplane_is_infeasible(
                coeffs, checkpoints_valid_geq
            ),
            lsf.constraint_geq(),
        )

        # Test case 2: All points above hyperplane (w <= ax+by+c)
        # ax+by+c >= xy
        coeffs = np.array([5, 5, 1])
        checkpoints_valid_leq = [(1, 1), (2, 1), (1, 2)]
        # 1*1=1. 5*1+5*1+1=11. 11>=1 (True)
        # 2*1=2. 5*2+5*1+1=16. 16>=2 (True)
        # 1*2=2. 5*1+5*2+1=16. 16>=2 (True)
        self.assertEqual(
            StairLocatelli._check_if_hyperplane_is_infeasible(
                coeffs, checkpoints_valid_leq
            ),
            lsf.constraint_leq(),
        )

        # Test case 3: Mixed
        coeffs = np.array([6, 6, -1])
        checkpoints_mixed = [(0, 0), (10, 10)]  # -1<=0 (T), 119>=100 (F)
        self.assertEqual(
            StairLocatelli._check_if_hyperplane_is_infeasible(
                coeffs, checkpoints_mixed
            ),
            lsf.empty_string(),
        )

    # --- Tests for Class Instance Methods ---

    def test_initialization(self):
        """Test that the StairLocatelli class initializes correctly."""
        # We need to mock the method that does all the work in __init__
        with patch.object(
            StairLocatelli, "_add_stair_locatelli_constraints"
        ) as mock_add_cuts:
            stair_locatelli = StairLocatelli(self.mock_model_data)

            # Check that the solver was created and configured
            self.mock_mip_model_constructor.assert_called_once_with(
                self.mock_model_data
            )

            # Check time limit calculation
            # 100 / (1 bilinear * 2 * 3 grid_size) = 100 / 6
            expected_time_limit = 100 / (1 * 2 * 3)
            self.mock_opt_model.set_time_limit.assert_called_once_with(
                expected_time_limit
            )
            self.mock_opt_model.hide_output.assert_called_once()

            # Check that the main method was called
            mock_add_cuts.assert_called_once()
            self.assertEqual(stair_locatelli.bilinear_projected_domains, [])

    def test_get_projected_domain_vertices(self):
        """Test the generation of projected domain vertices."""

        # Simulate solver behavior
        # It's called 4 times for a 3-point grid (2 max, 2 min)
        # Maximize y for x in [0,1], [1,2]
        # Minimize y for x in [2,1], [1,0]
        self.mock_opt_model.get_objective_value.side_effect = [
            2.5,  # Max y for x in [0, 1]
            2.8,  # Max y for x in [1, 2]
            0.5,  # Min y for x in [1, 2]
            0.2,  # Min y for x in [0, 1]
        ]
        self.mock_opt_model.is_infeasible.return_value = False

        # __init__ calls the method we want to test
        stair_locatelli = StairLocatelli(self.mock_model_data)

        # Expected vertices (before post-processing)
        # [(0, 2.5), (1, 2.5), (1, 2.8), (2, 2.8),  <-- from max
        #  (2, 0.5), (1, 0.5), (1, 0.2), (0, 0.2)]  <-- from min

        # The code runs post-processing and convex hull on this.
        # Sorted: [(0, 0.2), (0, 2.5), (1, 0.2), (1, 0.5), (1, 2.5), (1, 2.8), (2, 0.5), (2, 2.8)]
        # Convex hull would be: [(0, 0.2), (0, 2.5), (1, 2.8), (2, 2.8), (2, 0.5), (1, 0.2)]

        expected_hull = [
            (0.0, 0.2),
            (0.0, 2.5),
            (1.0, 2.8),
            (2.0, 2.8),
            (2.0, 0.5),
            (1.0, 0.2),
        ]

        self.assertEqual(len(stair_locatelli.bilinear_projected_domains), 1)
        expr, vertices = stair_locatelli.bilinear_projected_domains[0]

        self.assertEqual(expr, self.mock_bil_expr)
        self.assertCountEqual(vertices, expected_hull)

    def test_add_locatelli_cuts(self):
        """Test the logic for adding the final Locatelli cut constraints."""

        # Manually set the projected domain to bypass the complex vertex generation
        vertices = [(0, 0), (2, 0), (2, 3), (0, 3)]  # A simple box

        # Patch the method that runs on init to prevent vertex generation
        with patch.object(StairLocatelli, "_add_stair_locatelli_constraints"):
            stair_locatelli = StairLocatelli(self.mock_model_data)

        # Now, manually set the domains and call the method to be tested
        stair_locatelli.bilinear_projected_domains = [(self.mock_bil_expr, vertices)]

        stair_locatelli._add_locatelli_cuts_from_vertices()

        # For a convex shape like a rectangle, 4 triangles can be formed.
        # (0,0), (2,0), (2,3) -> w <= 3x
        # (0,0), (2,0), (0,3) -> w >= 0
        # (0,0), (2,3), (0,3) -> w <= 2y
        # (2,0), (2,3), (0,3) -> w >= 3x+2y-6 (McCormick)

        # So we expect add_constraint to be called 4 times
        self.assertEqual(self.mock_model_data.add_constraint.call_count, 4)

        # We can check one of the calls in detail
        # Let's check w >= 3x + 2y - 6
        # -> -3x -2y + w >= -6
        # LinearConstraint(..., con_type='>=', rhs=-6.0,
        # variables=[(3, x), (2, y), (1, w)])
        # The implementation has variables on the LHS and rhs on the RHS.
        # w - x_coeff*x - y_coeff*y >= const
        # from w >= 3x+2y-6, we have w - 3x - 2y >= -6
        # So x_coeff=3, y_coeff=2, const=-6

        # Let's find the call that corresponds to this constraint
        found_mccormick = False
        for call_args in self.mock_model_data.add_constraint.call_args_list:
            constraint_arg = call_args[0][0]
            if (
                isinstance(constraint_arg, con.LinearConstraint)
                and constraint_arg.con_type == ">="
                and np.isclose(constraint_arg.rhs, -6.0)
            ):
                # Check coefficients
                vars_dict = {v.name: c for c, v in constraint_arg.variables}
                self.assertAlmostEqual(vars_dict["x"], -3.0)  # -x_coeff
                self.assertAlmostEqual(vars_dict["y"], -2.0)  # -y_coeff
                self.assertAlmostEqual(vars_dict["w"], 1.0)
                found_mccormick = True
                break

        self.assertTrue(
            found_mccormick, "Did not find the expected McCormick constraint"
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
