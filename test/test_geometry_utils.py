# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np
import pytest
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

import alpaca.utils.geometry as uge


class TestGeometryUtils:  # pylint: disable=too-many-public-methods
    """Test suite for the geometry module."""

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def simple_triangle(self):
        """Return a simple right triangle."""
        return [(0, 0), (1, 0), (0, 1)]

    @pytest.fixture
    def unit_square(self):
        """Return a unit square polygon."""
        return [(0, 0), (1, 0), (1, 1), (0, 1)]

    @pytest.fixture
    def x_range_default(self):
        """Return default x range for domain calculations."""
        return np.linspace(0, 1, 11)

    @pytest.fixture
    def y_range_default(self):
        """Return default y range for domain calculations."""
        return np.linspace(0, 1, 11)

    @pytest.fixture
    def convexified_area_fixture(self, unit_square, x_range_default, y_range_default):
        """Return pre-calculated convexified area for testing."""
        return uge.calculate_convexified_area(
            x_range_default, y_range_default, unit_square
        )

    # =========================================================================
    # Tests: calculate_hyperplane_for_vertex_triple
    # =========================================================================

    def test_hyperplane_valid_non_collinear_points(self):
        """Test hyperplane calculation with valid non-collinear points."""
        vertex_1 = (0, 0)
        vertex_2 = (1, 0)
        vertex_3 = (0, 1)

        success, coeffs = uge.calculate_hyperplane_for_vertex_triple(
            vertex_1, vertex_2, vertex_3
        )

        assert success is True
        assert len(coeffs) == 3
        assert isinstance(coeffs, np.ndarray)

    def test_hyperplane_coefficients_satisfy_points(self):
        """Test that returned coefficients satisfy all three points."""
        vertex_1 = (1, 2)
        vertex_2 = (3, 4)
        vertex_3 = (2, 5)

        success, coeffs = uge.calculate_hyperplane_for_vertex_triple(
            vertex_1, vertex_2, vertex_3
        )

        assert success is True
        coeff_a, coeff_b, coeff_c = coeffs

        for vertex in [vertex_1, vertex_2, vertex_3]:
            expected_z = vertex[0] * vertex[1]
            calculated_z = coeff_a * vertex[0] + coeff_b * vertex[1] + coeff_c
            assert np.isclose(expected_z, calculated_z, atol=1e-10)

    def test_hyperplane_collinear_points_singular_matrix(self):
        """Test hyperplane with collinear points that create a singular matrix."""
        vertex_1 = (0, 0)
        vertex_2 = (1, 1)
        vertex_3 = (2, 2)

        success, coeffs = uge.calculate_hyperplane_for_vertex_triple(
            vertex_1, vertex_2, vertex_3
        )

        assert success is False
        assert np.allclose(coeffs, [0.0, 0.0, 0.0])

    # =========================================================================
    # Tests: check_if_vertices_are_collinear
    # =========================================================================

    def test_collinear_horizontal(self):
        """Test collinear points on horizontal line."""
        vertex_1 = (0, 0)
        vertex_2 = (1, 0)
        vertex_3 = (2, 0)

        assert (
            uge.check_if_vertices_are_collinear(vertex_1, vertex_2, vertex_3)
            == np.True_
        )

    def test_non_collinear_triangle(self):
        """Test non-collinear points forming a triangle."""
        vertex_1 = (0, 0)
        vertex_2 = (1, 0)
        vertex_3 = (0.5, 1)

        assert (
            uge.check_if_vertices_are_collinear(vertex_1, vertex_2, vertex_3)
            == np.False_
        )

    @pytest.mark.parametrize(
        "v1,v2,v3,expected",
        [
            ((0, 0), (1, 0), (2, 0), True),
            ((0, 0), (1, 1), (2, 2), True),
            ((0, 0), (1, 0), (1, 1), False),
            ((0, 0), (1, 0), (0, 1), False),
        ],
    )
    def test_collinear_parametrized(self, v1, v2, v3, expected):
        """Parametrized collinearity tests."""
        assert uge.check_if_vertices_are_collinear(v1, v2, v3) == expected

    # =========================================================================
    # Tests: get_convex_hull
    # =========================================================================

    def test_convex_hull_simple_triangle(self, simple_triangle):
        """Test convex hull of a triangle."""
        hull = uge.get_convex_hull(simple_triangle)

        assert len(hull) == 3
        for vertex in simple_triangle:
            assert vertex in hull

    def test_convex_hull_unit_square(self, unit_square):
        """Test convex hull of a square."""
        hull = uge.get_convex_hull(unit_square)

        assert len(hull) == 4
        for vertex in unit_square:
            assert vertex in hull

    def test_convex_hull_with_interior_point(self):
        """Test convex hull with interior points that should be excluded."""
        vertices = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]

        hull = uge.get_convex_hull(vertices)

        assert len(hull) == 4
        assert (1, 1) not in hull

    def test_convex_hull_empty_list(self):
        """Test convex hull with empty list."""
        hull = uge.get_convex_hull([])

        assert hull == []

    # =========================================================================
    # Tests: filter_equal_vertices
    # =========================================================================

    def test_filter_equal_no_duplicates(self, unit_square):
        """Test filter_equal_vertices with no duplicate vertices."""
        filtered = uge.filter_equal_vertices(unit_square)

        assert len(filtered) == 4

    def test_filter_equal_consecutive_duplicates(self):
        """Test removal of consecutive duplicates."""
        vertices = [(0, 0), (0, 0), (1, 0), (1, 1)]

        filtered = uge.filter_equal_vertices(vertices)

        assert len(filtered) == 3

    def test_filter_equal_empty_list(self):
        """Test filter_equal_vertices with empty list."""
        filtered = uge.filter_equal_vertices([])

        assert not filtered

    # =========================================================================
    # Tests: filter_collinear_vertices
    # =========================================================================

    def test_filter_collinear_no_collinear(self, simple_triangle):
        """Test filter_collinear_vertices with no collinear vertices."""
        filtered = uge.filter_collinear_vertices(simple_triangle)

        assert len(filtered) == 3

    def test_filter_collinear_middle_point(self):
        """Test removal of collinear middle point."""
        vertices = [(0, 0), (0.5, 0), (1, 0), (1, 1), (0, 1)]

        filtered = uge.filter_collinear_vertices(vertices)

        assert (0.5, 0) not in filtered
        assert len(filtered) == 4

    def test_filter_collinear_empty_list(self):
        """Test filter_collinear_vertices with empty list."""
        filtered = uge.filter_collinear_vertices([])

        assert filtered == []

    # =========================================================================
    # Tests: calculate_x_y_domain_polygon
    # =========================================================================

    def test_domain_polygon_unit_square_all_inside(self):
        """Test that points inside unit square are correctly identified."""
        x_range = np.array([0.25, 0.5, 0.75])
        y_range = np.array([0.25, 0.5, 0.75])
        domain = [(0, 0), (1, 0), (1, 1), (0, 1)]

        points = uge.calculate_x_y_domain_polygon(x_range, y_range, domain)

        assert len(points) == 9

    def test_domain_polygon_triangle(self, simple_triangle):
        """Test points inside triangle."""
        x_range = np.linspace(0, 1, 11)
        y_range = np.linspace(0, 1, 11)

        points = uge.calculate_x_y_domain_polygon(x_range, y_range, simple_triangle)

        assert len(points) > 0
        for point in points:
            assert point[0] + point[1] <= 1 + 1e-10

    def test_domain_polygon_points_outside(self):
        """Test that points outside polygon are excluded."""
        x_range = np.array([2, 3, 4])
        y_range = np.array([2, 3, 4])
        domain = [(0, 0), (1, 0), (1, 1), (0, 1)]

        points = uge.calculate_x_y_domain_polygon(x_range, y_range, domain)

        assert len(points) == 0

    # =========================================================================
    # Tests: calculate_x_y_domain_polytope
    # =========================================================================

    def test_domain_polytope_unit_square(
        self, unit_square, x_range_default, y_range_default
    ):
        """Test calculate_x_y_domain_polytope with unit square domain."""
        points = uge.calculate_x_y_domain_polytope(
            x_range_default, y_range_default, unit_square
        )

        assert len(points) > 0
        assert isinstance(points, np.ndarray)

    def test_domain_polytope_triangle(self, simple_triangle):
        """Test calculate_x_y_domain_polytope with triangle domain."""
        x_range = np.linspace(0, 1, 21)
        y_range = np.linspace(0, 1, 21)

        points = uge.calculate_x_y_domain_polytope(x_range, y_range, simple_triangle)

        assert len(points) > 0

    # =========================================================================
    # Tests: calculate_convexified_area
    # =========================================================================

    def test_convexified_area_returns_convex_hull(
        self, unit_square, x_range_default, y_range_default
    ):
        """Test that function returns a ConvexHull object."""
        result = uge.calculate_convexified_area(
            x_range_default, y_range_default, unit_square
        )

        assert isinstance(result, ConvexHull)

    def test_convexified_area_volume_positive(
        self, unit_square, x_range_default, y_range_default
    ):
        """Test that hull has positive volume."""
        result = uge.calculate_convexified_area(
            x_range_default, y_range_default, unit_square
        )

        assert result.volume > 0

    # =========================================================================
    # Tests: calculate_feasible_height_convexified
    # =========================================================================

    def test_feasible_height_point_inside_returns_positive(
        self, convexified_area_fixture
    ):
        """Test that point inside domain returns positive height."""
        values = (0.5, 0.5)

        height = uge.calculate_feasible_height_convexified(
            values, convexified_area_fixture
        )

        assert height >= 0

    def test_feasible_height_point_outside_returns_zero(self, convexified_area_fixture):
        """Test that point outside domain returns zero."""
        values = (10.0, 10.0)

        height = uge.calculate_feasible_height_convexified(
            values, convexified_area_fixture
        )

        assert height == 0.0

    def test_feasible_height_non_negative(self, convexified_area_fixture):
        """Test that height is never negative."""
        test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (-0.5, 0.5)]

        for point in test_points:
            height = uge.calculate_feasible_height_convexified(
                point, convexified_area_fixture
            )
            assert height >= 0, f"Negative height at point {point}"

    # =========================================================================
    # Tests: Integration
    # =========================================================================

    def test_integration_full_workflow_square(self):
        """Test complete workflow with square domain."""
        domain = [(0, 0), (1, 0), (1, 1), (0, 1)]

        filtered = uge.filter_equal_vertices(domain)
        filtered = uge.filter_collinear_vertices(filtered)
        assert len(filtered) == 4

        x_range = np.linspace(0, 1, 11)
        y_range = np.linspace(0, 1, 11)
        hull = uge.calculate_convexified_area(x_range, y_range, filtered)

        height = uge.calculate_feasible_height_convexified((0.5, 0.5), hull)
        assert height > 0
