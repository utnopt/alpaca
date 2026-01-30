# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay  # pylint: disable=no-name-in-module


def calculate_hyperplane_for_vertex_triple(v1, v2, v3):
    """
    Calculate hyperplane coefficients for three points in 2D (z = ax + by + c).
    Returns: (success: bool, coefficients: np.ndarray[a, b, c])
    """
    matrix = np.array([[v1[0], v1[1], 1], [v2[0], v2[1], 1], [v3[0], v3[1], 1]])
    b = np.array([v1[0] * v1[1], v2[0] * v2[1], v3[0] * v3[1]])
    try:
        return True, np.linalg.solve(matrix, b)
    except np.linalg.LinAlgError:
        return False, np.array([0.0, 0.0, 0.0])


def check_if_vertices_are_collinear(v1, v2, v3, tolerance=1e-5):
    """Check if three points are collinear using cross product."""
    vec1 = np.array([v2[0] - v1[0], v2[1] - v1[1]])
    vec2 = np.array([v3[0] - v2[0], v3[1] - v2[1]])
    return abs(np.cross(vec1, vec2)) < tolerance


def get_convex_hull(vertices):
    """
    Calculates the convex hull of a set of 2D vertices using the Monotone Chain algorithm.
    Returns vertices in counter-clockwise order.
    """

    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    unique_vertices = sorted(list(set(vertices)))
    if len(unique_vertices) <= 2:
        return unique_vertices

    # Build lower hull
    lower_hull = []
    for v in unique_vertices:
        while (
            len(lower_hull) >= 2
            and cross_product(lower_hull[-2], lower_hull[-1], v) <= 0
        ):
            lower_hull.pop()
        lower_hull.append(v)

    # Build upper hull
    upper_hull = []
    for v in reversed(unique_vertices):
        while (
            len(upper_hull) >= 2
            and cross_product(upper_hull[-2], upper_hull[-1], v) <= 0
        ):
            upper_hull.pop()
        upper_hull.append(v)

    # Concatenate and remove duplicates (first/last points are present in both)
    return lower_hull + upper_hull[1:-1]


def filter_equal_vertices(vertices, atol=1e-3):
    """Removes consecutive duplicate vertices."""
    if not vertices:
        return []
    filtered = []
    # Check against the next vertex (wrapping around)
    for i, v1 in enumerate(vertices):
        v2 = vertices[(i + 1) % len(vertices)]
        if not (
            np.isclose(v1[0], v2[0], atol=atol) and np.isclose(v1[1], v2[1], atol=atol)
        ):
            filtered.append(v1)
    return filtered


def filter_collinear_vertices(vertices, tolerance=1e-5):
    """Removes vertices that lie on the straight line between their neighbors."""
    if len(vertices) < 3:
        return vertices

    triple_vertices = vertices + vertices[:2]
    indices_to_keep = []

    for i in range(len(vertices)):
        v1, v2, v3 = triple_vertices[i], triple_vertices[i + 1], triple_vertices[i + 2]

        vec1 = np.array([v2[0] - v1[0], v2[1] - v1[1]])
        vec2 = np.array([v3[0] - v2[0], v3[1] - v2[1]])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # If points are too close, treat as collinear/redundant
        if norm1 < tolerance or norm2 < tolerance:
            continue

        normalized_vec1 = vec1 / norm1
        normalized_vec2 = vec2 / norm2

        # Dot product of 1.0 means they are parallel in same direction
        if not np.isclose(
            np.dot(normalized_vec1, normalized_vec2), 1.0, atol=tolerance
        ):
            indices_to_keep.append((i + 1) % len(vertices))

    return [vertices[i] for i in indices_to_keep]


def calculate_locatelli_volume_polytope(
    bilinear_expression,
    domain_vertices: list[tuple[float, float]],
    grid_size: int = 100,
) -> float:
    """
    Calculates the volume of the convex hull of points mapped to z = x*y over a specific domain.
    """
    x = bilinear_expression.variables[0]
    y = bilinear_expression.variables[1]
    x_range = np.linspace(x.lb, x.ub, grid_size)
    y_range = np.linspace(y.lb, y.ub, grid_size)
    valid_points = calculate_x_y_domain_polytope(x_range, y_range, domain_vertices)
    return calculate_3d_volume_polytope(valid_points)


def calculate_3d_volume_polytope(valid_points: np.ndarray) -> float:
    """Calculate the volume of the convex hull of 3D points (x, y, x*y)."""
    # Create 3D points (x, y, x*y)
    point_cloud = np.column_stack(
        (
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 0] * valid_points[:, 1],
        )
    )

    if len(point_cloud) < 4:
        return 0.0

    hull = ConvexHull(point_cloud)
    return hull.volume


def calculate_x_y_domain_polygon(  # pylint: disable=too-many-locals
    x_range: np.ndarray, y_range: np.ndarray, domain_vertices: list[tuple[float, float]]
) -> np.ndarray:
    """Calculate (x, y) points inside a polygon defined by domain_vertices."""
    vertices_arr = np.array(domain_vertices)

    x_points, y_points = np.meshgrid(x_range, y_range, indexing="xy")
    points = np.column_stack([x_points.ravel(), y_points.ravel()])

    x, y = points[:, 0], points[:, 1]
    xp, yp = vertices_arr[:, 0], vertices_arr[:, 1]

    inside = np.zeros(len(points), dtype=bool)
    j = len(vertices_arr) - 1

    # Ray casting algorithm (even-odd rule)
    for i in range(len(vertices_arr)):
        xi, yi = xp[i], yp[i]
        xj, yj = xp[j], yp[j]

        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        )
        inside ^= intersect
        j = i

    return points[inside]


def calculate_x_y_domain_polytope(
    x_range: np.ndarray, y_range: np.ndarray, domain_vertices: list[tuple[float, float]]
) -> np.ndarray:
    """
    Calculate the (x, y) points within a polytope defined by domain_vertices.
    Uses Delaunay triangulation to determine if points are inside the convex hull.
    """
    vertices_arr = np.array(domain_vertices)

    x_points, y_points = np.meshgrid(x_range, y_range, indexing="xy")
    points = np.column_stack([x_points.ravel(), y_points.ravel()])

    hull = ConvexHull(vertices_arr)

    tri = Delaunay(vertices_arr[hull.vertices])

    inside = tri.find_simplex(points) >= 0
    return points[inside]
