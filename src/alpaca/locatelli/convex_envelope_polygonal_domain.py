"""Construct candidate cells for the convex envelope of ``f(x, y) = x*y``.

The implementation follows Locatelli's KKT construction.  Vertices and polygon
edges with a strictly convex restriction of ``x*y`` are the generators.  For
each admissible generator set ``J`` of size two or three, the module derives a
cell and the corresponding analytical expression of the envelope.  The domain
may be non-convex; every candidate cell is intersected with the input polygon.

Paper A:
    Polyhedral subdivisions and functional forms for the convex envelopes of
    bilinear, fractional and other bivariate functions over general polytopes.
Paper B:
    Convex envelopes of bivariate functions through the solution of KKT systems.
"""

from dataclasses import dataclass, field
import math
from itertools import combinations, product
from mpmath.libmp.libhyper import NoConvergence
import shapely.geometry as sg

from sympy import Eq, Ge, Gt, linsolve, simplify
import gurobipy as gp
from tqdm import tqdm
import numpy as np
from scipy.spatial import ConvexHull

import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as PolygonPatch

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box
from shapely.ops import split
from shapely.geometry import LineString




def plot_cell(cell_inequalities, polygon):
    """Plot the part of a symbolic candidate cell contained in ``polygon``."""
    x, y = sp.symbols('x y', real=True)

    polygon_vertices = np.array([[v.x, v.y] for v in polygon.vertex_sequence] + [[polygon.vertex_sequence[0].x, polygon.vertex_sequence[0].y]])

    # Use one common grid for the polygon mask and all symbolic inequalities.

    padding = 0.2

    x_min = polygon_vertices[:, 0].min() - padding
    x_max = polygon_vertices[:, 0].max() + padding
    y_min = polygon_vertices[:, 1].min() - padding
    y_max = polygon_vertices[:, 1].max() + padding

    resolution = 800

    x_values = np.linspace(x_min, x_max, resolution)
    y_values = np.linspace(y_min, y_max, resolution)

    X, Y = np.meshgrid(x_values, y_values)

    # Mask of the original, possibly non-convex polygon.

    points = np.column_stack((X.ravel(), Y.ravel()))

    polygon_path = Path(polygon_vertices)

    mask_P = polygon_path.contains_points(
        points,
        radius=scaled_tolerance(*polygon_vertices.ravel())
    ).reshape(X.shape)

    # Mask of the candidate cell.

    mask_C = np.ones(X.shape, dtype=bool)

    for inequality in cell_inequalities:
        # Evaluate the symbolic inequality over the complete plotting grid.
        inequality_function = sp.lambdify(
            (x, y),
            inequality,
            modules="numpy"
        )

        current_mask = inequality_function(X, Y)

        # Constant inequalities produce one Boolean; broadcast it to the grid.
        current_mask = np.broadcast_to(
            np.asarray(current_mask, dtype=bool),
            X.shape
        )

        mask_C &= current_mask

    # Only the portion inside the original domain is a valid envelope cell.
    mask_intersection = mask_C & mask_P

    fig, ax = plt.subplots(figsize=(7, 6))

    # Shade the feasible intersection and show both kinds of boundary.
    ax.contourf(
        X,
        Y,
        mask_intersection.astype(float),
        levels=[0.5, 1.5],
        alpha=0.5
    )

    polygon_patch = PolygonPatch(
        polygon_vertices,
        closed=True,
        fill=False,
        edgecolor="black",
        linewidth=2,
        label=r"$P$"
    )
    ax.add_patch(polygon_patch)

    for inequality in cell_inequalities:
        boundary_expression = inequality.lhs - inequality.rhs
        boundary_function = sp.lambdify(
            (x, y),
            boundary_expression,
            modules="numpy"
        )

        Z = np.asarray(boundary_function(X, Y), dtype=float)
        Z = np.broadcast_to(Z, X.shape)

        ax.contour(
            X,
            Y,
            Z,
            levels=[0],
            linewidths=1.5,
            linestyles="--"
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$C \cap P$")
    ax.grid(alpha=0.2)

    plt.show()


def get_active_point_from_generator(generator, a, b):
    """Return the contact point selected by the supporting slope ``(a, b)``.

    A vertex is constant.  On an edge, the stationary point of
    ``xy - a*x - b*y`` is clamped to the edge endpoints; this is the piecewise
    definition of ``x_k(alpha)`` in equations (5)--(8) of Paper B.
    """
    if isinstance(generator, Vertex):
        x_r = generator.x
        y_r = generator.y
    else:
        x_r = (a + generator.m * b - generator.q) / (2 * generator.m)
        if x_r < generator.v1.x:
            x_r = generator.v1.x
        if x_r > generator.v2.x:
            x_r = generator.v2.x
        y_r = generator.m * x_r + generator.q

    return x_r, y_r


def scaled_tolerance(*values, relative=1e-12):
    """Return a roundoff allowance with the units of ``values``.

    The relative part preserves scale invariance.  The machine-epsilon floor
    covers cancellation near zero without imposing a problem-scale absolute
    tolerance (which previously broke the ``small_scale`` instance).
    """
    scale = max((abs(float(value)) for value in values), default=0.0)
    return max(relative * scale, 64 * np.finfo(float).eps * max(1.0, scale))


def roundoff_tolerance(*values):
    """Return an allowance for floating-point evaluation roundoff only."""
    scale = max((abs(float(value)) for value in values), default=0.0)
    return 64 * np.finfo(float).eps * max(1.0, scale)


def active_point_is_strictly_inside_edge(edge, x_coordinate):
    # A repeated root evaluated by the ten-digit fallback can place an exact
    # endpoint contact roughly 1e-8 edge lengths inside the segment.  Exclude
    # that numerical sliver as the lower-dimensional endpoint case it is.
    edge_length_x = edge.v2.x - edge.v1.x
    tolerance = max(
        ACTIVE_POINT_REL_TOL * edge_length_x,
        roundoff_tolerance(edge.v1.x, edge.v2.x, x_coordinate),
    )
    return edge.v1.x + tolerance < x_coordinate < edge.v2.x - tolerance


def is_numerically_real(root):
    return abs(float(sp.im(root))) <= scaled_tolerance(float(sp.re(root)))


def numerical_polynomial_roots(poly):
    """Compute accurate roots, with a fallback for ill-conditioned multiples.

    ``nroots`` may fail to certify high precision for repeated roots even after
    many iterations.  Ten digits are sufficient for the fallback because every
    returned candidate is subsequently checked against the KKT conditions.
    """
    try:
        return sp.nroots(poly, n=20, maxsteps=1000)
    except NoConvergence:
        return sp.nroots(poly, n=10, maxsteps=1000)


def feasibility_outside_J_generators(generator_in_J, generators_outside_J, a, b):
    """Check the inactive-generator inequalities for a candidate KKT solution.

    All generators in ``J`` have the same eta value.  It therefore suffices to
    compare one active generator with every generator outside ``J``.
    """
    x_k, y_k = get_active_point_from_generator(generator_in_J, a, b)

    for generator_outside_J in generators_outside_J:
        x_r, y_r = get_active_point_from_generator(generator_outside_J, a, b)

        eta_k = float(x_k * y_k - a * x_k - b * y_k)
        eta_r = float(x_r * y_r - a * x_r - b * y_r)
        if eta_k > eta_r + scaled_tolerance(eta_k, eta_r):
            return False

    return True


def satisfies_all(point, inequalities, tolerance=None):
    """Evaluate all cell inequalities at a point with feasibility tolerance.

    Returns ``(False, 'undefined')`` if a rational expression has a denominator
    close to zero, and ``(False, 'violation')`` for an ordinary failed bound.
    This distinction is used by the validation summary.
    """
    if tolerance is None:
        tolerance = FEAS_TOL

    symbols = set().union(
        *(ineq.free_symbols for ineq in inequalities)
    )

    coordinates = {"x": point[0], "y": point[1]}
    substitutions = {
        symbol: coordinates[symbol.name]
        for symbol in symbols
        if symbol.name in coordinates
    }

    for ineq in inequalities:
        expressions = (ineq.lhs, ineq.rhs)

        for expr in expressions:
            denominator = sp.denom(sp.together(expr))

            if abs(float(denominator.subs(substitutions))) <= tolerance:
                return False, 'undefined'

        # Relax lower bounds downward and upper bounds upward.
        bound_tolerance = -tolerance if isinstance(ineq, (Ge, Gt)) else tolerance
        ineq_numeric = ineq.func(ineq.lhs, ineq.rhs + bound_tolerance)
        if not ineq_numeric.subs(substitutions):
            return False, 'violation'


    return True, None


def compute_cells_one_vertex_one_edge(solutions, v, e, inequalities, functional):
    """Convert the remaining one-dimensional KKT bounds into spatial cells.

    Clearing a sign-unknown denominator creates two cases.  The strict sign
    constraint in each case preserves the direction of the multiplied bounds.
    """
    # introduce variables for symbolic computation
    x, y, a, b, z = sp.symbols("x y a b z", real=True)

    # solve 1D inequality system
    sol = sp.reduce_inequalities(inequalities, z)
    sol_set = sol.as_set()

    # handling of solution set
    if sol_set is sp.S.EmptySet:
        return []
    elif isinstance(sol_set, sp.FiniteSet) or isinstance(sol_set, sp.Interval):
        pass
    elif sol_set is sp.S.UniversalSet:
        print('solution is of unexpected type UniversalSet')
        exit()
    elif sol_set is sp.S.Reals:
        print('solution is of unhandled type Reals')
        exit()
    else:
        print('solution is of unhandled type ', type(sol_set))
        exit()

    # build cell inequalities (two subcells occur here)
    cell_inequalities_base = []
    cell_inequalities_base.append((e.m * (x - v.x) + v.y - y) / (v.y - e.q - e.m * v.x) >= 0)
    cell_inequalities_base.append((e.m * (x - v.x) + v.y - y) / (v.y - e.q - e.m * v.x) <= 1)

    if isinstance(sol_set, sp.Interval):
        lower = sol_set.start
        upper = sol_set.end

        cell_inequalities1 = cell_inequalities_base.copy()
        cell_inequalities1.append(e.m * (x - v.x) + v.y - y > 0)
        cell_inequalities1.append(
            2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (v.y - y) >= lower * (
                        e.m * (x - v.x) + v.y - y))
        cell_inequalities1.append(
            2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                    v.y - y) <= upper * (e.m * (x - v.x) + v.y - y))

        solutions.append(((v, e), cell_inequalities1, functional))

        cell_inequalities2 = cell_inequalities_base.copy()
        cell_inequalities2.append(e.m * (x - v.x) + v.y - y < 0)
        cell_inequalities2.append(
            2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                    v.y - y) <= lower * (e.m * (x - v.x) + v.y - y))
        cell_inequalities2.append(
            2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                    v.y - y) >= upper * (e.m * (x - v.x) + v.y - y))

        solutions.append(((v, e), cell_inequalities2, functional))

    elif isinstance(sol_set, sp.FiniteSet):
        if len(sol_set) == 1:
            sol = next(iter(sol_set))
            lower = sol
            upper = sol

            cell_inequalities1 = cell_inequalities_base.copy()
            cell_inequalities1.append(e.m * (x - v.x) + v.y - y > 0)
            cell_inequalities1.append(
                2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                            v.y - y) >= lower * (
                        e.m * (x - v.x) + v.y - y))
            cell_inequalities1.append(
                2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                        v.y - y) <= upper * (e.m * (x - v.x) + v.y - y))


            solutions.append(((v, e), cell_inequalities1, functional))

            cell_inequalities2 = cell_inequalities_base.copy()
            cell_inequalities2.append(e.m * (x - v.x) + v.y - y < 0)
            cell_inequalities2.append(
                2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                        v.y - y) <= lower * (e.m * (x - v.x) + v.y - y))
            cell_inequalities2.append(
                2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                        v.y - y) >= upper * (e.m * (x - v.x) + v.y - y))

            solutions.append(((v, e), cell_inequalities2, functional))

        else:
            print('function compute_cells_one_vertex_one_edge')
            print('solution is of below unhandled type ', type(sol_set), len(sol_set))

    else:
        print('function compute_cells_one_vertex_one_edge')
        print('solution is of below unhandled type ', type(sol_set))
        print(sol_set)
        exit()


def compute_cells_two_edges_equal_m(solutions, e1, e2, beta1, beta0, inequalities, functional):
    """Build the cell for two active edges having the same slope."""
    # introduce variables for symbolic computation
    x, y, a, b = sp.symbols("x y a b", real=True)

    # solve 1D inequality system
    sol = sp.reduce_inequalities(inequalities, b)
    sol_set = sol.as_set()

    # handling of solution set
    if sol_set is sp.S.EmptySet:
        return []
    elif isinstance(sol_set, sp.FiniteSet) or isinstance(sol_set, sp.Interval):
        pass
    elif sol_set is sp.S.UniversalSet:
        print('solution is of unexpected type UniversalSet')
        exit()
    elif sol_set is sp.S.Reals:
        print('solution is of unhandled type Reals')
        exit()
    else:
        print('solution is of unhandled type ', type(sol_set))
        exit()

    # build cell inequalities
    cell_inequalities = []
    cell_inequalities.append((e1.m * x - y + e1.q) / (e1.q - e2.q) >= 0)
    cell_inequalities.append((e1.m * x - y + e1.q) / (e1.q - e2.q) <= 1)

    if isinstance(sol_set, sp.Interval):
        lower = sol_set.start
        upper = sol_set.end

        cell_inequalities.append((y + e1.m * x - beta0) / (2 * e1.m) >= lower)
        cell_inequalities.append((y + e1.m * x - beta0) / (2 * e1.m) <= upper)

        solutions.append(((e1, e2), cell_inequalities, functional))

    elif isinstance(sol_set, sp.FiniteSet):
        if len(sol_set) == 1:
            sol = next(iter(sol_set))
            lower = sol
            upper = sol

            cell_inequalities.append((y + e1.m * x - beta0) / (2 * e1.m) >= lower)
            cell_inequalities.append((y + e1.m * x - beta0) / (2 * e1.m) <= upper)

            solutions.append(((e1, e2), cell_inequalities, functional))
        else:
            print('function compute_cells_two_edges_equal_m')
            print('solution is of below unhandled type ', type(sol_set), len(sol_set))
            exit()


    else:
        print('function compute_cells_two_edges_equal_m')
        print('solution is of below unhandled type ', type(sol_set))
        exit()


def compute_cells_two_edges_non_equal_m(solutions, e1, e2, beta1, beta0, inequalities, functional):
    """Build the two orientation cases for active edges of different slopes."""
    # introduce variables for symbolic computation
    x, y, a, b = sp.symbols("x y a b", real=True)

    # solve 1D inequality system
    sol = sp.reduce_inequalities(inequalities, b)
    sol_set = sol.as_set()

    # handling of solution set
    if sol_set is sp.S.EmptySet:
        return []
    elif isinstance(sol_set, sp.FiniteSet) or isinstance(sol_set, sp.Interval):
        pass
    elif sol_set is sp.S.UniversalSet:
        print('solution is of unexpected type UniversalSet')
        exit()
    elif sol_set is sp.S.Reals:
        print('solution is of unhandled type Reals')
        exit()
    else:
        print('solution is of unhandled type ', type(sol_set))
        exit()

    # build cell inequalities (two subcells occur here)
    cell_inequalities_base = []
    if isinstance(sol_set, sp.Interval):
        lower = sol_set.start
        upper = sol_set.end

        term1 = (2 * math.sqrt(e1.m) * (y + math.sqrt(e1.m * e2.m) * x - e1.q) + (e1.q - beta0) * (
                math.sqrt(e1.m) + math.sqrt(e2.m))) / (
                        (math.sqrt(e1.m) + math.sqrt(e2.m)) * (beta1 + e1.m))
        term2 = (2 * math.sqrt(e2.m) * (y + math.sqrt(e1.m * e2.m) * x - e2.q) + (e2.q - beta0) * (
                math.sqrt(e1.m) + math.sqrt(e2.m))) / (
                        (math.sqrt(e1.m) + math.sqrt(e2.m)) * (beta1 + e2.m))

        cell_inequalities_base.append(term1 >= lower)
        cell_inequalities_base.append(term1 <= upper)
        cell_inequalities_base.append(term2 >= lower)
        cell_inequalities_base.append(term2 <= upper)

    else:
        print('solution is of unhandled type ', type(sol_set))

    x_i = (y + math.sqrt(e1.m * e2.m) * x - e1.q) / (math.sqrt(e1.m) * (math.sqrt(e1.m) + math.sqrt(e2.m)))
    x_j = (y + math.sqrt(e1.m * e2.m) * x - e2.q) / (math.sqrt(e2.m) * (math.sqrt(e1.m) + math.sqrt(e2.m)))

    cell_inequalities1 = cell_inequalities_base.copy()
    cell_inequalities1.append(x_j > x_i)
    cell_inequalities1.append(x - x_i >= 0)
    cell_inequalities1.append(x - x_i <= x_j - x_i)
    solutions.append(((e1, e2), cell_inequalities1, functional))

    cell_inequalities2 = cell_inequalities_base.copy()
    cell_inequalities2.append(x_j < x_i)
    cell_inequalities2.append(x - x_i <= 0)
    cell_inequalities2.append(x - x_i >= x_j - x_i)
    solutions.append(((e1, e2), cell_inequalities2, functional))


def add_domain_inequalities(inequalities, edge, domain, substitutions):
    """Add the inequalities selecting one branch of an edge generator.

    ``domain`` is ``-1`` at the left endpoint, ``0`` on the stationary middle
    segment, and ``1`` at the right endpoint.  Substitutions eliminate the KKT
    variables that were solved in the current generator case.
    """
    # introduce variables for symbolic computation
    a, b, z = sp.symbols('a b z', real=True)

    # (21) in Paper B
    s_e = (a + edge.m * b - edge.q) / (2 * edge.m)

    # add inequality depending on domain
    if domain == 0:
        ineq = s_e >= edge.v1.x

        for key, value in substitutions.items():
            ineq = ineq.subs(key, value)
        inequalities.append(ineq)

        ineq = s_e <= edge.v2.x
        for key, value in substitutions.items():
            ineq = ineq.subs(key, value)
        inequalities.append(ineq)
    elif domain == -1:
        ineq = s_e <= edge.v1.x
        for key, value in substitutions.items():
            ineq = ineq.subs(key, value)
        inequalities.append(ineq)
    else:
        ineq = s_e >= edge.v2.x
        for key, value in substitutions.items():
            ineq = ineq.subs(key, value)
        inequalities.append(ineq)


def add_eta_inequalities_for_vertices_outside_J(inequalities, edge, vertices_without_pair, substitutions):
    """Require every outside-vertex eta value to be at least the active value."""
    # introduce variables for symbolic computation
    a, b, z = sp.symbols('a b z', real=True)

    # (21), (22) in Paper B
    s_e = (a + edge.m * b - edge.q) / (2 * edge.m)
    eta_e = -edge.m * (s_e) ** 2 - b * edge.q

    # add for each vertex r outside J the inequality eta_e <= eta_r (including variable substitutions)
    for r in vertices_without_pair:
        eta_r = r.x * r.y - a * r.x - b * r.y

        ineq = eta_e <= eta_r
        for key, value in substitutions.items():
            ineq = ineq.subs(key, value)
        inequalities.append(ineq)


def add_eta_inequalities_for_edges_outside_J(inequalities, edge, edges_without_pair, domains, substitutions):
    """Add eta comparisons for outside edges in their stationary domain.

    In endpoint domains the outside edge has the same active point and eta
    value as one of its vertices.  The vertex inequalities already impose that
    condition, so adding it again is redundant and can amplify roundoff.
    """
    # introduce variables for symbolic computation
    a, b, z = sp.symbols('a b z', real=True)

    # (21), (22) in Paper B; edge belongs to J and is in its middle domain
    s_e = (a + edge.m * b - edge.q) / (2 * edge.m)
    eta_e = -edge.m * s_e ** 2 - b * edge.q

    # Endpoint minima are already covered by vertex inequalities or active
    # equalities in J. Repeating them can turn float roundoff into false bounds.
    for r, domain in zip(edges_without_pair, domains):
        if domain != 0:
            continue

        s_r = (a + r.m * b - r.q) / (2 * r.m)
        eta_r = -r.m * s_r ** 2 - b * r.q

        ineq = eta_e <= eta_r
        for key, value in substitutions.items():
            ineq = ineq.subs(key, value)
        inequalities.append(ineq)


def has_2d_intersection(vertex_sequence, inequalities):
    """Return whether a linear cell intersects the polygon with positive area.

    Each inequality is converted to a half-plane and clipped against the
    possibly non-convex polygon.  Lower-dimensional contacts are deliberately
    discarded because the envelope on their boundary follows by continuity.
    """
    x, y = sp.symbols('x y', real=True)
    polygon = ShapelyPolygon([(float(v.x), float(v.y)) for v in vertex_sequence])

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if polygon.is_empty:
        return False

    intersection = polygon

    # Bounding box etwas vergrößern.
    minx, miny, maxx, maxy = polygon.bounds

    scale = max(maxx - minx, maxy - miny)
    if scale == 0:
        return False
    margin = 10 * scale
    area_tolerance = scaled_tolerance(scale * scale)

    bounding_box = box(
        minx - margin,
        miny - margin,
        maxx + margin,
        maxy + margin,
    )

    for inequality in inequalities:

        # Normalize the boundary as lhs - rhs = 0.
        expr = sp.expand(inequality.lhs - inequality.rhs)

        poly = sp.Poly(expr, x, y)

        if poly.total_degree() > 1:
            raise ValueError(
                f"Ungleichung ist nicht linear: {inequality}"
            )

        a = float(expr.coeff(x))
        b = float(expr.coeff(y))
        c = float(expr.subs({x: 0, y: 0}))

        # A constant inequality either keeps or removes the whole intersection.
        if abs(a) <= scaled_tolerance(a, b) and abs(b) <= scaled_tolerance(a, b):
            # Ungleichung enthält gar kein x oder y.
            if not bool(inequality):
                return False
            continue

        # Construct a line long enough to split the enlarged bounding box.
        if abs(b) > abs(a):
            x1 = minx - margin
            x2 = maxx + margin

            y1 = -(a * x1 + c) / b
            y2 = -(a * x2 + c) / b

        else:
            y1 = miny - margin
            y2 = maxy + margin

            x1 = -(b * y1 + c) / a
            x2 = -(b * y2 + c) / a

        line = LineString([
            (x1, y1),
            (x2, y2),
        ])

        # Bounding Box durch die Gerade teilen.
        pieces = split(bounding_box, line)

        valid_pieces = []

        for piece in pieces.geoms:
            p = piece.representative_point()

            value = a * p.x + b * p.y + c

            if isinstance(
                inequality,
                (sp.core.relational.LessThan,
                 sp.core.relational.StrictLessThan)
            ):
                valid = value <= scaled_tolerance(a * p.x, b * p.y, c)

            elif isinstance(
                inequality,
                (sp.core.relational.GreaterThan,
                 sp.core.relational.StrictGreaterThan)
            ):
                valid = value >= -scaled_tolerance(a * p.x, b * p.y, c)

            else:
                raise ValueError(
                    f"Nicht unterstützte Bedingung: {inequality}"
                )

            if valid:
                valid_pieces.append(piece)

        if not valid_pieces:
            return False

        halfplane = valid_pieces[0]

        intersection = intersection.intersection(halfplane)

        if intersection.is_empty:
            return False

        if intersection.area <= area_tolerance:
            return False

    return intersection.area > area_tolerance

@dataclass(frozen=True)
class Vertex:
    """A polygon vertex and a constant generator in Locatelli's construction."""
    x: float
    y: float


@dataclass(frozen=True)
class Edge:
    """A polygon edge represented by ``y = m*x + q`` when non-vertical."""
    v1: Vertex
    v2: Vertex
    m: float = field(init=False)
    q: float = field(init=False)

    def __post_init__(self):
        # Ordering endpoints by x makes the three edge domains unambiguous.
        if self.v1.x > self.v2.x:
            tmp = self.v1
            object.__setattr__(self, "v1", self.v2)
            object.__setattr__(self, "v2", tmp)

        if self.v1.x == self.v2.x:
            # Vertical edges are linear restrictions of x*y and are therefore
            # not strictly convex edge generators for this particular function.
            m = math.nan
            q = math.nan
        else:
            m = (self.v2.y - self.v1.y) / (self.v2.x - self.v1.x)
            q = self.v1.y - m * self.v1.x

        object.__setattr__(self, "m", m)
        object.__setattr__(self, "q", q)


def vertex_lies_on_edge_line(vertex, edge):
    """Return whether ``vertex`` lies on the supporting line of ``edge``."""
    slope_term = edge.m * vertex.x
    lhs = slope_term + edge.q
    return abs(lhs - vertex.y) <= roundoff_tolerance(slope_term, edge.q, vertex.y)


@dataclass(frozen=True)
class Polygon:
    """Polygon boundary data; ``vertex_sequence`` preserves cyclic adjacency."""
    vertex_sequence: tuple[Vertex, ...]
    vertices: frozenset[Vertex] = field(init=False)
    edges: frozenset[Edge] = field(init=False)
    is_ortho: bool = field(init=False)


    def __post_init__(self):
        vertices = frozenset(self.vertex_sequence)
        object.__setattr__(self, "vertices", vertices)

        n = len(self.vertices)
        edges = frozenset(
            Edge(self.vertex_sequence[i], self.vertex_sequence[(i + 1) % n])
            for i in range(n)
        )
        object.__setattr__(self, "edges", edges)

        is_ortho = True
        for e in edges:
            if e.v1.x != e.v2.x and e.v1.y != e.v2.y:
                is_ortho = False
                break
        object.__setattr__(self, "is_ortho", is_ortho)


class EnvelopePolygonalDomain:
    """Derive, plot, and validate all envelope cells."""

    def __init__(self, polygon: Polygon):
        self.polygon = polygon
        self.generators = self._determine_generators()

        solutions_two_elements_J = self._two_elements_J()
        solutions_three_elements_J = self._three_elements_J()

        self.all_solutions = solutions_two_elements_J + solutions_three_elements_J

        for _, cell_inequalities, _ in self.all_solutions:
            plot_cell(cell_inequalities, self.polygon)

        self._validation()

    def _validation(self):
        """Compare analytical cells with a discretized lifted convex hull.

        A normalized lifted sampling of the polygon only approximates the true
        reference, so the reported errors depend on
        ``VALIDATION_DISCRETIZATION``.
        Cell coverage, invalid functional values, and disagreements between
        overlapping cells are checked independently of that reference error.
        """
        # read discretization and polygon bounds
        dx = VALIDATION_DISCRETIZATION
        dy = VALIDATION_DISCRETIZATION
        minx = min(v.x for v in self.polygon.vertices)
        miny = min(v.y for v in self.polygon.vertices)
        maxx = max(v.x for v in self.polygon.vertices)
        maxy = max(v.y for v in self.polygon.vertices)
        x_span = maxx - minx
        y_span = maxy - miny
        coordinate_span = max(abs(x_span), abs(y_span))
        validation_tolerance = max(
            DEGENERACY_TOL * coordinate_span,
            roundoff_tolerance(minx, miny, maxx, maxy),
        )
        shapely_polygon = sg.Polygon((v.x, v.y) for v in self.polygon.vertex_sequence)

        # construct grid and store all 3D points over polygon
        x_vals = np.arange(minx, maxx + dx, dx)
        y_vals = np.arange(miny, maxy + dy, dy)
        evaluation_points = []
        for x in x_vals:
            for y in y_vals:
                p = sg.Point(x, y)
                if shapely_polygon.covers(p):
                    evaluation_points.append([p.x, p.y, p.x * p.y])
        evaluation_points = np.array(evaluation_points)

        # A Cartesian grid generally misses sloped boundaries and may even miss
        # vertices when a side length is not an integer multiple of the step.
        # Include a step-size-controlled sampling of every polygon edge in the
        # lifted reference hull.
        def normalized_lift(x_coordinate, y_coordinate):
            normalized_x = (x_coordinate - minx) / x_span
            normalized_y = (y_coordinate - miny) / y_span
            return [normalized_x, normalized_y, normalized_x * normalized_y]

        reference_points = [
            normalized_lift(x_coordinate, y_coordinate)
            for x_coordinate, y_coordinate, _ in evaluation_points
        ]
        for edge in self.polygon.edges:
            x_distance = abs(edge.v2.x - edge.v1.x)
            y_distance = abs(edge.v2.y - edge.v1.y)
            number_of_steps = max(1, math.ceil(max(x_distance / dx, y_distance / dy)))
            for step in range(number_of_steps + 1):
                weight = step / number_of_steps
                x_coordinate = edge.v1.x + weight * (edge.v2.x - edge.v1.x)
                y_coordinate = edge.v1.y + weight * (edge.v2.y - edge.v1.y)
                reference_points.append(normalized_lift(x_coordinate, y_coordinate))
        reference_points = np.unique(np.asarray(reference_points), axis=0)

        # The lower surface of this 3D hull is the discretized reference
        # envelope.  Finer grids improve it but increase the LP workload.
        hull = ConvexHull(reference_points)

        # for each 2D point from polygon compute minimal z-coordinate in approximate convex hull and
        # compare to computed analytical result
        hull_ineqs = hull.equations  # consider hull facets
        error = 0
        maximum_absolute_error = 0
        compared_points = 0
        minimum_reference_value = math.inf
        maximum_reference_value = -math.inf
        undefined_points = 0
        infeasible_points = 0
        largest_cell_difference = 0
        has_multiple_valid_cells = False
        nan_counter = 0
        invalid_value_counter = 0
        # Reuse one environment and model for all grid points.
        with gp.Env(empty=True) as env:
            env.setParam("LogToConsole", 0)
            env.start()

            with gp.Model(env=env) as model:
                x_var = model.addVar(lb=0, ub=1)
                y_var = model.addVar(lb=0, ub=1)
                z_var = model.addVar(lb=-math.inf)

                x_fix = model.addConstr(x_var == 0)
                y_fix = model.addConstr(y_var == 0)

                for hull_ineq in hull_ineqs:
                    model.addConstr(
                        hull_ineq[0] * x_var
                        + hull_ineq[1] * y_var
                        + hull_ineq[2] * z_var
                        + hull_ineq[3]
                        <= 0
                    )

                model.setObjective(z_var, gp.GRB.MINIMIZE)

                for x0, y0, _ in tqdm(evaluation_points):
                    x_fix.RHS = (x0 - minx) / x_span
                    y_fix.RHS = (y0 - miny) / y_span
                    model.optimize()

                    if model.status != gp.GRB.Status.OPTIMAL:
                        raise RuntimeError(
                            f"Validation failed at ({x0}, {y0}): "
                            f"Gurobi status {model.status}"
                        )

                    # store obtained minimal z
                    # Undo the normalized affine coordinate transformation:
                    # xy = (x-minx)(y-miny) + miny(x-minx)
                    #      + minx(y-miny) + minx*miny.
                    z_manual = (
                        z_var.X * x_span * y_span
                        + miny * (x0 - minx)
                        + minx * (y0 - miny)
                        + minx * miny
                    )

                    # Evaluate every covering analytical cell.  Overlap is
                    # allowed on boundaries, but all resulting values must agree.
                    feasible_cell_counter = 0
                    has_undefined_cell = False
                    has_invalid_functional_value = False
                    z_list = []


                    for generators, cell_inequalities, functional in self.all_solutions:
                        feasible, msg = satisfies_all(
                            (x0, y0), cell_inequalities, tolerance=validation_tolerance
                        )

                        # if fixed point is within considered cell region then we can compute analytical z accordingly
                        if feasible:
                            feasible_cell_counter += 1
                            coordinates = {"x": x0, "y": y0}
                            substitutions = {
                                symbol: coordinates[symbol.name]
                                for symbol in functional.free_symbols
                                if symbol.name in coordinates
                            }
                            z_analytical = functional.subs(substitutions).evalf()
                            if z_analytical is sp.nan:
                                nan_counter += 1

                            if z_analytical.is_finite is not True or z_analytical.is_real is not True:
                                invalid_value_counter += 1
                                has_invalid_functional_value = True
                                continue

                            z_list.append(z_analytical)


                        elif msg == 'undefined':
                            has_undefined_cell = True

                    if has_invalid_functional_value:
                        # A valid value from another overlapping cell must not
                        # hide a singular or non-real functional evaluation.
                        undefined_points += 1
                        continue

                    if not z_list:
                        # No analytical value can mean either definite lack of
                        # cell coverage or an undecidable symbolic evaluation.
                        if feasible_cell_counter == 0 and not has_undefined_cell:
                            infeasible_points += 1
                        else:
                            undefined_points += 1
                        continue

                    if len(z_list) > 1:
                        # This diagnostic checks that alternative active sets
                        # define the same envelope value on their intersection.
                        cell_difference = max(z_list) - min(z_list)
                        has_multiple_valid_cells = True
                        largest_cell_difference = max(largest_cell_difference, cell_difference)
                    # Use the last valid value, not a potentially discarded NaN.
                    z_analytical = z_list[-1]
                    # sum up absolute differences to error
                    absolute_error = abs(z_manual - z_analytical)
                    error += absolute_error
                    maximum_absolute_error = max(maximum_absolute_error, absolute_error)
                    compared_points += 1
                    minimum_reference_value = min(minimum_reference_value, z_manual)
                    maximum_reference_value = max(maximum_reference_value, z_manual)


        print('Numbers of undefined and infeasible (but not undefined) points are: ', undefined_points, infeasible_points)
        print("Number of compared grid points is: ", compared_points)
        if compared_points > 0:
            average_absolute_error = error / compared_points
            reference_value_range = maximum_reference_value - minimum_reference_value
            print("Average absolute error for each compared grid point is: ", average_absolute_error)
            print("Maximum absolute error over compared grid points is: ", maximum_absolute_error)
            print("Reference value range for compared grid points is: ", reference_value_range)
            if reference_value_range > 0:
                print(
                    "Relative average absolute error for each compared grid point is: ",
                    average_absolute_error / reference_value_range,
                )
                print(
                    "Relative maximum absolute error over compared grid points is: ",
                    maximum_absolute_error / reference_value_range,
                )
            else:
                print("Relative average absolute error is undefined because the reference range is zero.")
        else:
            print("No evaluable grid points; average absolute error is undefined.")
        print("Number of nan functional values is: ", nan_counter)
        print("Number of non-finite or non-real functional values is: ", invalid_value_counter)
        if has_multiple_valid_cells:
            print("Largest difference between analytical cell values is: ", largest_cell_difference)
        else:
            print("No grid point with multiple valid cell values found.")


    def _two_elements_J(self):
        """Enumerate active sets ``J`` containing two generators."""
        pairs = list(combinations(self.generators, 2))

        all_solutions_two_elements_J = []
        for pair in pairs:
            number_of_vertices = sum(isinstance(el, Vertex) for el in pair)
            generators_without_pair = [el for el in self.generators if el not in pair]

            solutions = []
            if number_of_vertices == 1:
                solutions = self._solve_one_vertex_one_edge(pair, generators_without_pair)
            if number_of_vertices == 0:
                solutions = self._solve_two_edges(pair, generators_without_pair)

            all_solutions_two_elements_J += solutions

        all_solutions_two_elements_J = [solution for solution in all_solutions_two_elements_J if has_2d_intersection(self.polygon.vertex_sequence, solution[1])]

        print('all solutions stemming from Js with two elements:')
        for sol in all_solutions_two_elements_J:
            print(sol)
        print('\n')
        print(100 * '#')

        return all_solutions_two_elements_J


    def _solve_one_vertex_one_edge(self, pair, generators_without_pair):
        """Create solutions for J being one vertex and one convex edge generator."""
        # extract vertex and edge from input pair
        v = tuple(el for el in pair if isinstance(el, Vertex))[0]
        e = tuple(el for el in pair if isinstance(el, Edge))[0]

        # The vertex cannot lie on the line containing the edge.  A tolerant
        # comparison is required because reconstructing the line from decimal
        # endpoints may not reproduce an endpoint's y-coordinate exactly.
        if vertex_lies_on_edge_line(v, e):
            return []

        # introduce variables for symbolic computation
        x, y = sp.symbols("x y", real=True)

        # compute a, b and functional as in Appendix A.2, Paper B
        x_j = (x * (v.y - e.q) - v.x * (y - e.q)) / (e.m * (x - v.x) + v.y - y)
        b = (e.m * x_j * x_j - 2 * e.m * v.x * x_j - e.q * v.x + v.x * v.y) / (v.y - e.m * v.x - e.q)
        a = 2 * e.m * x_j + e.q - e.m * b
        functional = v.x * v.y + a * (x - v.x) + b * (y - v.y)
        functional = simplify(functional)

        # the return object solutions contains the whole polyhedral subdivision, the cells are computed in the following
        solutions = []

        # define variable subsitutions to simplify inequality systems, cf. Section 5.2, Paper A
        a, b, z = sp.symbols("a b z", real=True)
        C = 4 * e.m * e.q + 4 * e.m * e.m * v.x - 4 * e.m * v.y
        beta2 = (-1) / C
        beta1 = (2 * e.q + 4 * e.m * v.x) / C
        beta0 = (e.q * e.q + 4 * e.m * v.x * v.y) / (-C)
        substitutions = {a: z - e.m * b, b: beta2 * z ** 2 + beta1 * z + beta0}

        # add middle-domain inequalities for edge within J
        base_inequalities = []
        add_domain_inequalities(base_inequalities, e, 0, substitutions)

        # add eta inequalities for vertices outside J
        vertices_without_pair = [el for el in generators_without_pair if isinstance(el, Vertex)]
        add_eta_inequalities_for_vertices_outside_J(base_inequalities, e, vertices_without_pair, substitutions)

        # Each inactive edge is piecewise defined on three alpha-domains, so
        # every domain combination represents a separate symbolic case.
        edges_without_pair = [el for el in generators_without_pair if isinstance(el, Edge)]
        if len(edges_without_pair) == 0: # if there is no edge outside of J
            compute_cells_one_vertex_one_edge(solutions, v, e, base_inequalities, functional)
        else: # if there are edges outside of J
            domain_combinations = list(product([-1, 0, 1], repeat=len(edges_without_pair)))
            for combination in domain_combinations:
                extended_inequalities = base_inequalities.copy()
                # add domain inequalities for edges outside of J
                for ind, r in enumerate(edges_without_pair):
                    add_domain_inequalities(extended_inequalities, r, combination[ind], substitutions)

                add_eta_inequalities_for_edges_outside_J(
                    extended_inequalities, e, edges_without_pair, combination, substitutions)

                compute_cells_one_vertex_one_edge(solutions, v, e, extended_inequalities, functional)

        return solutions


    def _solve_two_edges(self, pair, generators_without_pair):
        """Solve system (15) for two convex edge generators."""
        # extract two edges from input pair
        e1 = pair[0]
        e2 = pair[1]

        # the edges cannot lie on the same line
        if e1.m == e2.m and e1.q == e2.q:
            return []

        # introduce variables for symbolic computation
        x, y = sp.symbols("x y", real=True)

        # compute a, b and functional as in Appendix A.3, Paper B
        if e1.m == e2.m:
            x_i = (y + e1.m * x - e1.q) / (2 * e1.m)
            b = x_i + (e1.q - e2.q) / 4
            a = 2 * e1.m * x_i - e1.m * b + e1.q
        else:
            x_i = (y + math.sqrt(e1.m * e2.m) * x - e1.q) / (math.sqrt(e1.m) * (math.sqrt(e1.m) + math.sqrt(e2.m)))
            x_j = (y + math.sqrt(e1.m * e2.m) * x - e2.q) / (math.sqrt(e2.m) * (math.sqrt(e1.m) + math.sqrt(e2.m)))
            b = (2 * e2.m * x_j + e2.q - 2 * e1.m * x_i - e1.q) / (e2.m - e1.m)
            a = 2 * e2.m * x_j + e2.q - e2.m * b

        functional = -e1.m * x_i ** 2 - b * e1.q + a * x + b * y
        functional = simplify(functional)

        # the return object solutions contains the whole polyhedral subdivision, the cells are computed in the following
        solutions = []

        # Parallel edges have one KKT branch.  Non-parallel edges yield the two
        # signs of sqrt(m1*m2) described in Lemma 5.1 of Paper A.
        a, b = sp.symbols("a b", real=True)
        if e1.m == e2.m:
            # define variable subsitutions to simplify inequality systems, cf. Lemma 5.1 and Section 5.1, Paper A
            beta1 = e1.m
            beta0 = (e1.q + e2.q) / 2
            substitutions = {a: beta1 * b + beta0}

            # add middle-domain inequalities for both edges within J
            base_inequalities = []
            add_domain_inequalities(base_inequalities, e1, 0, substitutions)
            add_domain_inequalities(base_inequalities, e2, 0, substitutions)

            # add eta inequalities for vertices outside J
            vertices_without_pair = [el for el in generators_without_pair if isinstance(el, Vertex)]
            add_eta_inequalities_for_vertices_outside_J(base_inequalities, e1, vertices_without_pair, substitutions)

            # consider edges outside J and introduce case distinction
            edges_without_pair = [el for el in generators_without_pair if isinstance(el, Edge)]
            if len(edges_without_pair) == 0:  # if there is no edge outside of J
                compute_cells_two_edges_equal_m(solutions, e1, e2, beta1, beta0, base_inequalities, functional)
            else:  # if there are edges outside of J
                domain_combinations = list(product([-1, 0, 1], repeat=len(edges_without_pair)))
                for combination in domain_combinations:
                    extended_inequalities = base_inequalities.copy()
                    # add domain inequalities for edges outside of J
                    for ind, r in enumerate(edges_without_pair):
                        add_domain_inequalities(extended_inequalities, r, combination[ind], substitutions)

                    add_eta_inequalities_for_edges_outside_J(
                        extended_inequalities, e1, edges_without_pair, combination, substitutions)

                    compute_cells_two_edges_equal_m(solutions, e1, e2, beta1, beta0, extended_inequalities, functional)

        else: # here we need to subdifferentiate for the two solutions stated in Lemma 5.1, Paper A
            # define variable subsitutions to simplify inequality systems, cf. Lemma 5.1 and Section 5.1, Paper A
            beta1 = math.sqrt(e1.m * e2.m)
            beta0 = (e2.q * e1.m - e1.q * e2.m + beta1 * e1.q - beta1 * e2.q) / (
                    e1.m - e2.m
            )
            substitutions = {a: beta1 * b + beta0}

            # add middle-domain inequalities for both edges within J
            base_inequalities = []
            add_domain_inequalities(base_inequalities, e1, 0, substitutions)
            add_domain_inequalities(base_inequalities, e2, 0, substitutions)

            # add eta inequalities for vertices outside J
            vertices_without_pair = [el for el in generators_without_pair if isinstance(el, Vertex)]
            add_eta_inequalities_for_vertices_outside_J(base_inequalities, e1, vertices_without_pair, substitutions)

            # consider edges outside J and introduce case distinction
            edges_without_pair = [el for el in generators_without_pair if isinstance(el, Edge)]
            if len(edges_without_pair) == 0:  # if there is no edge outside of J
                compute_cells_two_edges_non_equal_m(solutions, e1, e2, beta1, beta0, base_inequalities, functional)
            else:  # if there are edges outside of J
                domain_combinations = list(product([-1, 0, 1], repeat=len(edges_without_pair)))
                for combination in domain_combinations:
                    extended_inequalities = base_inequalities.copy()
                    # add domain inequalities for edges outside of J
                    for ind, r in enumerate(edges_without_pair):
                        add_domain_inequalities(extended_inequalities, r, combination[ind], substitutions)

                    add_eta_inequalities_for_edges_outside_J(
                        extended_inequalities, e1, edges_without_pair, combination, substitutions)

                    compute_cells_two_edges_non_equal_m(solutions, e1, e2, beta1, beta0, extended_inequalities, functional)

            # define variable subsitutions to simplify inequality systems, cf. Lemma 5.1 and Section 5.1, Paper A
            beta1 = -math.sqrt(e1.m * e2.m)
            beta0 = (e2.q * e1.m - e1.q * e2.m + beta1 * e1.q - beta1 * e2.q) / (
                    e1.m - e2.m
            )
            substitutions = {a: beta1 * b + beta0}

            # add middle-domain inequalities for both edges within J
            base_inequalities = []
            add_domain_inequalities(base_inequalities, e1, 0, substitutions)
            add_domain_inequalities(base_inequalities, e2, 0, substitutions)

            # add eta inequalities for vertices outside J
            vertices_without_pair = [el for el in generators_without_pair if isinstance(el, Vertex)]
            add_eta_inequalities_for_vertices_outside_J(base_inequalities, e1, vertices_without_pair, substitutions)

            # consider edges outside J and introduce case distinction
            edges_without_pair = [el for el in generators_without_pair if isinstance(el, Edge)]
            if len(edges_without_pair) == 0:  # if there is no edge outside of J
                compute_cells_two_edges_non_equal_m(solutions, e1, e2, beta1, beta0, base_inequalities, functional)
            else:  # if there are edges outside of J
                domain_combinations = list(product([-1, 0, 1], repeat=len(edges_without_pair)))
                for combination in domain_combinations:
                    extended_inequalities = base_inequalities.copy()
                    # add domain inequalities for edges outside of J
                    for ind, r in enumerate(edges_without_pair):
                        add_domain_inequalities(extended_inequalities, r, combination[ind], substitutions)

                    add_eta_inequalities_for_edges_outside_J(
                        extended_inequalities, e1, edges_without_pair, combination, substitutions)

                    compute_cells_two_edges_non_equal_m(solutions, e1, e2, beta1, beta0, extended_inequalities,
                                                        functional)

        return solutions


    def _solve_three_vertices(self, triple, generators_without_triple):
        """Build an affine cell supported by three non-collinear vertices."""
        v1, v2, v3 = triple
        # todo
        # Potential future reduction: discard triples ruled out by convex
        # connections between their vertices before solving the KKT system.

        x_span = max(v1.x, v2.x, v3.x) - min(v1.x, v2.x, v3.x)
        y_span = max(v1.y, v2.y, v3.y) - min(v1.y, v2.y, v3.y)
        if x_span == 0 or y_span == 0:
            return []
        A = np.array([
            [(v1.x - min(v1.x, v2.x, v3.x)) / x_span,
             (v1.y - min(v1.y, v2.y, v3.y)) / y_span, 1],
            [(v2.x - min(v1.x, v2.x, v3.x)) / x_span,
             (v2.y - min(v1.y, v2.y, v3.y)) / y_span, 1],
            [(v3.x - min(v1.x, v2.x, v3.x)) / x_span,
             (v3.y - min(v1.y, v2.y, v3.y)) / y_span, 1],
            ])

        if abs(np.linalg.det(A)) <= DEGENERACY_TOL:
            # Collinear contacts span no two-dimensional cell.
            return []

        a, b, c = sp.symbols("a, b, c", real=True)

        eqs = [
            Eq(a * v1.x + b * v1.y + c, v1.x * v1.y),
            Eq(a * v2.x + b * v2.y + c, v2.x * v2.y),
            Eq(a * v3.x + b * v3.y + c, v3.x * v3.y)
        ]

        a, b, c = list(linsolve(eqs, (a, b, c)))[0]

        if not feasibility_outside_J_generators(v1, generators_without_triple, a, b):
            return []


        cell_ineq_coeffs = ConvexHull([(v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y)]).equations
        x, y = sp.symbols("x y", real=True)
        cell_inequalities = [row[0]*x + row[1]*y + row[2] <= 0 for row in cell_ineq_coeffs]

        functional = a * x + b * y + c
        solution = [(triple, cell_inequalities, functional)]

        return solution


    def _solve_two_vertices_one_edge(self, triple, generators_without_triple):
        """Find affine KKT supports touching two vertices and one edge."""
        v1, v2 = (el for el in triple if isinstance(el, Vertex))
        e = tuple(el for el in triple if isinstance(el, Edge))[0]


        if vertex_lies_on_edge_line(v2, e):
            return []

        C = 4 * e.m * (e.q + e.m * v2.x - v2.y)
        beta2 = -1 / C
        beta1 = (2 * e.q + 4 * e.m * v2.x) / C
        beta0 = -(4 * e.m * v2.x * v2.y + e.q ** 2) / C

        z = sp.symbols("z", real=True)

        b = beta2 * z ** 2 + beta1 * z + beta0
        a = z - e.m * b

        expr = sp.expand(
            v1.x * v1.y
            - a * v1.x
            - b * v1.y
            - v2.x * v2.y
            + a * v2.x
            + b * v2.y
        )

        poly = sp.Poly(sp.expand(expr), z)
        # The contact equations reduce to a univariate polynomial.  Only real
        # roots whose edge contact lies strictly inside the edge are admissible.
        roots = numerical_polynomial_roots(poly)

        sol = [
            float(sp.re(r))
            for r in roots
            if is_numerically_real(r)
        ]

        solutions = []
        for z in sol:
            b = beta2 * z ** 2 + beta1 * z + beta0
            a = z - e.m * b
            c = v1.x * v1.y - a * v1.x - b * v1.y

            if not active_point_is_strictly_inside_edge(e, (a + e.m * b - e.q) / (2 * e.m)):
                continue

            if not feasibility_outside_J_generators(v1, generators_without_triple, a, b):
                continue

            cell_ineq_coeffs = ConvexHull(
                [(v1.x, v1.y), (v2.x, v2.y), (get_active_point_from_generator(e, a, b))]).equations
            x, y = sp.symbols("x y", real=True)
            cell_inequalities = [row[0] * x + row[1] * y + row[2] <= 0 for row in cell_ineq_coeffs]

            functional = a * x + b * y + c
            solutions.append((triple, cell_inequalities, functional))

        return solutions


    def _solve_one_vertex_two_edges(self, triple, generators_without_triple):
        """Find affine KKT supports touching one vertex and two edges."""
        v = tuple(el for el in triple if isinstance(el, Vertex))[0]
        e1, e2 = (el for el in triple if isinstance(el, Edge))

        if e1.m == e2.m and e1.q == e2.q:
            return []

        if vertex_lies_on_edge_line(v, e1):
            return []

        C = 4 * e1.m * (e1.q + e1.m * v.x - v.y)
        beta2 = -1 / C
        beta1 = (2 * e1.q + 4 * e1.m * v.x) / C
        beta0 = (4 * e1.m * v.x * v.y + e1.q * e1.q) / (-C)

        z = sp.symbols("z", real=True)

        b = beta2 * z ** 2 + beta1 * z + beta0
        a = z - e1.m * b

        expr = (
                v.x * v.y
                - a * v.x
                - b * v.y
                + (a + e2.m * b - e2.q) ** 2 / (4 * e2.m)
                + b * e2.q
        )
        poly = sp.Poly(sp.expand(expr), z)
        roots = numerical_polynomial_roots(poly)

        sol = [
            float(sp.re(r))
            for r in roots
            if is_numerically_real(r)
        ]

        solutions = []
        for z in sol:
            b = beta2 * z ** 2 + beta1 * z + beta0
            a = z - e1.m * b
            c = v.x * v.y - a * v.x - b * v.y

            if not active_point_is_strictly_inside_edge(e1, (a + e1.m * b - e1.q) / (2 * e1.m)):
                continue

            if not active_point_is_strictly_inside_edge(e2, (a + e2.m * b - e2.q) / (2 * e2.m)):
                continue

            if not feasibility_outside_J_generators(v, generators_without_triple, a, b):
                continue

            cell_ineq_coeffs = ConvexHull([(v.x, v.y), (get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b))]).equations
            x, y = sp.symbols("x y", real=True)
            cell_inequalities = [row[0] * x + row[1] * y + row[2] <= 0 for row in cell_ineq_coeffs]

            functional = a * x + b * y + c
            solutions.append((triple, cell_inequalities, functional))

        return solutions


    def _solve_three_edges(self, triple, generators_without_triple):
        """Find affine KKT supports touching three convex edge generators."""
        e1, e2, e3 = triple

        if e1.m == e2.m and e1.q == e2.q:
            return []
        if e1.m == e3.m and e1.q == e3.q:
            return []
        if e2.m == e3.m and e2.q == e3.q:
            return []

        if e1.m == e2.m:
            b = sp.symbols("b", real=True)
            a = e1.m * b + (e1.q + e2.q) / 2

            expr = (a + e3.m * b - e3.q)**2 / (4 * e3.m) + b * e3.q - (a + e2.m * b - e2.q)**2 / (4 * e2.m) - b * e2.q
            poly = sp.Poly(sp.expand(expr), b)
            roots = numerical_polynomial_roots(poly)

            sol = [
                float(sp.re(r))
                for r in roots
                if is_numerically_real(r)
            ]


            solutions = []
            for b in sol:
                a = e1.m * b + (e1.q + e2.q) / 2
                x_r, y_r = get_active_point_from_generator(e1, a, b)
                c = x_r * y_r - a * x_r - b * y_r

                if not active_point_is_strictly_inside_edge(e1, (a + e1.m * b - e1.q) / (2 * e1.m)):
                    continue

                if not active_point_is_strictly_inside_edge(e2, (a + e2.m * b - e2.q) / (2 * e2.m)):
                    continue

                if not active_point_is_strictly_inside_edge(e3, (a + e3.m * b - e3.q) / (2 * e3.m)):
                    continue

                if not feasibility_outside_J_generators(e1, generators_without_triple, a, b):
                    continue

                cell_ineq_coeffs = ConvexHull([(get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b)), (get_active_point_from_generator(e3, a, b))]).equations
                x, y = sp.symbols("x y", real=True)
                cell_inequalities = [row[0] * x + row[1] * y + row[2] <= 0 for row in cell_ineq_coeffs]

                functional = a * x + b * y + c
                solutions.append((triple, cell_inequalities, functional))

        else:
            b = sp.symbols("b", real=True)
            a = (e2.q * e1.m - e1.q * e2.m + math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (e1.m - e2.m)

            expr = (a + e3.m * b - e3.q) ** 2 / (4 * e3.m) + b * e3.q - (a + e2.m * b - e2.q) ** 2 / (4 * e2.m) - b * e2.q
            poly = sp.Poly(sp.expand(expr), b)
            roots = numerical_polynomial_roots(poly)

            sol = [
                float(sp.re(r))
                for r in roots
                if is_numerically_real(r)
            ]

            solutions = []
            for b in sol:
                a = (e2.q * e1.m - e1.q * e2.m + math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (e1.m - e2.m)
                x_r, y_r = get_active_point_from_generator(e1, a, b)
                c = x_r * y_r - a * x_r - b * y_r

                if not active_point_is_strictly_inside_edge(e1, (a + e1.m * b - e1.q) / (2 * e1.m)):
                    continue

                if not active_point_is_strictly_inside_edge(e2, (a + e2.m * b - e2.q) / (2 * e2.m)):
                    continue

                if not active_point_is_strictly_inside_edge(e3, (a + e3.m * b - e3.q) / (2 * e3.m)):
                    continue

                if not feasibility_outside_J_generators(e1, generators_without_triple, a, b):
                    continue

                cell_ineq_coeffs = ConvexHull(
                    [(get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b)),
                     (get_active_point_from_generator(e3, a, b))]).equations
                x, y = sp.symbols("x y", real=True)
                cell_inequalities = [row[0] * x + row[1] * y + row[2] <= 0 for row in cell_ineq_coeffs]

                functional = a * x + b * y + c
                solutions.append((triple, cell_inequalities, functional))

            b = sp.symbols("b", real=True)
            a = (e2.q * e1.m - e1.q * e2.m - math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (e1.m - e2.m)

            expr = (a + e3.m * b - e3.q) ** 2 / (4 * e3.m) + b * e3.q - (a + e2.m * b - e2.q) ** 2 / (4 * e2.m) - b * e2.q
            poly = sp.Poly(sp.expand(expr), b)
            roots = numerical_polynomial_roots(poly)

            sol = [
                float(sp.re(r))
                for r in roots
                if is_numerically_real(r)
            ]

            for b in sol:
                a = (e2.q * e1.m - e1.q * e2.m - math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (
                            e1.m - e2.m)
                x_r, y_r = get_active_point_from_generator(e1, a, b)
                c = x_r * y_r - a * x_r - b * y_r

                if not active_point_is_strictly_inside_edge(e1, (a + e1.m * b - e1.q) / (2 * e1.m)):
                    continue

                if not active_point_is_strictly_inside_edge(e2, (a + e2.m * b - e2.q) / (2 * e2.m)):
                    continue

                if not active_point_is_strictly_inside_edge(e3, (a + e3.m * b - e3.q) / (2 * e3.m)):
                    continue

                if not feasibility_outside_J_generators(e1, generators_without_triple, a, b):
                    continue

                cell_ineq_coeffs = ConvexHull(
                    [(get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b)),
                     (get_active_point_from_generator(e3, a, b))]).equations

                x, y = sp.symbols("x y", real=True)
                cell_inequalities = [row[0] * x + row[1] * y + row[2] <= 0 for row in cell_ineq_coeffs]
                functional = a * x + b * y + c
                solutions.append((triple, cell_inequalities, functional))

        return solutions


    def _three_elements_J(self):
        """Enumerate all four type combinations for active sets of size three."""
        triples = list(combinations(self.generators, 3))
        all_solutions_three_elements_J = []
        for triple in triples:
            number_of_vertices = sum(isinstance(el, Vertex) for el in triple)
            generators_without_triple = [el for el in self.generators if el not in triple]

            if number_of_vertices == 3:
                solutions = self._solve_three_vertices(triple, generators_without_triple)
            elif number_of_vertices == 2:
                solutions = self._solve_two_vertices_one_edge(triple, generators_without_triple)
            elif number_of_vertices == 1:
                solutions = self._solve_one_vertex_two_edges(triple, generators_without_triple)
            else:
                solutions = self._solve_three_edges(triple, generators_without_triple)

            all_solutions_three_elements_J += solutions

        # Keep only full-dimensional intersections with the original domain.
        all_solutions_three_elements_J = [solution for solution in all_solutions_three_elements_J if
                                        has_2d_intersection(self.polygon.vertex_sequence, solution[1])]

        print('all solutions stemming from Js with three elements:')
        for sol in all_solutions_three_elements_J:
            print(sol)
        print('\n')
        print(100*'#')

        return all_solutions_three_elements_J


    def _determine_generators(self):
        """Return vertices and edges on which ``x*y`` is strictly convex.

        Along ``y=m*x+q``, the second derivative of ``x*y`` with respect to x
        equals ``2*m``.  Hence precisely the positive-slope edges are convex
        edge generators; vertical, horizontal, and negative-slope edges are not.
        """
        generators = list(self.polygon.vertices) + list(e for e in self.polygon.edges if e.m > 0)

        if self.polygon.is_ortho:
            pass
            # todo
            # A specialized generator reduction for orthogonal polygons can be
            # inserted here; the general enumeration remains correct without it.

        return generators



# Numerical tolerance for algebraic degeneracy checks.
DEGENERACY_TOL = 1e-12
# Relative endpoint exclusion for contacts obtained from multiple roots.
ACTIVE_POINT_REL_TOL = 1e-7
# Default absolute tolerance for evaluating already-normalized cell inequalities.
# Scale-sensitive KKT and geometry checks use ``scaled_tolerance`` instead.
FEAS_TOL = 1e-12
# Step size of the independent lifted-grid reference used by _validation().
VALIDATION_DISCRETIZATION = 0.1

# Polygon instances used by the validation suite.
VALIDATION_INSTANCES = {
    # Original examples from the module entry point.
    "original_convexified_staircase": ((0, 0), (2, 0), (2, 2), (1, 2), (0, 1)),
    "original_staircase": ((0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (0, 1)),
    "original_eight_vertex_notch": ((0, 0), (3, 0), (4, 1), (3, 3), (2, 2), (1, 3), (0, 2), (1, 1)),
    "original_critical": ((1, 0), (4, 0), (5, 1), (6, 3), (5, 5), (3, 4), (2, 5), (1, 3), (0, 1)),
    "quadrilateral_base": ((0, 0), (2, 0), (3, 1), (1, 2)),
    "quadrilateral_reversed": ((1, 2), (3, 1), (2, 0), (0, 0)),
    "quadrilateral_rotated_start": ((3, 1), (1, 2), (0, 0), (2, 0)),
    "quadrilateral_split_edge": ((0, 0), (2, 0), (2.5, 0.5), (3, 1), (1, 2)),
    "quadrilateral_negative": ((-2, -1), (0, -1), (1, 0), (-1, 1)),
    "deep_u_notch": ((0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)),
    "sloped_notch": ((0, 0), (3, 0), (3, 3), (2, 2), (1, 3), (0, 2)),
    "narrow_l": ((0, 0), (3, 0), (3, 0.2), (0.2, 0.2), (0.2, 2), (0, 2)),
    "small_scale": ((0, 0), (0.002, 0), (0.003, 0.001), (0.001, 0.002)),
    "large_scale": ((0, 0), (2000, 0), (3000, 1000), (1000, 2000)),
    "near_parallel_edges": ((0, 0), (4, 0), (6, 1), (5.9, 2), (2, 1.9), (0, 3)),
    "axis_aligned_mixed": ((0, 0), (4, 0), (4, 3), (3, 3), (3, 1), (1, 1), (1, 4), (0, 4)),
    "three_edge_branch": (
        (3.27943, 5.70233), (-1.25825, 3.59399), (-2.96916, 3.20979),
        (-1.95154, 1.29355), (-5.31055, -0.35844), (-0.58458, -6.66487),
        (1.61146, -2.04535),
    ),
    "one_vertex_two_edges": (
        (3.8571, 3.0023), (-0.4998, 3.4407), (-4.7796, 1.1793),
        (-1.2897, -3.0072), (0.6373, -6.69), (2.3745, -4.5562),
        (6.3981, -0.9325),
    ),
}



if __name__ == "__main__":

    validation_instance = "quadrilateral_base"
    vertex_sequence = tuple(Vertex(*point) for point in VALIDATION_INSTANCES[validation_instance])

    polygon = Polygon(vertex_sequence)
    envelope_generator = EnvelopePolygonalDomain(polygon)
