from dataclasses import dataclass, field
import math
from itertools import combinations, product
import shapely.geometry as sg
from pygments.lexers import r

from sympy import S, Eq, Le, Ge, Lt, Gt, linsolve, simplify, expand, Poly, PolynomialError, together, fraction
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

"""
Paper A: 
Polyhedral subdivisions and functional forms for the convex envelopes of bilinear, fractional and other
bivariate functions over general polytopes, Marco Locatelli

Paper B:
Convex envelopes of bivariate functions through the solution of KKT systems, Marco Locatelli
"""


FEAS_TOL = 1e-6
VALIDATION_TOL = 0.05
VALIDATION_DISCRETIZATION = 0.1


def plot_cell(cell_inequalities, polygon):
    x, y = sp.symbols('x y', real=True)

    polygon_vertices = np.array([[v.x, v.y] for v in polygon.vertex_sequence] + [[polygon.vertex_sequence[0].x, polygon.vertex_sequence[0].y]])

    # ------------------------------------------------------------
    # 3. Plotbereich aus Bounding Box des Polygons
    # ------------------------------------------------------------

    padding = 0.2

    x_min = polygon_vertices[:, 0].min() - padding
    x_max = polygon_vertices[:, 0].max() + padding
    y_min = polygon_vertices[:, 1].min() - padding
    y_max = polygon_vertices[:, 1].max() + padding

    resolution = 800

    x_values = np.linspace(x_min, x_max, resolution)
    y_values = np.linspace(y_min, y_max, resolution)

    X, Y = np.meshgrid(x_values, y_values)

    # ------------------------------------------------------------
    # 4. Maske des Polygons P
    # ------------------------------------------------------------

    points = np.column_stack((X.ravel(), Y.ravel()))

    polygon_path = Path(polygon_vertices)

    mask_P = polygon_path.contains_points(
        points,
        radius=1e-10
    ).reshape(X.shape)

    # ------------------------------------------------------------
    # 5. Maske der Zelle C
    # ------------------------------------------------------------

    mask_C = np.ones(X.shape, dtype=bool)

    for inequality in cell_inequalities:
        # SymPy-Ungleichung direkt als boolesche NumPy-Funktion
        inequality_function = sp.lambdify(
            (x, y),
            inequality,
            modules="numpy"
        )

        current_mask = inequality_function(X, Y)

        # Falls SymPy bei konstanten Ausdrücken nur einen booleschen
        # Einzelwert zurückgibt
        current_mask = np.broadcast_to(
            np.asarray(current_mask, dtype=bool),
            X.shape
        )

        mask_C &= current_mask

    # ------------------------------------------------------------
    # 6. Schnitt C ∩ P
    # ------------------------------------------------------------

    mask_intersection = mask_C & mask_P

    # ------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    # Zulässige Schnittmenge einfärben
    ax.contourf(
        X,
        Y,
        mask_intersection.astype(float),
        levels=[0.5, 1.5],
        alpha=0.5
    )

    # Rand des Polygons einzeichnen
    polygon_patch = PolygonPatch(
        polygon_vertices,
        closed=True,
        fill=False,
        edgecolor="black",
        linewidth=2,
        label=r"$P$"
    )
    ax.add_patch(polygon_patch)

    # Grenzen der polynomialen Zelle einzeichnen
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


def feasibility_outside_J_generators(generator_in_J, generators_outside_J, a, b):
    x_k, y_k = get_active_point_from_generator(generator_in_J, a, b)

    for generator_outside_J in generators_outside_J:
        x_r, y_r = get_active_point_from_generator(generator_outside_J, a, b)

        if x_k * y_k - a * x_k - b * y_k > x_r * y_r - a * x_r - b * y_r + FEAS_TOL:
            return False

    return True


def chop_floats(expr, tol=1e-10):
    replacements = {}
    for f in expr.atoms(sp.Float):
        if abs(float(f)) < tol:
            replacements[f] = sp.Integer(0)

    return expr.xreplace(replacements)


def satisfies_all(point, inequalities):
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
        ineq = ineq.func(
            chop_floats(ineq.lhs),
            chop_floats(ineq.rhs)
        )
        expressions = (ineq.lhs, ineq.rhs)

        for expr in expressions:
            denominator = sp.denom(sp.together(expr))

            if abs(float(denominator.subs(substitutions))) <= FEAS_TOL:
                return False, 'undefined'

        # Relax lower bounds downward and upper bounds upward.
        tolerance = -FEAS_TOL if isinstance(ineq, (Ge, Gt)) else FEAS_TOL
        ineq_numeric = ineq.func(ineq.lhs, ineq.rhs + tolerance)
        if not ineq_numeric.subs(substitutions):
            return False, 'violation'


    return True, None


def compute_cells_one_vertex_one_edge(solutions, v, e, inequalities, functional):
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
    # todo checken, ist von codex
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


def has_2d_intersection(vertex_sequence, inequalities, tol=1e-10):
    """
    Prüft, ob ein ggf. nicht-konvexes Polygon einen zweidimensionalen
    Schnitt mit einer durch lineare SymPy-Ungleichungen gegebenen Zelle hat.

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Vertex sequence des Polygons, z.B.
        [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)].

    inequalities : list
        Liste linearer SymPy-Ungleichungen in x und y, z.B.
        [x >= 0, y >= 0, x + y <= 2].

    x, y : sympy.Symbol
        Die verwendeten SymPy-Variablen.

    tol : float
        Flächentoleranz.

    Returns
    -------
    bool
        True genau dann, wenn der Schnitt positive Fläche besitzt.
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

    scale = max(maxx - minx, maxy - miny, 1.0)
    margin = 10 * scale

    bounding_box = box(
        minx - margin,
        miny - margin,
        maxx + margin,
        maxy + margin,
    )

    for inequality in inequalities:

        # Bringe Ungleichung auf Form expr <= 0 bzw. expr >= 0.
        expr = sp.expand(inequality.lhs - inequality.rhs)

        poly = sp.Poly(expr, x, y)

        if poly.total_degree() > 1:
            raise ValueError(
                f"Ungleichung ist nicht linear: {inequality}"
            )

        a = float(expr.coeff(x))
        b = float(expr.coeff(y))
        c = float(expr.subs({x: 0, y: 0}))

        # a*x + b*y + c = 0 ist die Begrenzungsgerade.
        if abs(a) < tol and abs(b) < tol:
            # Ungleichung enthält gar kein x oder y.
            if not bool(inequality):
                return False
            continue

        # Zwei weit auseinanderliegende Punkte auf der Geraden.
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
                valid = value <= tol

            elif isinstance(
                inequality,
                (sp.core.relational.GreaterThan,
                 sp.core.relational.StrictGreaterThan)
            ):
                valid = value >= -tol

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

        if intersection.area <= tol:
            return False

    return intersection.area > tol

@dataclass(frozen=True)
class Vertex:
    x: float
    y: float


@dataclass(frozen=True)
class Edge:
    v1: Vertex
    v2: Vertex
    m: float = field(init=False)
    q: float = field(init=False)

    def __post_init__(self):
        if self.v1.x > self.v2.x:
            tmp = self.v1
            object.__setattr__(self, "v1", self.v2)
            object.__setattr__(self, "v2", tmp)

        if self.v1.x == self.v2.x:
            m = math.nan
            q = math.nan
        else:
            m = (self.v2.y - self.v1.y) / (self.v2.x - self.v1.x)
            q = self.v1.y - m * self.v1.x

        object.__setattr__(self, "m", m)
        object.__setattr__(self, "q", q)


@dataclass(frozen=True)
class Polygon:
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
        # read discretization and polygon bounds
        dx = VALIDATION_DISCRETIZATION
        dy = VALIDATION_DISCRETIZATION
        minx = min(v.x for v in self.polygon.vertices)
        miny = min(v.y for v in self.polygon.vertices)
        maxx = max(v.x for v in self.polygon.vertices)
        maxy = max(v.y for v in self.polygon.vertices)
        shapely_polygon = sg.Polygon((v.x, v.y) for v in self.polygon.vertex_sequence)

        # construct grid and store all 3D points over polygon
        x_vals = np.arange(minx, maxx + dx, dx)
        y_vals = np.arange(miny, maxy + dy, dy)
        points = []
        for x in x_vals:
            for y in y_vals:
                p = sg.Point(x, y)
                if shapely_polygon.covers(p):
                    points.append([p.x, p.y, p.x * p.y])
        points = np.array(points)

        # construct convex hull of 3D points
        hull = ConvexHull(points)

        # for each 2D point from polygon compute minimal z-coordinate in approximate convex hull and
        # compare to computed analytical result
        hull_ineqs = hull.equations  # consider hull facets
        error = 0
        compared_points = 0
        undefined_points = 0
        infeasible_points = 0
        exceeded_validation_tolerances = []
        largest_cell_difference = 0
        worst_cell_comparison = None
        nan_counter = 0
        invalid_value_counter = 0
        unassigned_diagnostics = []
        # Reuse one environment and model for all grid points.
        with gp.Env(empty=True) as env:
            env.setParam("LogToConsole", 0)
            env.start()

            with gp.Model(env=env) as model:
                x_var = model.addVar(lb=minx, ub=maxx)
                y_var = model.addVar(lb=miny, ub=maxy)
                z_var = model.addVar(lb=-math.inf)

                x_fix = model.addConstr(x_var == minx)
                y_fix = model.addConstr(y_var == miny)

                for hull_ineq in hull_ineqs:
                    model.addConstr(
                        hull_ineq[0] * x_var
                        + hull_ineq[1] * y_var
                        + hull_ineq[2] * z_var
                        + hull_ineq[3]
                        <= 0
                    )

                model.setObjective(z_var, gp.GRB.MINIMIZE)

                for x0, y0, _ in tqdm(points):
                    x_fix.RHS = x0
                    y_fix.RHS = y0
                    model.optimize()

                    if model.status != gp.GRB.Status.OPTIMAL:
                        raise RuntimeError(
                            f"Validation failed at ({x0}, {y0}): "
                            f"Gurobi status {model.status}"
                        )

                    # store obtained minimal z
                    z_manual = z_var.X

                    # compute analytical z from computed cells
                    feasible_cell_counter = 0
                    has_undefined_cell = False
                    z_list = []
                    functionals_list = []
                    valid_cells = []


                    for generators, cell_inequalities, functional in self.all_solutions:
                        feasible, msg = satisfies_all((x0, y0), cell_inequalities)

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
                                continue

                            z_list.append(z_analytical)
                            functionals_list.append(functional)
                            valid_cells.append((generators, cell_inequalities, functional, z_analytical))


                        elif msg == 'undefined':
                            has_undefined_cell = True

                    if not z_list:
                        if len(unassigned_diagnostics) < 10:
                            violation_count = 0
                            undefined_count = 0
                            for _, candidate_inequalities, _ in self.all_solutions:
                                _, candidate_msg = satisfies_all(
                                    (x0, y0), candidate_inequalities
                                )
                                if candidate_msg == 'violation':
                                    violation_count += 1
                                elif candidate_msg == 'undefined':
                                    undefined_count += 1
                            unassigned_diagnostics.append(
                                ((x0, y0), violation_count, undefined_count)
                            )
                            if len(unassigned_diagnostics) == 1:
                                print("Detailed diagnosis for first unassigned point:", (x0, y0))
                                for cell_index, (candidate_generators, candidate_inequalities, _) in enumerate(
                                        self.all_solutions, start=1):
                                    candidate_ok, candidate_msg = satisfies_all(
                                        (x0, y0), candidate_inequalities
                                    )
                                    if not candidate_ok:
                                        failed_index = next(
                                            (index for index, inequality in enumerate(candidate_inequalities)
                                             if not bool(inequality.subs({sp.Symbol("x", real=True): x0,
                                                                           sp.Symbol("y", real=True): y0}))),
                                            None,
                                        )
                                        failed_inequality = (
                                            candidate_inequalities[failed_index]
                                            if failed_index is not None else None
                                        )
                                        failed_slack = None
                                        if failed_inequality is not None:
                                            coordinate_values = {"x": x0, "y": y0}
                                            failed_substitutions = {
                                                symbol: coordinate_values[symbol.name]
                                                for symbol in failed_inequality.free_symbols
                                                if symbol.name in coordinate_values
                                            }
                                            failed_slack = float(sp.N(
                                                (failed_inequality.lhs - failed_inequality.rhs)
                                                .subs(failed_substitutions)
                                            ))
                                        print(
                                            "  Cell", cell_index,
                                            "generators=", candidate_generators,
                                            "reason=", candidate_msg,
                                            "failed_inequality_index=", failed_index,
                                            "failed_inequality=", failed_inequality,
                                            "lhs_minus_rhs=", failed_slack,
                                        )
                        if feasible_cell_counter == 0 and not has_undefined_cell:
                            infeasible_points += 1
                        else:
                            undefined_points += 1
                        continue

                    if len(z_list) > 1:
                        cell_difference = max(z_list) - min(z_list)
                        if worst_cell_comparison is None or cell_difference > largest_cell_difference:
                            largest_cell_difference = cell_difference
                            worst_cell_comparison = ((x0, y0), z_manual, valid_cells)

                        print('more than one feasible cell was found')
                        print('analytical z-values are ', z_list)
                        print('functionals are ', functionals_list)
                        print('###########')

                        if abs(max(z_list) - min(z_list)) > VALIDATION_TOL:
                            print('too much difference in z-values')
                            exceeded_validation_tolerances.append(abs(max(z_list) - min(z_list)))
                    # Use the last valid value, not a potentially discarded NaN.
                    z_analytical = z_list[-1]
                    # sum up absolute differences to error
                    error += abs(z_manual - z_analytical)
                    compared_points += 1
                    if z_manual + 1e-08 <= z_analytical:
                        print(z_manual, z_analytical)

        print('undefined vs infeasible: ', undefined_points, infeasible_points)
        print("Number of compared grid points is: ", compared_points)
        print("Total sum of absolute errors is ", error)
        if compared_points > 0:
            print("Average absolute error for each compared grid point is ", error / compared_points)
        else:
            print("No evaluable grid points; average absolute error is undefined.")
        print("Exceeded validation tolerances are ", exceeded_validation_tolerances)
        print("Number of nan values occured in functional evaluation is ", nan_counter)
        print("Number of non-finite or non-real functional values is ", invalid_value_counter)
        print("First unassigned grid points (point, violations, undefined): ", unassigned_diagnostics)
        if worst_cell_comparison is not None:
            point, reference_value, cells = worst_cell_comparison
            print("Largest difference between analytical cell values is ", largest_cell_difference)
            print("Grid point is ", point)
            print("Discretized reference value is ", reference_value)
            for index, (generators, inequalities, functional, value) in enumerate(cells, start=1):
                print(f"Cell {index}:")
                print("  Generators: ", generators)
                print("  Inequalities: ", inequalities)
                print("  Functional: ", functional)
                print("  Analytical value: ", value)
        else:
            print("No grid point with multiple valid cell values found.")


    def _two_elements_J(self):
        pairs = list(combinations(self.generators, 2))

        all_solutions_two_elements_J = []
        for pair in pairs:
            number_of_vertices = sum(isinstance(el, Vertex) for el in pair)
            generators_without_pair = [el for el in self.generators if el not in pair]

            solutions = []
            if number_of_vertices == 1:
                solutions = self._solve_one_vertex_one_edge_linear(pair, generators_without_pair)
            if number_of_vertices == 0:
                solutions = self._solve_two_edges_linear(pair, generators_without_pair)

            all_solutions_two_elements_J += solutions

        # filter all cells that are not 2D
        all_solutions_two_elements_J = [solution for solution in all_solutions_two_elements_J if has_2d_intersection(self.polygon.vertex_sequence, solution[1], tol=1e-10)]

        print('all solutions stemming from Js with two elements:')
        for sol in all_solutions_two_elements_J:
            print(sol)

        return all_solutions_two_elements_J


    def _solve_one_vertex_one_edge_linear(self, pair, generators_without_pair):
        # extract vertex and edge from input pair
        v = tuple(el for el in pair if isinstance(el, Vertex))[0]
        e = tuple(el for el in pair if isinstance(el, Edge))[0]

        # the vertex cannot lie on the line containing the edge
        if e.m * v.x + e.q == v.y:
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

        # consider edges outside J and introduce case distinction
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


    def _solve_two_edges_linear(self, pair, generators_without_pair):
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

        # case distinction on equality of slopes of edges
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
        v1, v2, v3 = triple
        # todo triple mit konvexen kanten ausschließen (later)

        A = np.array([
            [v1.x, v1.y, 1],
            [v2.x, v2.y, 1],
            [v3.x, v3.y, 1],
            ])

        if abs(np.linalg.det(A)) <= FEAS_TOL:
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
        v1, v2 = (el for el in triple if isinstance(el, Vertex))
        e = tuple(el for el in triple if isinstance(el, Edge))[0]


        C = 4 * e.m * (e.q + e.m * v2.x - v2.y)

        if np.isclose(float(C), 0.0, atol=1e-12):
            return []

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
        roots = sp.nroots(poly, n=10, maxsteps=200)

        sol = [
            float(sp.re(r))
            for r in roots
            if abs(float(sp.im(r))) < 1e-10
        ]

        solutions = []
        for z in sol:
            b = beta2 * z ** 2 + beta1 * z + beta0
            a = z - e.m * b
            c = v1.x * v1.y - a * v1.x - b * v1.y

            if not e.v1.x + FEAS_TOL < (a + e.m * b - e.q) / (2 * e.m) < e.v2.x - FEAS_TOL:
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
        v = tuple(el for el in triple if isinstance(el, Vertex))[0]
        e1, e2 = (el for el in triple if isinstance(el, Edge))

        if e1.m == e2.m and e1.q == e2.q:
            return []

        C = 4 * e1.m * (e1.q + e1.m * v.x - v.y)
        if C == 0:
            return []
        else:
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
            roots = sp.nroots(poly, n=10, maxsteps=200)

            sol = [
                float(sp.re(r))
                for r in roots
                if abs(float(sp.im(r))) < 1e-10
            ]

            solutions = []
            for z in sol:
                b = beta2 * z ** 2 + beta1 * z + beta0
                a = z - e1.m * b
                c = v.x * v.y - a * v.x - b * v.y

                if not e1.v1.x + FEAS_TOL < (a + e1.m * b - e1.q) / (2 * e1.m) < e1.v2.x - FEAS_TOL:
                    continue

                if not e2.v1.x + FEAS_TOL < (a + e2.m * b - e2.q) / (2 * e2.m) < e2.v2.x - FEAS_TOL:
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
            roots = sp.nroots(poly, n=10, maxsteps=200)

            sol = [
                float(sp.re(r))
                for r in roots
                if abs(float(sp.im(r))) < 1e-10
            ]


            solutions = []
            for b in sol:
                a = e1.m * b + (e1.q + e2.q) / 2
                x_r, y_r = get_active_point_from_generator(e1, a, b)
                c = x_r * y_r - a * x_r - b * y_r

                if not e1.v1.x + FEAS_TOL < (a + e1.m * b - e1.q) / (2 * e1.m) < e1.v2.x - FEAS_TOL:
                    continue

                if not e2.v1.x + FEAS_TOL < (a + e2.m * b - e2.q) / (2 * e2.m) < e2.v2.x - FEAS_TOL:
                    continue

                if not e3.v1.x + FEAS_TOL < (a + e3.m * b - e3.q) / (2 * e3.m) < e3.v2.x - FEAS_TOL:
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
            roots = sp.nroots(poly, n=10, maxsteps=200)

            sol = [
                float(sp.re(r))
                for r in roots
                if abs(float(sp.im(r))) < 1e-10
            ]

            solutions = []
            for b in sol:
                a = (e2.q * e1.m - e1.q * e2.m + math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (e1.m - e2.m)
                x_r, y_r = get_active_point_from_generator(e1, a, b)
                c = x_r * y_r - a * x_r - b * y_r

                if not e1.v1.x + FEAS_TOL < (a + e1.m * b - e1.q) / (2 * e1.m) < e1.v2.x - FEAS_TOL:
                    continue

                if not e2.v1.x + FEAS_TOL < (a + e2.m * b - e2.q) / (2 * e2.m) < e2.v2.x - FEAS_TOL:
                    continue

                if not e3.v1.x + FEAS_TOL < (a + e3.m * b - e3.q) / (2 * e3.m) < e3.v2.x - FEAS_TOL:
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
            roots = sp.nroots(poly, n=10, maxsteps=200)

            sol = [
                float(sp.re(r))
                for r in roots
                if abs(float(sp.im(r))) < 1e-10
            ]

            for b in sol:
                a = (e2.q * e1.m - e1.q * e2.m - math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (
                            e1.m - e2.m)
                x_r, y_r = get_active_point_from_generator(e1, a, b)
                c = x_r * y_r - a * x_r - b * y_r

                if not e1.v1.x + FEAS_TOL < (a + e1.m * b - e1.q) / (2 * e1.m) < e1.v2.x - FEAS_TOL:
                    continue

                if not e2.v1.x + FEAS_TOL < (a + e2.m * b - e2.q) / (2 * e2.m) < e2.v2.x - FEAS_TOL:
                    continue

                if not e3.v1.x + FEAS_TOL < (a + e3.m * b - e3.q) / (2 * e3.m) < e3.v2.x - FEAS_TOL:
                    continue

                if not feasibility_outside_J_generators(e1, generators_without_triple, a, b):
                    continue

                cell = ConvexHull(
                    [(get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b)),
                     (get_active_point_from_generator(e3, a, b))]).equations

                x, y = sp.symbols("x y", real=True)
                functional = a * x + b * y + c
                solutions.append((triple, cell, functional))

        return solutions


    def _three_elements_J(self):
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

        # filter all cells that are not 2D
        all_solutions_three_elements_J = [solution for solution in all_solutions_three_elements_J if
                                        has_2d_intersection(self.polygon.vertex_sequence, solution[1], tol=1e-10)]

        print('all solutions stemming from Js with three elements:')
        for sol in all_solutions_three_elements_J:
            print(sol)

        return all_solutions_three_elements_J


    def _determine_generators(self):
        generators = list(self.polygon.vertices) + list(e for e in self.polygon.edges if e.m > 0)

        if self.polygon.is_ortho:
            pass
            # TODO implement reduction

        return generators

#todo remove
def f1(x, y):
    return 0.34314575050762*x**2 + 0.48528137423857*x*y - 0.68629150101524*x + 0.17157287525381*y**2 + 0.343145750507619*y - 4.44089209850063e-16

#todo remove
def f2(x, y):
    return 1.0*(2.0*x**2 + 2.0*x*y - 6.0*x - 1.0*y**2 + 4.0)/(x - y + 1)

if __name__ == "__main__":

    # todo union of cells with same functionals

    # convexified staircase polygon
    if False:
        vertex_sequence = (
            Vertex(0, 0),
            Vertex(2, 0),
            Vertex(2, 2),
            Vertex(1, 2),
            Vertex(0, 1)
        )

    # staircase polygon
    if False:
        vertex_sequence = (
            Vertex(0, 0),
            Vertex(2, 0),
            Vertex(2, 2),
            Vertex(1, 2),
            Vertex(1, 1),
            Vertex(0, 1)
        )

    if False:
        vertex_sequence = (
            Vertex(0, 0),
            Vertex(3, 0),
            Vertex(4, 1),
            Vertex(3, 3),
            Vertex(2, 2),
            Vertex(1, 3),
            Vertex(0, 2),
            Vertex(1, 1),
        )

    if False:
        vertex_sequence = (
            Vertex(1, 0),
            Vertex(4, 0),
            Vertex(5, 1),  # positive slope
            Vertex(6, 3),  # positive slope
            Vertex(5, 5),
            Vertex(3, 4),  # positive slope
            Vertex(2, 5),
            Vertex(1, 3),  # positive slope
            Vertex(0, 1),  # positive slope
        )

    if True:
        vertex_sequence = (
            Vertex(0, 0),
            Vertex(2, 0),
            Vertex(3, 1),
            Vertex(1, 2),
        )

    polygon = Polygon(vertex_sequence)
    envelope_generator = EnvelopePolygonalDomain(polygon)
