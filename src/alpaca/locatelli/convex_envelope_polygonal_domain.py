from dataclasses import dataclass, field
import math
from itertools import combinations, product
import shapely.geometry as sg

from sympy import S, symbols, Eq, Le, Ge, Lt, Gt, linsolve, solveset, simplify, expand, Poly, PolynomialError, together, fraction
import gurobipy as gp
from tqdm import tqdm
import numpy as np
from scipy.spatial import ConvexHull

import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as PolygonPatch

"""
Paper A: 
Polyhedral subdivisions and functional forms for the convex envelopes of bilinear, fractional and other
bivariate functions over general polytopes, Marco Locatelli

Paper B:
Convex envelopes of bivariate functions through the solution of KKT systems, Marco Locatelli
"""


FEAS_TOL = 1e-6
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

    symbols_by_name = {
        symbol.name: symbol
        for symbol in symbols
    }

    substitutions = {
        symbols_by_name["x"]: point[0],
        symbols_by_name["y"]: point[1],
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

        ineq_numeric = ineq.func(ineq.lhs, ineq.rhs + FEAS_TOL)
        if not ineq_numeric.subs(substitutions):
            #print('infeasible ineq ', ineq)
            return False, 'violation'
        #lhs = float(sp.N(ineq.lhs.subs(substitutions)))
        #rhs = float(sp.N(ineq.rhs.subs(substitutions)))

        #if not lhs <= rhs + FEAS_TOL:
        #    return False

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

        # todo implement feasibility checker here
        solutions.append(((v, e), cell_inequalities1, functional))

        cell_inequalities2 = cell_inequalities_base.copy()
        cell_inequalities2.append(e.m * (x - v.x) + v.y - y < 0)
        cell_inequalities2.append(
            2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                    v.y - y) <= lower * (e.m * (x - v.x) + v.y - y))
        cell_inequalities2.append(
            2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                    v.y - y) >= upper * (e.m * (x - v.x) + v.y - y))

        # todo implement feasibility checker here
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

            # todo implement feasibility checker here
            solutions.append(((v, e), cell_inequalities1, functional))

            cell_inequalities2 = cell_inequalities_base.copy()
            cell_inequalities2.append(e.m * (x - v.x) + v.y - y < 0)
            cell_inequalities2.append(
                2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                        v.y - y) <= lower * (e.m * (x - v.x) + v.y - y))
            cell_inequalities2.append(
                2 * e.m * x * (v.y - e.q) - 2 * e.m * v.x * (y - e.q) + e.q * e.m * (x - v.x) + e.q * (
                        v.y - y) >= upper * (e.m * (x - v.x) + v.y - y))

            # todo implement feasibility checker here
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
        undefined_points = 0
        infeasible_points = 0
        for x0, y0, _ in tqdm(points):
            # use env to completely suppress output
            env = gp.Env(empty=True)
            env.setParam("LogToConsole", 0)
            env.start()

            # build Gurobi model
            model = gp.Model(env=env)
            x_var = model.addVar(lb=minx, ub=maxx)
            y_var = model.addVar(lb=miny, ub=maxy)
            z_var = model.addVar(lb=-math.inf)

            # fix considered 2D point
            model.addConstr(x_var == x0)
            model.addConstr(y_var == y0)

            # add all facet defining inequalities from approximate convex hull
            for hull_ineq in hull_ineqs:
                model.addConstr(
                    hull_ineq[0] * x_var
                    + hull_ineq[1] * y_var
                    + hull_ineq[2] * z_var
                    + hull_ineq[3]
                    <= 0
                )

            # minimize in z direction
            model.setObjective(z_var, gp.GRB.MINIMIZE)
            model.optimize()

            if not model.status == gp.GRB.Status.OPTIMAL:
                print("Optimization failed")
                exit()

            # store obtained minimal z
            z_manual = z_var.X

            # compute analytical z from computed cells
            feasible_cell_counter = 0
            z_list = []

            for _, cell_inequalities, functional in self.all_solutions:
                feasible, msg = satisfies_all((x0, y0), cell_inequalities)

                # if fixed point is within considered cell region then we can compute analytical z accordingly
                if feasible:
                    feasible_cell_counter += 1
                    symbols = set().union(
                        *(ineq.free_symbols for ineq in cell_inequalities)
                    )

                    symbols_by_name = {
                        symbol.name: symbol
                        for symbol in symbols
                    }

                    substitutions = {
                        symbols_by_name["x"]: x0,
                        symbols_by_name["y"]: y0,
                    }
                    z_analytical = functional.subs(substitutions).evalf()
                    z_list.append(z_analytical)

                    # if imaginary part of z_analytical is not zero, there was probably division by zero happening
                    if not sp.im(z_analytical) == 0:
                        print()
                        print(
                            "Imaginary part is not zero, probably division by zero was happening."
                        )
                        z_analytical = z_manual  # add no error in this case

                else:
                    if msg == 'undefined':
                        undefined_points += 1
                        break

            if msg == 'undefined':
                continue

            if feasible_cell_counter > 1:
                print()
                print('more than one feasible cell was found, analytical z-values are ', z_list)
            if feasible_cell_counter == 0:
                infeasible_points += 1
                continue


            # sum up absolute differences to error
            error += abs(z_manual - z_analytical)
            if z_manual + 1e-08 <= z_analytical:
                print(z_manual, z_analytical)

        print('undefined vs infeasible: ', undefined_points, infeasible_points)
        print("Number of grid points is: ", len(points)- undefined_points - infeasible_points)
        print("Total sum of absolute errors is ", error)
        print("Average absolute error for each grid point is ", error / (len(points) - undefined_points - infeasible_points))
        # print("Maximum absolute error: is ", max()) # todo maximum printen


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
        x, y = symbols("x y", real=True)

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
        x, y = symbols("x y", real=True)

        # compute a, b and functional as in Appendix A.3, Paper B
        if e1.m == e2.m:
            x_i = (y + e1.m * x - e1.q) / (2 * e1.m)
            b = x_i + (e1.q - e2.q) / 4
            a = 2 * e1.m * x_i - e1.m * b + e1.q
        else:
            x_i = (y + math.sqrt(e1.m * e2.m) * x - e1.q) / (math.sqrt(e1.m) * (math.sqrt(e1.m) + math.sqrt(e2.m)))
            x_j = (y + math.sqrt(e1.m * e2.m) * x - e1.q) / (math.sqrt(e2.m) * (math.sqrt(e1.m) + math.sqrt(e2.m)))
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

        a, b, c = symbols("a, b, c")

        eqs = [
            Eq(a * v1.x + b * v1.y + c, v1.x * v1.y),
            Eq(a * v2.x + b * v2.y + c, v2.x * v2.y),
            Eq(a * v3.x + b * v3.y + c, v3.x * v3.y)
        ]

        a, b, c = list(linsolve(eqs, (a, b, c)))[0]

        if not feasibility_outside_J_generators(v1, generators_without_triple, a, b):
            return []


        cell_ineq_coeffs = ConvexHull([(v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y)]).equations
        x, y = symbols("x, y")
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

        poly = sp.Poly(expr, z)

        roots = np.roots([
            float(coefficient)
            for coefficient in poly.all_coeffs()
        ])

        sol = [
            root.real
            for root in roots
            if abs(root.imag) < 1e-10
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
            x, y = symbols("x, y")
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

            z = symbols("z")
            b = beta2 * z ** 2 + beta1 * z + beta0
            a = z - e1.m * b
            sol = list(solveset(v.x * v.y - a * v.x - b * v.y + (a + e2.m * b - e2.q)**2 / (4 * e2.m) + b * e2.q, z, domain=S.Reals))

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
                x, y = symbols("x, y")
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
            b = symbols("b")
            a = e1.m * b + (e1.q + e2.q) / 2

            sol = list(solveset((a + e3.m * b - e3.q)**2 / (4 * e3.m) + b * e3.q - (a + e2.m * b - e2.q)**2 / (4 * e2.m) - b * e2.q, b, domain=S.Reals))

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
                x, y = symbols("x, y")
                cell_inequalities = [row[0] * x + row[1] * y + row[2] <= 0 for row in cell_ineq_coeffs]

                functional = a * x + b * y + c
                solutions.append((triple, cell_inequalities, functional))

        else:
            b = symbols("b")
            a = (e2.q * e1.m - e1.q * e2.m + math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (e1.m - e2.m)

            sol = list(solveset(
                (a + e3.m * b - e3.q) ** 2 / (4 * e3.m) + b * e3.q - (a + e2.m * b - e2.q) ** 2 / (4 * e2.m) - b * e2.q,
                b, domain=S.Reals))

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
                x, y = symbols("x, y")
                cell_inequalities = [row[0] * x + row[1] * y + row[2] <= 0 for row in cell_ineq_coeffs]

                functional = a * x + b * y + c
                solutions.append((triple, cell_inequalities, functional))

            b = symbols("b")
            a = (e2.q * e1.m - e1.q * e2.m - math.sqrt(e1.m * e2.m) * ((e1.m - e2.m) * b + e1.q - e2.q)) / (e1.m - e2.m)

            sol = list(solveset(
                (a + e3.m * b - e3.q) ** 2 / (4 * e3.m) + b * e3.q - (a + e2.m * b - e2.q) ** 2 / (4 * e2.m) - b * e2.q,
                b, domain=S.Reals))

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

                x, y = symbols("x, y")
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




if __name__ == "__main__":
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