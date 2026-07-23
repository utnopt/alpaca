from dataclasses import dataclass, field
import math
from itertools import combinations


from sympy import S, symbols, Eq, linsolve, solveset, simplify, expand, Poly, PolynomialError, together, fraction
from sympy.core.relational import Relational
import numpy as np
from scipy.spatial import ConvexHull

FEAS_TOL = 1e-6


#todo plot function bauen #############################
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as PolygonPatch

# ------------------------------------------------------------
# 1. SymPy-Variablen
# ------------------------------------------------------------

x, y = sp.symbols("x y", real=True)

# Beispiel:
# C = {(x,y): x*y <= 1, x**2 + y**2 >= 0.5}
inequalities = [
    x * y <= 1,
    x**2 + y**2 >= 0.5,
]

# ------------------------------------------------------------
# 2. Nicht-konvexes Polygon P
#    Eckpunkte müssen entlang des Randes sortiert sein
# ------------------------------------------------------------

polygon_vertices = np.array([
    [-2.0, -1.5],
    [ 2.0, -1.5],
    [ 2.0,  1.5],
    [ 0.5,  0.3],
    [-0.5,  1.5],
    [-2.0,  1.5],
])

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

for inequality in inequalities:
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
for inequality in inequalities:
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


def extract_bivariate_quadratic(ineq, x, y):
    print("ineq ", ineq)

    if not isinstance(ineq, Relational):
        raise TypeError("ineq must be a sympy relational.")
    expr = expand(ineq.lhs - ineq.rhs)
    print("expr ", expr)
    try:
        poly = Poly(expr, x, y)

    except PolynomialError as exc:
        raise ValueError("The expression is no polynomial in x, y.") from exc

    if poly.total_degree() > 2:
        raise ValueError("The degree is bigger than 2.")

    return {
        "x2": poly.coeff_monomial(x**2),
        "xy": poly.coeff_monomial(x*y),
        "y2": poly.coeff_monomial(y**2),
        "x": poly.coeff_monomial(x),
        "y": poly.coeff_monomial(y),
        "constant": poly.coeff_monomial(1),
        "relation": ineq.rel_op,
    }


#############################################################################################

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
        print(self.polygon.vertices)
        print(self.polygon.edges)
        print(self.polygon.is_ortho)
        self.generators = self._determine_generators()

        self._three_elements_J()
        self._two_elements_J()



    def _two_elements_J(self):
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

        print('all solutions stemming from Js with two elements:')
        for sol in all_solutions_two_elements_J:
            print(sol)

    def _solve_one_vertex_one_edge(self, pair, generators_without_pair):
        v = tuple(el for el in pair if isinstance(el, Vertex))[0]
        e = tuple(el for el in pair if isinstance(el, Edge))[0]



        print('vertex, edge ', v, e)
        if v == e.v1 or v == e.v2:
            return []

        x, y = symbols("x y")
        x_j = (x * (v.y - e.q) - v.x * (y - e.q)) / (e.m * (x - v.x) + v.y - y)
        lambda_ = (e.m * (x - v.x) + v.y - y) / (v.y - e.q - e.m * v.x)
        b = (e.m * x_j * x_j - 2 * e.m * v.x * x_j - e.q * v.x + v.x * v.y) / (v.y - e.m * v.x - e.q)
        a = 2 * e.m * x_j + e.q - e.m * b

        functional = v.x * v.y + a * (x - v.x) + b * (y - v.y)
        functional = simplify(functional)
        print('functional ', functional)


        cell = []
        eta_v = v.x * v.y - a * v.x - b * v.y

        s_e = (a + e.m * b - e.q) / (2 * e.m)
        eta_e = -e.m * (s_e) ** 2 - b * e.q
        for r in generators_without_pair:
            if isinstance(r, Vertex):
                eta_r = r.x * r.y - a * r.x - b * r.y
            else:
                s_r = (a + r.m * b - r.q) / (2 * r.m )
                eta_r = -r.m * (s_r)**2 - b * r.q


            ineq1 = eta_v <= eta_r
            expr = together(ineq1.lhs - ineq1.rhs)
            num, den = fraction(expr)
            poly_ineq1 = expand(num * den) <= 0

            ineq2 = eta_e <= eta_r
            expr = together(ineq2.lhs - ineq2.rhs)
            num, den = fraction(expr)
            poly_ineq2 = expand(num * den) <= 0

            coeffs_ineq1 = extract_bivariate_quadratic(poly_ineq1, x, y)
            coeffs_ineq2 = extract_bivariate_quadratic(poly_ineq2, x, y)

            cell.append([coeffs_ineq1["x2"], coeffs_ineq1["xy"], coeffs_ineq1["y2"], coeffs_ineq1["x"], coeffs_ineq1["y"], coeffs_ineq1["constant"]])
            cell.append(
                [coeffs_ineq2["x2"], coeffs_ineq2["xy"], coeffs_ineq2["y2"], coeffs_ineq2["x"], coeffs_ineq2["y"],
                 coeffs_ineq2["constant"]])





        exit()


    def _solve_two_edges(self, pair, generators_without_pair):
        pass
        # todo implement

    def _solve_three_vertices(self, triple, generators_without_triple):
        v1, v2, v3 = triple
        # todo triple mit konvexen kanten ausschließen (later)

        A = np.array([
            [v1.x, v1.y, 1],
            [v2.x, v2.y, 1],
            [v3.x, v3.y, 1],
            ])

        if np.linalg.det(A) == 0:
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

        cell = ConvexHull([(v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y)]).equations

        x, y = symbols("x, y")
        functional = a * x + b * y + c
        solution = [(triple, cell, functional)]

        return solution

    def _solve_two_vertices_one_edge(self, triple, generators_without_triple):
        v1, v2 = (el for el in triple if isinstance(el, Vertex))
        e = tuple(el for el in triple if isinstance(el, Edge))[0]

        C = 4 * e.m * (e.q + e.m * v2.x - v2.y)
        if C == 0:
            return []
        else:
            beta2 = -1/C
            beta1 = (2 * e.q + 4 * e.m * v2.x) / C
            beta0 = (4 * e.m * v2.x * v2.y + e.q * e.q) / (-C)

            z = symbols("z")
            b = beta2 * z**2 + beta1 * z + beta0
            a = z - e.m * b

            sol = list(solveset(v1.x * v1.y - a * v1.x - b * v1.y - v2.x * v2.y + a * v2.x + b * v2.y, z, domain=S.Reals))

            solutions = []
            for z in sol:
                b = beta2 * z ** 2 + beta1 * z + beta0
                a = z - e.m * b
                c = v1.x * v1.y - a * v1.x - b * v1.y

                if not e.v1.x + FEAS_TOL < (a + e.m * b - e.q) / (2 * e.m) < e.v2.x - FEAS_TOL:
                    continue

                if not feasibility_outside_J_generators(v1, generators_without_triple, a, b):
                    continue

                cell = ConvexHull([(v1.x, v1.y), (v2.x, v2.y), (get_active_point_from_generator(e, a, b))]).equations

                x, y = symbols("x, y")
                functional = a * x + b * y + c
                solutions.append((triple, cell, functional))

            return solutions



    def _solve_one_vertex_two_edges(self, triple, generators_without_triple):
        v = tuple(el for el in triple if isinstance(el, Vertex))[0]
        e1, e2 = (el for el in triple if isinstance(el, Edge))

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

                cell = ConvexHull([(v.x, v.y), (get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b))]).equations

                x, y = symbols("x, y")
                functional = a * x + b * y + c
                solutions.append((triple, cell, functional))

            return solutions



    def _solve_three_edges(self, triple, generators_without_triple):
        e1, e2, e3 = triple

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

                cell = ConvexHull([(get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b)), (get_active_point_from_generator(e3, a, b))]).equations

                x, y = symbols("x, y")
                functional = a * x + b * y + c
                solutions.append((triple, cell, functional))

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

                cell = ConvexHull(
                    [(get_active_point_from_generator(e1, a, b)), (get_active_point_from_generator(e2, a, b)),
                     (get_active_point_from_generator(e3, a, b))]).equations

                x, y = symbols("x, y")
                functional = a * x + b * y + c
                solutions.append((triple, cell, functional))

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







    def _determine_generators(self):
        generators = list(self.polygon.vertices) + list(e for e in self.polygon.edges if e.m > 0)

        if self.polygon.is_ortho:
            pass
            # TODO implement reduction

        return generators







if __name__ == "__main__":
    vertex_sequence = (
        Vertex(0, 0),
        Vertex(2, 0),
        Vertex(2, 2),
        Vertex(1, 2),
        Vertex(0, 1)
    )

    polygon = Polygon(vertex_sequence)
    envelope_generator = EnvelopePolygonalDomain(polygon)