from dataclasses import dataclass, field
import math
from itertools import combinations


from sympy import S, symbols, Eq, linsolve, solveset
import numpy as np
from scipy.spatial import ConvexHull

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
            object.__setattr__(self, "v1", self.v2)
            object.__setattr__(self, "v2", self.v1)

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

        self._two_elements_J()
        self._three_elements_J()


    def _two_elements_J(self):
        pairs = list(combinations(self.generators, 2))
        #todo implement

    def _solve_three_vertices(self, triple, generators_without_triple):
        v1, v2, v3 = triple
        # todo triple mit konvexen kanten ausschließen

        A = np.array([
            [v1.x, v1.y, 1],
            [v2.x, v2.y, 1],
            [v3.x, v3.y, 1],
            ])

        if np.linalg.det(A) == 0:
            return None

        a, b, c = symbols("a, b, c")

        eqs = [
            Eq(a * v1.x + b * v1.y + c, v1.x * v1.y),
            Eq(a * v2.x + b * v2.y + c, v2.x * v2.y),
            Eq(a * v3.x + b * v3.y + c, v3.x * v3.y)
        ]

        a, b, c = list(linsolve(eqs, (a, b, c)))[0]
        for generator_outside_J in generators_without_triple:
            x_r, y_r = get_active_point_from_generator(generator_outside_J, a, b)

            if v1.x * v1.y - a * v1.x - b * v1.y > x_r * y_r - a * x_r - b * y_r:
                return None

        cell = ConvexHull([(v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y)]).equations

        x, y = symbols("x, y")
        functional = a * x + b * y + c
        solution = [(cell, functional)]

        return solution

    def _solve_two_vertices_one_edge(self, triple, generators_without_triple):
        v1, v2 = (el for el in triple if isinstance(el, Vertex))
        e = tuple(el for el in triple if isinstance(el, Edge))[0]

        C = 4 * e.m * (e.q + e.m * v2.x - v2.y)
        if C == 0:
            return None
        else:
            beta2 = -1/C
            beta1 = (2 * e.q + 4 * e.m * v2.x) / C
            beta0 = (4 * e.m * v2.x * v2.y + e.q * e.q) / (-C)

            z = symbols("z")
            b = beta2 * z**2 + beta1 * z + beta0
            a = z - e.m * b

            sol = list(solveset(v1.x * v1.y - a * v1.x - b * v1.y - v2.x * v2.y + a * v2.x + b * v2.y, z, domain=S.Reals))

            for z in sol:
                if e.v1.x < (z - e.q) / (2 * e.m) < e.v2.x:
                    b = beta2 * z ** 2 + beta1 * z + beta0
                    a = z - e.m * b
                    c = v1.x * v1.y - a * v1.x - b * v1.y




        #todo implement hier weiter


    def _solve_one_vertex_two_edges(self, triple, generators_without_triple):
        v = tuple(el for el in triple if isinstance(el, Vertex))[0]
        e1, e2 = (el for el in triple if isinstance(el, Edge))

        #todo implement
        pass

    def _solve_three_edges(self, triple, generators_without_triple):
        e1, e2, e3 = triple

        #todo implement
        pass

    def _three_elements_J(self):
        triples = list(combinations(self.generators, 3))
        all_solutions = []
        for triple in triples:
            number_of_vertices = sum(isinstance(el, Vertex) for el in triple)
            generators_without_triple = [el for el in self.generators if el not in triple]

            if number_of_vertices == 3:
                solutions = self._solve_three_vertices(triple, generators_without_triple)
                print(triple)
                print('3er solutions ', solutions)
            elif number_of_vertices == 2:
                solutions = self._solve_two_vertices_one_edge(triple, generators_without_triple)
            elif number_of_vertices == 1:
                solutions = self._solve_one_vertex_two_edges(triple, generators_without_triple)
            else:
                solutions = self._solve_three_edges(triple, generators_without_triple)

            if solutions is None:
                continue
            else:
                all_solutions += solutions


            # TODO generators_without_triple inequalities pruefen




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