"""Analytical checks independent of plotting and Gurobi validation."""
import io
import math
import unittest
from unittest.mock import patch

import sympy as sp
from alpaca.locatelli import convex_envelope_polygonal_domain as domain

x, y = sp.symbols("x y", real=True)
SQUARE = tuple(domain.Vertex(*p) for p in ((0, 0), (1, 0), (1, 1), (0, 1)))


class TestPolygonalEnvelope(unittest.TestCase):
    def test_outside_edge_eta_uses_selected_domain(self):
        vertex = lambda x, y: domain.Vertex(sp.Integer(x), sp.Integer(y))
        edge = domain.Edge(vertex(4, 0), vertex(5, 1))
        outside = domain.Edge(vertex(1, 3), vertex(2, 5))
        a, b = sp.symbols("a b", real=True)
        substitutions = {a: sp.Rational(419, 252), b: sp.Rational(673, 252)}
        # Only the inner minimum needs an additional inequality.
        for region in [-1, 0, 1]:
            with self.subTest(region=region):
                inequalities = []
                domain.add_eta_inequalities_for_edges_outside_J(
                    inequalities, edge, [outside], [region], substitutions)
                if region == 0:
                    self.assertEqual(len(inequalities), 1)
                    self.assertFalse(bool(inequalities[0]))
                else:
                    self.assertEqual(inequalities, [])

    def test_edge_geometry(self):
        edge = domain.Edge(domain.Vertex(3, 7), domain.Vertex(1, 3))
        self.assertEqual(edge.v1, domain.Vertex(1, 3))
        self.assertEqual((edge.m, edge.q), (2, 1))
        self.assertTrue(math.isnan(domain.Edge(domain.Vertex(1, 0), domain.Vertex(1, 2)).m))

    def test_square_polygon(self):
        polygon = domain.Polygon(SQUARE)
        self.assertTrue(polygon.is_ortho)
        self.assertEqual(len(polygon.edges), 4)
        self.assertEqual(polygon.vertices, frozenset(SQUARE))

    def test_active_edge_point_is_clamped(self):
        edge = domain.Edge(domain.Vertex(1, 3), domain.Vertex(3, 7))
        for a, expected in [(-10, (1, 3)), (7, (1.5, 4)), (20, (3, 7))]:
            with self.subTest(a=a):
                self.assertEqual(domain.get_active_point_from_generator(edge, a, 0), expected)

    def test_active_edge_interior_classification_is_translation_invariant(self):
        for offset in (0, 1e8):
            with self.subTest(offset=offset):
                edge = domain.Edge(
                    domain.Vertex(offset, 0),
                    domain.Vertex(offset + 1, 1),
                )
                self.assertFalse(domain.active_point_is_strictly_inside_edge(edge, offset + 5e-8))
                self.assertTrue(domain.active_point_is_strictly_inside_edge(edge, offset + 1e-3))

    def test_intersection_requires_positive_area(self):
        for inequalities, expected in [
            ([x >= 0, y >= 0, x + y <= 1], True),
            ([x >= 2], False), ([x <= 0], False), ([], True),
        ]:
            with self.subTest(inequalities=inequalities):
                self.assertEqual(domain.has_2d_intersection(SQUARE, inequalities), expected)

    def test_satisfies_all(self):
        for inequalities, point, expected in [
            ([x >= 0, y >= 0], (0, 0), (True, None)),
            ([x <= 1, y <= 1], (2, 0), (False, "violation")),
            ([x >= 0], (1, 0), (True, None)),
            ([y >= 0], (0, 1), (True, None)),
            ([], (0, 0), (True, None)),
            ([x / y <= 1], (1, 0), (False, "undefined")),
        ]:
            with self.subTest(inequalities=inequalities, point=point):
                self.assertEqual(domain.satisfies_all(point, inequalities), expected)

    def test_feasibility_tolerance_relaxes_both_directions(self):
        for relation, direction in [(sp.Ge, -1), (sp.Gt, -1), (sp.Le, 1), (sp.Lt, 1)]:
            for multiple, expected in [(0.5, (True, None)), (2, (False, "violation"))]:
                with self.subTest(relation=relation, multiple=multiple):
                    point = (direction * multiple * domain.FEAS_TOL, 0)
                    self.assertEqual(domain.satisfies_all(point, [relation(x, 0)]), expected)

    def test_unit_square_envelope_matches_known_formula(self):
        with patch.object(domain, "plot_cell"), patch.object(domain.EnvelopePolygonalDomain, "_validation"):
            envelope = domain.EnvelopePolygonalDomain(domain.Polygon(SQUARE))
        self.assertEqual(len(envelope.all_solutions), 2)
        for px in (0, 0.25, 0.5, 0.75, 1):
            for py in (0, 0.25, 0.5, 0.75, 1):
                with self.subTest(point=(px, py)):
                    values = [float(functional.subs({x: px, y: py}))
                              for _, inequalities, functional in envelope.all_solutions
                              if all(bool(ineq.subs({x: px, y: py})) for ineq in inequalities)]
                    self.assertTrue(values)
                    for value in values:
                        self.assertAlmostEqual(value, max(0, px + py - 1))

    def test_translated_unit_square_envelope_matches_known_formula(self):
        offset_x = 1e8
        offset_y = -1e8
        translated_square = tuple(
            domain.Vertex(offset_x + px, offset_y + py)
            for px, py in ((0, 0), (1, 0), (1, 1), (0, 1))
        )
        with patch.object(domain, "plot_cell"), patch.object(domain.EnvelopePolygonalDomain, "_validation"):
            envelope = domain.EnvelopePolygonalDomain(domain.Polygon(translated_square))

        for local_x in (0.25, 0.5, 0.75):
            for local_y in (0.25, 0.5, 0.75):
                point_x = offset_x + local_x
                point_y = offset_y + local_y
                values = [
                    float(functional.subs({x: point_x, y: point_y}))
                    for _, inequalities, functional in envelope.all_solutions
                    if domain.satisfies_all((point_x, point_y), inequalities)[0]
                ]
                expected = (
                    max(0, local_x + local_y - 1)
                    + offset_y * local_x
                    + offset_x * local_y
                    + offset_x * offset_y
                )
                self.assertTrue(values)
                for value in values:
                    self.assertAlmostEqual(value, expected, delta=4.0)

    def test_plotting_groups_cells_with_identical_functional(self):
        coordinates = domain.VALIDATION_INSTANCES["deep_u_notch"]
        polygon = domain.Polygon(tuple(domain.Vertex(*point) for point in coordinates))
        with patch.object(domain, "plot_cell") as plot_cell, \
                patch.object(domain.EnvelopePolygonalDomain, "_validation"), \
                patch("sys.stdout", new_callable=io.StringIO):
            envelope = domain.EnvelopePolygonalDomain(polygon)

        groups = domain.group_cells_by_functional(envelope.all_solutions)
        self.assertLess(len(groups), len(envelope.all_solutions))
        self.assertEqual(sum(len(cell_sets) for _, cell_sets in groups), len(envelope.all_solutions))
        self.assertTrue(any(len(cell_sets) > 1 for _, cell_sets in groups))
        self.assertEqual(plot_cell.call_count, len(groups))
        self.assertEqual(
            [call.args[2] for call in plot_cell.call_args_list],
            [functional for functional, _ in groups],
        )

    def test_line_incidence_distinguishes_geometry_after_translation(self):
        edge = domain.Edge(
            domain.Vertex(1e8, 1e8),
            domain.Vertex(1e8 + 1, 1e8 + 1),
        )
        self.assertTrue(domain.vertex_lies_on_edge_line(domain.Vertex(1e8 + 0.5, 1e8 + 0.5), edge))
        self.assertFalse(domain.vertex_lies_on_edge_line(domain.Vertex(1e8 + 0.5, 1e8 + 0.50001), edge))

    def test_three_edge_negative_branch_returns_symbolic_cell_inequalities(self):
        coordinates = domain.VALIDATION_INSTANCES["three_edge_branch"]
        polygon = domain.Polygon(tuple(domain.Vertex(*point) for point in coordinates))
        envelope = domain.EnvelopePolygonalDomain.__new__(domain.EnvelopePolygonalDomain)
        envelope.polygon = polygon
        envelope.generators = envelope._determine_generators()

        with patch("sys.stdout", new_callable=io.StringIO):
            solutions = envelope._three_elements_J()

        edge_triples = [
            (generators, inequalities, functional)
            for generators, inequalities, functional in solutions
            if all(isinstance(generator, domain.Edge) for generator in generators)
        ]
        self.assertTrue(edge_triples)
        self.assertTrue(all(
            hasattr(inequality, "lhs")
            for _, inequalities, _ in edge_triples
            for inequality in inequalities
        ))

        # This instance exercises the negative square-root branch in Lemma 5.1.
        for generators, _, functional in edge_triples:
            e1, e2, _ = generators
            a_value = float(functional.coeff(x))
            b_value = float(functional.coeff(y))
            common = (e2.q * e1.m - e1.q * e2.m) / (e1.m - e2.m)
            root_term = (
                math.sqrt(e1.m * e2.m)
                * ((e1.m - e2.m) * b_value + e1.q - e2.q)
                / (e1.m - e2.m)
            )
            self.assertAlmostEqual(a_value, common - root_term)

    def test_one_vertex_two_edges_case_produces_full_dimensional_cell(self):
        coordinates = domain.VALIDATION_INSTANCES["one_vertex_two_edges"]
        polygon = domain.Polygon(tuple(domain.Vertex(*point) for point in coordinates))
        envelope = domain.EnvelopePolygonalDomain.__new__(domain.EnvelopePolygonalDomain)
        envelope.polygon = polygon
        envelope.generators = envelope._determine_generators()

        with patch("sys.stdout", new_callable=io.StringIO):
            solutions = envelope._three_elements_J()

        matching_cells = [
            (generators, inequalities, functional)
            for generators, inequalities, functional in solutions
            if sum(isinstance(generator, domain.Vertex) for generator in generators) == 1
        ]
        self.assertTrue(matching_cells)
        for _, inequalities, _ in matching_cells:
            self.assertTrue(domain.has_2d_intersection(polygon.vertex_sequence, inequalities))

    def test_vertex_edge_solver_rejects_endpoint_despite_roundoff(self):
        coordinates = domain.VALIDATION_INSTANCES["three_edge_branch"]
        vertex = domain.Vertex(*coordinates[3])
        edge = domain.Edge(vertex, domain.Vertex(*coordinates[4]))
        envelope = domain.EnvelopePolygonalDomain.__new__(domain.EnvelopePolygonalDomain)

        # Decimal endpoint reconstruction differs by roughly 2e-16 here.
        self.assertNotEqual(edge.m * vertex.x + edge.q, vertex.y)
        self.assertEqual(
            envelope._solve_one_vertex_one_edge((vertex, edge), []),
            [],
        )

    def test_kkt_feasibility_is_scale_invariant(self):
        base = domain.VALIDATION_INSTANCES["small_scale"]
        three_vertex_cell_counts = []

        for factor in (1, 1000):
            coordinates = tuple((factor * px, factor * py) for px, py in base)
            polygon = domain.Polygon(tuple(domain.Vertex(*point) for point in coordinates))
            envelope = domain.EnvelopePolygonalDomain.__new__(domain.EnvelopePolygonalDomain)
            envelope.polygon = polygon
            envelope.generators = envelope._determine_generators()

            with patch("sys.stdout", new_callable=io.StringIO):
                solutions = envelope._three_elements_J()

            three_vertex_cell_counts.append(sum(
                all(isinstance(generator, domain.Vertex) for generator in generators)
                for generators, _, _ in solutions
            ))

        self.assertEqual(three_vertex_cell_counts, [0, 0])

    def test_convexified_staircase_has_no_endpoint_sliver_cells(self):
        coordinates = domain.VALIDATION_INSTANCES["original_convexified_staircase"]
        polygon = domain.Polygon(tuple(domain.Vertex(*point) for point in coordinates))

        with patch.object(domain, "plot_cell"), \
                patch.object(domain.EnvelopePolygonalDomain, "_validation"), \
                patch("sys.stdout", new_callable=io.StringIO):
            envelope = domain.EnvelopePolygonalDomain(polygon)

        two_vertex_one_edge_cells = []
        for generators, _, functional in envelope.all_solutions:
            edges = [generator for generator in generators if isinstance(generator, domain.Edge)]
            if len(generators) != 3 or len(edges) != 1:
                continue

            two_vertex_one_edge_cells.append((generators, functional))
            edge = edges[0]
            a = float(functional.coeff(x))
            b = float(functional.coeff(y))
            contact_x = (a + edge.m * b - edge.q) / (2 * edge.m)
            self.assertTrue(domain.active_point_is_strictly_inside_edge(edge, contact_x))

        self.assertEqual(two_vertex_one_edge_cells, [])


if __name__ == "__main__":
    unittest.main()
