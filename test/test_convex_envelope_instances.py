"""Quality checks for the polygon instances used during validation."""

import contextlib
import io
import re
import unittest
from unittest.mock import patch


from alpaca.locatelli import convex_envelope_polygonal_domain as domain


# The error is normalized by the observed reference-value range, avoiding
# scale-dependent absolute thresholds across different polygon instances.
MAX_RELATIVE_AVERAGE_ERROR = 0.05
MAX_RELATIVE_POINTWISE_ERROR = 0.05
MAX_EXCEPTIONAL_POINT_FRACTION = 0.01
MAX_EXCEPTIONAL_POINTS = 3
MAX_OVERLAP_DIFFERENCE_RELATIVE = 1e-8


class TestValidationInstances(unittest.TestCase):
    def test_all_validation_instances(self):
        summaries = {}
        quality_metrics = {}
        covered_generator_sets = set()
        for name, coordinates in domain.VALIDATION_INSTANCES.items():
            with self.subTest(instance=name):
                polygon = domain.Polygon(tuple(domain.Vertex(*point) for point in coordinates))
                x_span = max(point[0] for point in coordinates) - min(point[0] for point in coordinates)
                y_span = max(point[1] for point in coordinates) - min(point[1] for point in coordinates)
                span = max(x_span, y_span)
                relative_step = span / 20
                output = io.StringIO()
                with patch.object(domain, "plot_cell"), \
                        patch.object(domain, "VALIDATION_DISCRETIZATION", relative_step), \
                        contextlib.redirect_stdout(output):
                    envelope = domain.EnvelopePolygonalDomain(polygon)

                for generators, _, _ in envelope.all_solutions:
                    covered_generator_sets.add((
                        len(generators),
                        sum(isinstance(generator, domain.Vertex) for generator in generators),
                    ))

                text = output.getvalue()
                summaries[name] = text
                coverage = re.search(
                    r"Numbers of undefined and infeasible \(but not undefined\) points are:\s+(\d+)\s+(\d+)",
                    text,
                )
                compared_points = re.search(r"Number of compared grid points is:\s+(\d+)", text)
                average_error = re.search(
                    r"Average absolute error for each compared grid point is:\s+([\deE+.-]+)", text
                )
                relative_average_error = re.search(
                    r"Relative average absolute error for each compared grid point is:\s+([\deE+.-]+)", text
                )
                relative_maximum_error = re.search(
                    r"Relative maximum absolute error over compared grid points is:\s+([\deE+.-]+)", text
                )
                largest_difference = re.search(
                    r"Largest difference between analytical cell values is:\s+([\deE+.-]+)", text
                )
                nan_values = re.search(r"Number of nan functional values is:\s+(\d+)", text)
                non_finite_values = re.search(
                    r"Number of non-finite or non-real functional values is:\s+(\d+)", text
                )

                self.assertIsNotNone(coverage, name)
                self.assertIsNotNone(compared_points, name)
                self.assertIsNotNone(average_error, name)
                self.assertIsNotNone(relative_average_error, name)
                self.assertIsNotNone(relative_maximum_error, name)
                self.assertIsNotNone(nan_values, name)
                self.assertIsNotNone(non_finite_values, name)

                undefined_points = int(coverage.group(1))
                infeasible_points = int(coverage.group(2))
                evaluated_points = int(compared_points.group(1))
                allowed_exceptional_points = min(
                    MAX_EXCEPTIONAL_POINTS,
                    max(1, evaluated_points // int(1 / MAX_EXCEPTIONAL_POINT_FRACTION)),
                )

                self.assertEqual(infeasible_points, 0, name)
                self.assertGreater(evaluated_points, 0, name)
                self.assertLessEqual(undefined_points, allowed_exceptional_points, name)
                self.assertLessEqual(int(nan_values.group(1)), allowed_exceptional_points, name)
                self.assertLessEqual(int(non_finite_values.group(1)), allowed_exceptional_points, name)
                self.assertLessEqual(
                    float(relative_average_error.group(1)),
                    MAX_RELATIVE_AVERAGE_ERROR,
                    name,
                )
                self.assertLessEqual(
                    float(relative_maximum_error.group(1)),
                    MAX_RELATIVE_POINTWISE_ERROR,
                    name,
                )
                quality_metrics[name] = (
                    float(relative_average_error.group(1)),
                    float(relative_maximum_error.group(1)),
                )

                if largest_difference is None:
                    self.assertIn("No grid point with multiple valid cell values found.", text, name)
                else:
                    self.assertLessEqual(
                        float(largest_difference.group(1)),
                        MAX_OVERLAP_DIFFERENCE_RELATIVE * x_span * y_span,
                        name,
                    )

        self.assertEqual(set(summaries), set(domain.VALIDATION_INSTANCES))
        self.assertEqual(
            covered_generator_sets,
            {
                (2, 1),  # vertex--edge
                (2, 0),  # edge--edge
                (3, 3),  # three vertices
                (3, 2),  # two vertices--one edge
                (3, 1),  # one vertex--two edges
                (3, 0),  # three edges
            },
        )
        for scaled_instance in ("small_scale", "large_scale"):
            for actual, expected in zip(
                    quality_metrics[scaled_instance],
                    quality_metrics["quadrilateral_base"],
            ):
                self.assertAlmostEqual(actual, expected, places=12, msg=scaled_instance)


if __name__ == "__main__":
    unittest.main()
