# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import numpy as np

from alpaca.model_data import constraint as con
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
import alpaca.expressions.bilinear_expression as ble
from alpaca.model_data import model_data as mda
import alpaca.settings as s
import alpaca.utils.geometry as geo


class LocatelliCutGenerator:
    """Generates Locatelli cuts for bilinear expressions based on projected domains."""

    def __init__(self, model_data: mda.ModelData):
        self.model_data = model_data

    def generate_cuts(
        self,
        bilinear_expression: ble.BilinearExpression,
        vertices: list[tuple[float, float]],
    ) -> int:
        """
        Generates and adds cuts for a specific bilinear expression based on its vertices.
        Returns the number of cuts added.
        """
        if len(vertices) < 3:
            return 0

        checkpoints = self._generate_edge_checkpoints(vertices)
        cut_count = 0

        # Iterate over all combinations of 3 vertices to find valid cutting planes
        for v1, v2, v3 in itertools.combinations(vertices, 3):
            if self._process_vertex_combination(
                bilinear_expression, (v1, v2, v3), checkpoints, cut_count
            ):
                cut_count += 1

        return cut_count

    @staticmethod
    def _generate_edge_checkpoints(
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Generates test points along the edges of the polygon."""
        checkpoints = []
        looped_vertices = vertices + [vertices[0]]
        for (vfx, vfy), (vtx, vty) in zip(looped_vertices, looped_vertices[1:]):
            for i in range(20):
                point_x = vfx + (vtx - vfx) * (i / 20)
                point_y = vfy + (vty - vfy) * (i / 20)
                checkpoints.append((point_x, point_y))
        return checkpoints

    def _process_vertex_combination(
        self,
        expr: ble.BilinearExpression,
        vertex_triple: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        checkpoints: list[tuple[float, float]],
        cut_index: int,
    ) -> bool:
        if geo.check_if_vertices_are_collinear(*vertex_triple):
            return False

        success, coeffs = geo.calculate_hyperplane_for_vertex_triple(*vertex_triple)
        if not success:
            return False

        con_type = self._check_if_hyperplane_is_infeasible(coeffs, checkpoints)
        if con_type == lsf.empty_string():
            return False

        self._create_and_add_cut(expr, cut_index, coeffs, con_type)
        return True

    @staticmethod
    def _check_if_hyperplane_is_infeasible(
        coeffs: tuple[float, float, float], checkpoints: list[tuple[float, float]]
    ) -> str:
        x_coeff, y_coeff, const = coeffs

        # Check if entire polygon is ABOVE the plane (<= constraint)
        # Plane: z = ax + by + c.
        # Condition: ax + by + c <= xy for all boundary points
        if np.all(
            [
                x_coeff * x + y_coeff * y + const
                <= x * y + s.StaticSettings.feasibility_tolerance
                for x, y in checkpoints
            ]
        ):
            return (
                lsf.constraint_geq()
            )  # Wait, if plane <= xy, then z >= plane is valid underestimator?
            # Original code mapping:
            # if plane <= xy -> returns constraint_geq() -> underestimator. Correct.

        # Check if entire polygon is BELOW the plane (>= constraint)
        if np.all(
            [
                x_coeff * x + y_coeff * y + const
                >= x * y - s.StaticSettings.feasibility_tolerance
                for x, y in checkpoints
            ]
        ):
            return lsf.constraint_leq()  # Overestimator

        return lsf.empty_string()

    def _create_and_add_cut(
        self,
        expr: ble.BilinearExpression,
        index: int,
        coeffs: tuple[float, float, float],
        con_type: str,
    ):
        x_coeff, y_coeff, const = coeffs
        name = f"locatelli_{expr.name}_{index}"

        # z >= ax + by + c  =>  z - ax - by >= c
        cut = con.LinearConstraint(
            name,
            con_type=con_type,
            rhs=const,
            variables=[
                (-x_coeff, expr.variables[0]),
                (-y_coeff, expr.variables[1]),
                (1.0, expr.representative_variable),
            ],
        )
        self.model_data.add_constraint(cut)

        if con_type == lsf.constraint_leq():
            expr.linear_relaxation_for_bilinear["overestimator"].append(cut)
        elif con_type == lsf.constraint_geq():
            expr.linear_relaxation_for_bilinear["underestimator"].append(cut)
