# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np

from alpaca.model_data import model_data as md, variable as var, constraint as con
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger
import alpaca.utils.geometry as uge
import alpaca.expressions.bilinear_expression as ble
from alpaca.locatelli import (
    domain_projector as dop,
    locatelli_cut_generator as lcg,
)
import alpaca.model_buildup.multilinear_handler as mlh
import alpaca.settings as s


class StairLocatelli:
    """Adds stair locatelli cuts to model data."""

    def __init__(self, model_data: md.ModelData, locatelli_vertices=None):
        logger.info(lsf.info_init_stair_locatelli())
        self.model_data = model_data
        self.settings = model_data.settings
        self.locatelli_vertices = (
            locatelli_vertices if locatelli_vertices is not None else {}
        )

        # Initialize components
        self.projector = dop.DomainProjector(model_data, self.locatelli_vertices)
        self.cut_generator = lcg.LocatelliCutGenerator(model_data)

        # Storage for results
        self.bilinear_projected_domains_polygon: list[
            tuple[ble.BilinearExpression, list[tuple[float, float]]]
        ] = []
        self.nr_cuts = 0

        # Execute
        self._run()

    def _run(self):
        """Orchestrate the cut generation process."""
        total_cuts = 0

        for expr in self.model_data.expressions.bilinear_expressions.values():
            # 1. Project Domain (Solver)
            vertices = (
                self.projector.get_projected_vertices(expr)
                if self.settings.feature_stair_locatelli <= 2
                else self.projector.get_projected_vertices_indicator(expr)
            )
            vertices = uge.straighten_vertices(vertices)
            if self.settings.feature_stair_locatelli <= 2:
                self.locatelli_vertices[expr.name] = vertices
            self.bilinear_projected_domains_polygon.append((expr, vertices))
            if self.settings.feature_stair_locatelli == 1:
                vertices = uge.calculate_convex_hull_2d(vertices)

            # 2. Generate Cuts (Constraint creation)
            cuts_added = self.cut_generator.generate_cuts(expr, vertices)
            total_cuts += cuts_added

        self.nr_cuts = total_cuts
        logger.info(lsf.info_total_stair_locatelli_cuts_added(total_cuts))

    def calculate_mean_bilinear_domain_volume(self, is_polytope=True):
        """Calculate 3d volume over polytope or polygon domain."""
        total_volume = 0.0
        bilinear_expressions = self.model_data.expressions.bilinear_expressions.values()

        for expr in bilinear_expressions:
            total_volume += self.calculate_bilinear_domain_volume(
                expr, is_polytope=is_polytope
            )

        if not bilinear_expressions:
            return 0.0

        return total_volume / len(bilinear_expressions)

    @classmethod
    def calculate_mean_bilinear_domain_volume_box(
        cls, model_data: md.ModelData, added_mccormick_envelopes
    ):
        """Calculate 3d volume over box domain."""
        if not added_mccormick_envelopes:
            multilinear_handler = mlh.MultilinearHandler(model_data)
            multilinear_handler.add_mccormick_envelopes()
        total_volume = 0.0
        bilinear_expressions = model_data.expressions.bilinear_expressions.values()

        for expr in bilinear_expressions:
            total_volume += cls.calculate_bilinear_domain_volume_box(
                expr, model_data.settings
            )

        if not bilinear_expressions:
            return 0.0

        return total_volume / len(bilinear_expressions)

    @classmethod
    def calculate_bilinear_domain_volume_box(
        cls, bilinear_expression: ble.BilinearExpression, settings: s.UserSettings
    ):
        """
        Calculate volume of mccormick relaxation.
        """
        cell_area, x_range, y_range = cls._create_evaluation_grid(
            bilinear_expression, settings
        )
        mc_cormick_volume = 0.0
        for x_grid in x_range:
            for y_grid in y_range:
                mc_height = cls._calculate_mc_cormick_size_at_point(
                    (x_grid, y_grid), bilinear_expression
                )
                mc_cormick_volume += mc_height * cell_area
        return mc_cormick_volume

    def calculate_bilinear_domain_volume(
        self, bilinear_expression: ble.BilinearExpression, is_polytope=True
    ):
        """
        Calculate volume and max difference improvement of bilinear relaxation.
        Refactored for readability.
        """
        x = bilinear_expression.variables[0]
        y = bilinear_expression.variables[1]
        polygon_domain_vertices = [
            vertices
            for bl_exp, vertices in self.bilinear_projected_domains_polygon
            if bl_exp.name == bilinear_expression.name
        ][0]
        if len(polygon_domain_vertices) < 3:
            return 0.0
        if (
            abs(x.ub - x.lb) < s.StaticSettings.feasibility_tolerance
            or abs(y.ub - y.lb) < s.StaticSettings.feasibility_tolerance
        ):
            return 0.0
        if sum(x_val * y_val for x_val, y_val in polygon_domain_vertices) == 0.0:
            return 0.0
        return self.calculate_3d_volume_polygon_over_domain(
            bilinear_expression, polygon_domain_vertices, is_polytope=is_polytope
        )

    @classmethod
    def _create_evaluation_grid(
        cls, bilinear_expression: ble.BilinearExpression, settings: s.UserSettings
    ):
        """Creates the evaluation grid for volume calculation."""
        x = bilinear_expression.variables[0]
        y = bilinear_expression.variables[1]
        grid_size = settings.feature_stair_locatelli_evaluation_grid_size
        x_range = np.linspace(x.lb, x.ub, grid_size)
        y_range = np.linspace(y.lb, y.ub, grid_size)

        # Calculate area of a single grid cell for volume integration
        if grid_size > 1:
            dx = (x.ub - x.lb) / (grid_size - 1)
            dy = (y.ub - y.lb) / (grid_size - 1)
            cell_area = dx * dy
        else:
            cell_area = 0.0
        return cell_area, x_range, y_range

    def calculate_3d_volume_polygon_over_domain(
        self,
        bilinear_expression: ble.BilinearExpression,
        domain_vertices,
        is_polytope=True,
    ) -> float:
        """Calculates the 3D volume under the McCormick envelope over the grid."""
        cell_area, x_range, y_range = self._create_evaluation_grid(
            bilinear_expression, self.model_data.settings
        )
        poly_volume = 0.0
        poly_grid = (
            uge.calculate_x_y_domain_polytope(x_range, y_range, domain_vertices)
            if is_polytope
            else uge.calculate_x_y_domain_polygon(x_range, y_range, domain_vertices)
        )
        convexified_area = None
        if self.settings.feature_stair_locatelli == 1:
            convexified_area = uge.calculate_convexified_area(
                x_range, y_range, domain_vertices
            )
        for x_grid_point, y_grid_point in poly_grid:
            if self.settings.feature_stair_locatelli == 1:
                # Standard locatelli, calculate height based on convex hull over the polytope.
                poly_height = uge.calculate_feasible_height_convexified(
                    (x_grid_point, y_grid_point), convexified_area
                )
            else:
                poly_height = self._calculate_feasible_height(
                    (x_grid_point, y_grid_point), bilinear_expression
                )
            # Multiply height by area to get volume of the column
            poly_volume += poly_height * cell_area
        return poly_volume

    def calculate_3d_volume_polytope_over_2d_polygon(
        self, bilinear_expression: ble.BilinearExpression, domain_vertices
    ) -> float:
        """Calculates the 3D volume under the McCormick envelope over the grid."""
        cell_area, x_range, y_range = self._create_evaluation_grid(
            bilinear_expression, self.model_data.settings
        )

        polytope_volume = 0.0
        for x_grid, y_grid in uge.calculate_x_y_domain_polygon(
            x_range, y_range, domain_vertices
        ):
            polytope_height = self._calculate_feasible_height(
                (x_grid, y_grid), bilinear_expression
            )
            # Multiply height by area to get volume of the column
            polytope_volume += polytope_height * cell_area
        return polytope_volume

    @staticmethod
    def _get_constraint_value(
        constraint: con.LinearConstraint,
        values: tuple[float, float],
        variables: tuple[var.Variable, var.Variable],
    ):
        """
        Evaluates a single constraint at a specific x, y point.
        Logic: -coeff_x * x - coeff_y * y + rhs
        """
        x_val, y_val = values
        x_var, y_var = variables
        x_coeff = -next(c for c, v in constraint.variables if v == x_var)
        y_coeff = -next(c for c, v in constraint.variables if v == y_var)
        return x_coeff * x_val + y_coeff * y_val + constraint.rhs

    @classmethod
    def _calculate_mc_cormick_size_at_point(
        cls,
        values: tuple[float, float],
        expr,
    ):
        """
        Calculates the size of the McCormick interval and the Locatelli interval
        at a specific grid point.
        """
        x_val, y_val = values
        x_var, y_var = expr.variables
        mc_upper = min(
            cls._get_constraint_value(c, (x_val, y_val), (x_var, y_var))
            for c in expr.mc_cormick_constraints["overestimator"]
        )
        mc_lower = max(
            cls._get_constraint_value(c, (x_val, y_val), (x_var, y_var))
            for c in expr.mc_cormick_constraints["underestimator"]
        )
        return mc_upper - mc_lower

    def _calculate_feasible_height(
        self,
        values: tuple[float, float],
        expr,
    ):
        """
        Calculates the size of the McCormick interval and the Locatelli interval
        at a specific grid point.
        """
        x_val, y_val = values
        x_var, y_var = expr.variables
        mc_upper = min(
            self._get_constraint_value(c, (x_val, y_val), (x_var, y_var))
            for c in expr.linear_relaxation_for_bilinear["overestimator"]
        )
        mc_lower = max(
            self._get_constraint_value(c, (x_val, y_val), (x_var, y_var))
            for c in expr.linear_relaxation_for_bilinear["underestimator"]
        )
        return max(0, mc_upper - mc_lower)
