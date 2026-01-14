# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np

from alpaca.model_data import model_data as md, variable as var, constraint as con
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger
import alpaca.utils.geometry as uge
import alpaca.expressions.bilinear_expression as ble
from alpaca.locatelli import (
    domain_projector as dop,
    locatelli_cut_generator as lcg,
)
import alpaca.settings as s


class StairLocatelli:
    """Adds stair locatelli cuts to model data."""

    def __init__(self, model_data: md.ModelData):
        logger.info(lsf.info_init_stair_locatelli())
        self.model_data = model_data
        self.settings = model_data.settings

        # Initialize components
        self.projector = dop.DomainProjector(model_data)
        self.cut_generator = lcg.LocatelliCutGenerator(model_data)

        # Storage for results
        self.bilinear_projected_domains: list[
            tuple[ble.BilinearExpression, list[tuple[float, float]]]
        ] = []

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
            self.bilinear_projected_domains.append((expr, vertices))

            # 2. Generate Cuts (Constraint creation)
            cuts_added = self.cut_generator.generate_cuts(expr, vertices)
            total_cuts += cuts_added

        logger.info(lsf.info_total_stair_locatelli_cuts_added(total_cuts))

    def calculate_mean_bilinear_relaxation_volume_improvement(self):
        """Calculate metrics for the improvement provided by these cuts."""
        total_volume_imp = 0.0
        bilinear_expressions = self.model_data.expressions.bilinear_expressions.values()

        for expr in bilinear_expressions:
            vol = self.calculate_volume_improvement_bilinear_expression(expr)
            total_volume_imp += vol

        if not bilinear_expressions:
            return -1.0

        return total_volume_imp / len(bilinear_expressions)

    def calculate_volume_improvement_bilinear_expression(
        self, bilinear_expression: ble.BilinearExpression
    ):
        """
        Calculate volume and max difference improvement of bilinear relaxation.
        Refactored for readability.
        """
        x = bilinear_expression.variables[0]
        y = bilinear_expression.variables[1]
        domain_vertices = [
            vertices
            for bl_exp, vertices in self.bilinear_projected_domains
            if bl_exp.name == bilinear_expression.name
        ][0]
        if (
            abs(x.ub - x.lb) < s.StaticSettings.feasibility_tolerance
            or abs(y.ub - y.lb) < s.StaticSettings.feasibility_tolerance
        ):
            return 0.0

        mc_cormick_volumes = []
        grid_size = (
            self.model_data.settings.feature_stair_locatelli_evaluation_grid_size
        )
        x_range = np.linspace(x.lb, x.ub, grid_size)
        y_range = np.linspace(y.lb, y.ub, grid_size)

        # Calculate area of a single grid cell for volume integration
        if grid_size > 1:
            dx = (x.ub - x.lb) / (grid_size - 1)
            dy = (y.ub - y.lb) / (grid_size - 1)
            cell_area = dx * dy
        else:
            cell_area = 0.0

        for x_grid in x_range:
            for y_grid in y_range:
                mc_height = self._calculate_mc_cormick_size_at_point(
                    (x_grid, y_grid), (x, y), bilinear_expression
                )
                # Multiply height by area to get volume of the column
                mc_cormick_volumes.append(mc_height * cell_area)

        return self._calculate_improvement_stats(
            mc_cormick_volumes,
            uge.calculate_locatelli_volume(
                x_range,
                y_range,
                domain_vertices,
                self.settings.feature_stair_locatelli == 1,
            ),
        )

    @staticmethod
    def _calculate_max_z_interval(x: var.Variable, y: var.Variable):
        """Calculates the maximum possible range of z = x*y given bounds of x and y."""
        corners = [x.ub * y.ub, x.lb * y.lb, x.lb * y.ub, x.ub * y.lb]
        return max(corners) - min(corners)

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

    def _calculate_mc_cormick_size_at_point(
        self,
        values: tuple[float, float],
        variables: tuple[var.Variable, var.Variable],
        expr,
    ):
        """
        Calculates the size of the McCormick interval and the Locatelli interval
        at a specific grid point.
        """
        x_val, y_val = values
        x_var, y_var = variables
        mc_upper = min(
            self._get_constraint_value(c, (x_val, y_val), (x_var, y_var))
            for c in expr.mc_cormick_constraints["overestimator"]
        )
        mc_lower = max(
            self._get_constraint_value(c, (x_val, y_val), (x_var, y_var))
            for c in expr.mc_cormick_constraints["underestimator"]
        )
        return mc_upper - mc_lower

    @staticmethod
    def _calculate_improvement_stats(
        mc_volumes: list[float], locatelli_volume: float
    ) -> float:
        """Calculates the final volume and max difference improvement metrics."""
        total_mc_volume = sum(mc_volumes)
        if total_mc_volume == 0:
            return 0.0
        volume_improvement = (total_mc_volume - locatelli_volume) / total_mc_volume
        return volume_improvement
