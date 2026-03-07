# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np

from alpaca.model_data import variable as var
from alpaca.external_solvers import mip_model as mm
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
import alpaca.expressions.bilinear_expression as ble
import alpaca.settings as s
import alpaca.utils.geometry as geo


class DomainProjector:
    """Calculates polygon containing the feasible set
    of the projected domain of a bilinear term."""

    def __init__(self, model_data):
        self.model_data = model_data
        self.settings = model_data.settings
        self.external_solver = mm.MIPModel(model_data)
        self._configure_solver()

    def _configure_solver(self):
        """Sets time limits and hides output for the internal solver."""
        bilinear_count = len(self.model_data.expressions.bilinear_expressions)
        if bilinear_count == 0:
            return
        limit = (
            self.settings.feature_stair_locatelli_obbt_time_limit
            / (bilinear_count * 2 * self.settings.feature_stair_locatelli_grid_size)
            if self.settings.feature_stair_locatelli <= 2
            else (
                self.settings.feature_stair_locatelli_obbt_time_limit
                * 0.5
                / bilinear_count
            )
        )
        self.external_solver.opt_model.set_time_limit(limit)
        self.external_solver.opt_model.hide_output()

    def get_projected_vertices_indicator(
        self, bilinear_expression: ble.BilinearExpression
    ) -> list[tuple[float, float]]:
        """
        Compute polygon for indicator bilinear terms.
        """
        x, y = tuple(bilinear_expression.variables)

        self.external_solver.opt_model.set_variable_lb(
            x.solver_variable, x.lb + self.settings.feature_stair_locatelli_mu
        )
        self.external_solver.opt_model.set_objective(
            y.solver_variable, sense=lsf.objective_sense_maximize()
        )
        self.external_solver.opt_model.optimize()
        y_ub = (
            self.external_solver.opt_model.get_objective_bound()
            if not self.external_solver.opt_model.is_infeasible()
            else None
        )
        self.external_solver.opt_model.set_variable_lb(x.solver_variable, x.lb)
        if y_ub is None:
            return [(x.lb, y.lb), (x.lb, y.ub)]

        self.external_solver.opt_model.set_variable_lb(
            y.solver_variable, y.lb + self.settings.feature_stair_locatelli_mu
        )
        self.external_solver.opt_model.set_objective(
            x.solver_variable, sense=lsf.objective_sense_maximize()
        )
        self.external_solver.opt_model.optimize()
        x_ub = (
            self.external_solver.opt_model.get_objective_bound()
            if not self.external_solver.opt_model.is_infeasible()
            else None
        )
        self.external_solver.opt_model.set_variable_lb(y.solver_variable, y.lb)
        if x_ub is None:
            return [(x.lb, y.lb), (x.ub, y.lb)]
        return geo.filter_equal_vertices(
            [
                (x.lb, y.lb),
                (x.lb, y.ub),
                (x.lb, y_ub),
                (x_ub, y_ub),
                (x_ub, y.lb),
                (x.ub, y.lb),
            ]
        )

    def get_projected_vertices(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self, bilinear_expression: ble.BilinearExpression
    ) -> list[tuple[float, float]]:
        """
        Main entry point to find the feasible polygon for a bilinear expression.
        Executes the exact scanning and tracing logic from the original StairLocatelli.
        """
        x, y = tuple(bilinear_expression.variables)

        # Initialize boundary trackers
        max_x_conditions = []
        max_x_boundary_values = {y.lb: -np.inf}
        max_y_conditions = []
        max_y_boundary_values = {x.lb: -np.inf}
        min_x_conditions = []
        min_x_boundary_values = {y.lb: np.inf}
        min_y_conditions = []
        min_y_boundary_values = {x.lb: np.inf}

        # Define the boundary lookup closures exactly as in original
        def max_y(x_value: float):
            if x_value in max_y_boundary_values:
                return max_y_boundary_values[x_value]
            for cond, val in max_y_conditions:
                if cond(x_value):
                    return val
            return None

        def max_x(y_value: float):
            if y_value in max_x_boundary_values:
                return max_x_boundary_values[y_value]
            for cond, val in max_x_conditions:
                if cond(y_value):
                    return val
            return None

        def min_y(x_value: float):
            if x_value in min_y_boundary_values:
                return min_y_boundary_values[x_value]
            for cond, val in min_y_conditions:
                if cond(x_value):
                    return val
            return None

        def min_x(y_value: float):
            if y_value in min_x_boundary_values:
                return min_x_boundary_values[y_value]
            for cond, val in min_x_conditions:
                if cond(y_value):
                    return val
            return None

        # Setup domains
        x_domain = np.linspace(
            x.lb, x.ub, self.settings.feature_stair_locatelli_grid_size
        ).tolist()
        y_domain = np.linspace(
            y.lb, y.ub, self.settings.feature_stair_locatelli_grid_size
        ).tolist()

        y_grid_addition = []
        x_grid_addition = []

        # --- PASS 1: Maximize Y over X intervals ---
        self.external_solver.opt_model.set_objective(
            y.solver_variable, sense=lsf.objective_sense_maximize()
        )
        for lb, ub in zip(x_domain, x_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(x, lb, ub)
            solution_value = (
                y.ub if solution_value is None else min(solution_value, y.ub)
            )
            if feasible:
                close_entry_in_y_grid = self._find_closest_entry_in_list(
                    y_grid_addition + y_domain, solution_value
                )
                if close_entry_in_y_grid is not None:
                    new_y_grid_value = close_entry_in_y_grid
                else:
                    y_grid_addition.append(solution_value)
                    new_y_grid_value = solution_value

                max_y_boundary_values[lb] = max(
                    max_y_boundary_values.get(lb, -np.inf), new_y_grid_value
                )
                max_y_boundary_values[ub] = new_y_grid_value
                max_y_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_y_grid_value)
                )

        # --- PASS 2: Minimize Y over X intervals ---
        self.external_solver.opt_model.set_objective(
            y.solver_variable, sense=lsf.objective_sense_minimize()
        )
        for lb, ub in zip(x_domain, x_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(x, lb, ub)
            solution_value = (
                y.lb if solution_value is None else max(solution_value, y.lb)
            )
            if feasible:
                close_entry_in_y_grid = self._find_closest_entry_in_list(
                    y_grid_addition + y_domain, solution_value
                )
                if close_entry_in_y_grid is not None:
                    new_y_grid_value = close_entry_in_y_grid
                else:
                    y_grid_addition.append(solution_value)
                    new_y_grid_value = solution_value

                min_y_boundary_values[lb] = min(
                    min_y_boundary_values.get(lb, np.inf), new_y_grid_value
                )
                min_y_boundary_values[ub] = new_y_grid_value
                min_y_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_y_grid_value)
                )

        # Reset X bounds before scanning Y
        self.external_solver.opt_model.set_variable_lb(x.solver_variable, x.lb)
        self.external_solver.opt_model.set_variable_ub(x.solver_variable, x.ub)

        # --- PASS 3: Maximize X over Y intervals ---
        self.external_solver.opt_model.set_objective(
            x.solver_variable, sense=lsf.objective_sense_maximize()
        )
        for lb, ub in zip(y_domain, y_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(y, lb, ub)
            solution_value = (
                x.ub if solution_value is None else min(solution_value, x.ub)
            )
            if feasible:
                close_entry_in_x_grid = self._find_closest_entry_in_list(
                    x_grid_addition + x_domain, solution_value
                )
                if close_entry_in_x_grid is not None:
                    new_x_grid_value = close_entry_in_x_grid
                else:
                    x_grid_addition.append(solution_value)
                    new_x_grid_value = solution_value

                max_x_boundary_values[lb] = max(
                    max_x_boundary_values.get(lb, -np.inf), new_x_grid_value
                )
                max_x_boundary_values[ub] = new_x_grid_value
                max_x_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_x_grid_value)
                )

        # --- PASS 4: Minimize X over Y intervals ---
        self.external_solver.opt_model.set_objective(
            x.solver_variable, sense=lsf.objective_sense_minimize()
        )
        for lb, ub in zip(y_domain, y_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(y, lb, ub)
            solution_value = (
                x.lb if solution_value is None else max(solution_value, x.lb)
            )
            if feasible:
                close_entry_in_x_grid = self._find_closest_entry_in_list(
                    x_grid_addition + x_domain, solution_value
                )
                if close_entry_in_x_grid is not None:
                    new_x_grid_value = close_entry_in_x_grid
                else:
                    x_grid_addition.append(solution_value)
                    new_x_grid_value = solution_value

                min_x_boundary_values[lb] = min(
                    min_x_boundary_values.get(lb, np.inf), new_x_grid_value
                )
                min_x_boundary_values[ub] = new_x_grid_value
                min_x_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_x_grid_value)
                )

        # Reset Y bounds
        self.external_solver.opt_model.set_variable_lb(y.solver_variable, y.lb)
        self.external_solver.opt_model.set_variable_ub(y.solver_variable, y.ub)

        # --- Point Feasibility Check ---
        x_grid = sorted(x_domain + x_grid_addition)
        y_grid = sorted(y_domain + y_grid_addition)
        feasible_grid_points = []

        for x_grid_point in x_grid:
            for y_grid_point in y_grid:
                y_ub = max_y(x_grid_point)
                if y_ub is not None and y_grid_point > y_ub:
                    continue
                y_lb = min_y(x_grid_point)
                if y_lb is not None and y_grid_point < y_lb:
                    continue
                x_ub = max_x(y_grid_point)
                if x_ub is not None and x_grid_point > x_ub:
                    continue
                x_lb = min_x(y_grid_point)
                if x_lb is not None and x_grid_point < x_lb:
                    continue
                feasible_grid_points.append((x_grid_point, y_grid_point))

        if len(feasible_grid_points) <= 1:
            return []

        # --- Trace Boundary Logic ---
        vertices = self._trace_boundary_polygon(feasible_grid_points)

        # --- Post Processing ---
        vertices = geo.filter_equal_vertices(vertices)
        vertices = geo.filter_collinear_vertices(vertices)

        return vertices

    def _trace_boundary_polygon(
        self, feasible_grid_points: list[tuple[float, float]]
    ):  # pylint: disable=too-many-branches
        """
        Left hand rule for mazes.
        """
        start_vertex = self._find_start_vertex(feasible_grid_points)
        if start_vertex == (None, None):
            return []

        # Initialize tracing variables
        current_vertex = start_vertex
        current_direction = -1
        traversed = []

        # Directions: 0=Up, 1=Right, 2=Down, 3=Left
        while True:
            for i in range(4):
                next_direction = (current_direction - 1 + i) % 4
                next_vertex = (-42, -42)

                if next_direction == 0:  # up
                    candidate_next_vertices = [
                        y_coord
                        for x_coord, y_coord in feasible_grid_points
                        if x_coord == current_vertex[0] and y_coord > current_vertex[1]
                    ]
                    if candidate_next_vertices:
                        next_vertex = (current_vertex[0], min(candidate_next_vertices))
                        break

                elif next_direction == 1:  # right
                    candidate_next_vertices = [
                        x_coord
                        for x_coord, y_coord in feasible_grid_points
                        if y_coord == current_vertex[1] and x_coord > current_vertex[0]
                    ]
                    if candidate_next_vertices:
                        next_vertex = (min(candidate_next_vertices), current_vertex[1])
                        break

                elif next_direction == 2:  # down
                    candidate_next_vertices = [
                        y_coord
                        for x_coord, y_coord in feasible_grid_points
                        if x_coord == current_vertex[0] and y_coord < current_vertex[1]
                    ]
                    if candidate_next_vertices:
                        next_vertex = (current_vertex[0], max(candidate_next_vertices))
                        break

                elif next_direction == 3:  # left
                    candidate_next_vertices = [
                        x_coord
                        for x_coord, y_coord in feasible_grid_points
                        if y_coord == current_vertex[1] and x_coord < current_vertex[0]
                    ]
                    if candidate_next_vertices:
                        next_vertex = (max(candidate_next_vertices), current_vertex[1])
                        break

            # Loop termination check
            if (current_vertex, next_direction) in traversed:
                break

            if current_direction != next_direction:
                traversed.append((current_vertex, next_direction))

            current_vertex = next_vertex
            current_direction = next_direction

        return [point for point, direction in traversed]

    def _find_start_vertex(self, feasible_grid_points: list[tuple[float, float]]):
        # Find start vertex: one that has no neighbor to the left
        for candidate_vertex in feasible_grid_points:
            candidate_next_vertices = [
                x_coord
                for x_coord, y_coord in feasible_grid_points
                if y_coord == candidate_vertex[1] and x_coord < candidate_vertex[0]
            ]
            if not candidate_next_vertices:
                return candidate_vertex
        return (None, None)

    def _get_solution_value_for_interval(
        self, variable: var.Variable, lb: float, ub: float
    ):
        self.external_solver.opt_model.set_variable_lb(variable.solver_variable, lb)
        self.external_solver.opt_model.set_variable_ub(variable.solver_variable, ub)
        self.external_solver.opt_model.optimize()
        if self.external_solver.opt_model.is_infeasible():
            return False, 0.0
        try:
            return True, self.external_solver.opt_model.get_objective_bound()
        except AttributeError:
            return True, None

    @staticmethod
    def _find_closest_entry_in_list(data, target):
        closest_val = min(data, key=lambda x: abs(x - target))
        if abs(closest_val - target) <= s.StaticSettings.feasibility_tolerance:
            return closest_val
        return None
