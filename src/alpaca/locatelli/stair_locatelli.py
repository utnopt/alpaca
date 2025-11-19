# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

from alpaca.model_data import model_data as md, variable as var, constraint as con
from alpaca.expressions import bilinear_expression as ble
from alpaca.external_solvers import mip_model as mm
import alpaca.settings as s
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger


class StairLocatelli:
    """Adds stair locatelli cuts to model data."""

    def __init__(self, model_data: md.ModelData):
        logger.info(lsf.info_init_stair_locatelli())
        self.model_data = model_data
        self.settings = model_data.settings
        self.external_solver = mm.MIPModel(model_data)
        self.external_solver.opt_model.set_time_limit(
            self.model_data.settings.feature_stair_locatelli_obbt_time_limit
            / (
                len(self.model_data.expressions.bilinear_expressions)
                * 2
                * self.settings.feature_stair_locatelli_grid_size
            )
        )
        self.external_solver.opt_model.hide_output()
        self.bilinear_projected_domains: list[
            tuple[ble.BilinearExpression, list[tuple[float, float]]]
        ] = []
        self._add_stair_locatelli_constraints()

    def _add_stair_locatelli_constraints(self):
        """Add stair locatelli cuts."""
        self._get_vertices_for_bilinear_projections()
        # self.plot_polygons()
        self._add_locatelli_cuts_from_vertices()

    def _get_vertices_for_bilinear_projections(self):
        for expression in self.model_data.expressions.bilinear_expressions.values():
            self._get_projected_domain_vertices_for_bilinear_expression(expression)

    def _get_projected_domain_vertices_for_bilinear_expression(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self, bilinear_expression: ble.BilinearExpression
    ):
        x, y = tuple(bilinear_expression.variables)

        max_x_conditions = []
        max_x_boundary_values = {y.lb: -np.inf}
        max_y_conditions = []
        max_y_boundary_values = {x.lb: -np.inf}
        min_x_conditions = []
        min_x_boundary_values = {y.lb: np.inf}
        min_y_conditions = []
        min_y_boundary_values = {x.lb: np.inf}

        def max_y(x_value):
            if x_value in max_y_boundary_values:
                return max_y_boundary_values[x_value]
            for cond, val in max_y_conditions:
                if cond(x_value):
                    return val
            return None

        def max_x(y_value):
            if y_value in max_x_boundary_values:
                return max_x_boundary_values[y_value]
            for cond, val in max_x_conditions:
                if cond(y_value):
                    return val
            return None

        def min_y(x_value):
            if x_value in min_y_boundary_values:
                return min_y_boundary_values[x_value]
            for cond, val in min_y_conditions:
                if cond(x_value):
                    return val
            return None

        def min_x(y_value):
            if y_value in min_x_boundary_values:
                return min_x_boundary_values[y_value]
            for cond, val in min_x_conditions:
                if cond(y_value):
                    return val
            return None

        x, y = tuple(bilinear_expression.variables)
        x_domain = np.linspace(
            x.lb, x.ub, self.settings.feature_stair_locatelli_grid_size
        ).tolist()
        y_domain = np.linspace(
            y.lb, y.ub, self.settings.feature_stair_locatelli_grid_size
        ).tolist()
        y_grid_addition = []
        x_grid_addition = []
        self.external_solver.opt_model.set_objective(
            y.solver_variable, sense=lsf.objective_sense_maximize()
        )
        for lb, ub in zip(x_domain, x_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(x, lb, ub)
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
                    max_y_boundary_values[lb], new_y_grid_value
                )
                max_y_boundary_values[ub] = new_y_grid_value
                max_y_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_y_grid_value)
                )

        self.external_solver.opt_model.set_objective(
            y.solver_variable, sense=lsf.objective_sense_minimize()
        )
        for lb, ub in zip(x_domain, x_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(x, lb, ub)
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
                    min_y_boundary_values[lb], new_y_grid_value
                )
                min_y_boundary_values[ub] = new_y_grid_value
                min_y_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_y_grid_value)
                )
        self.external_solver.opt_model.set_variable_lb(x.solver_variable, x.lb)
        self.external_solver.opt_model.set_variable_ub(x.solver_variable, x.ub)

        self.external_solver.opt_model.set_objective(
            x.solver_variable, sense=lsf.objective_sense_maximize()
        )
        for lb, ub in zip(y_domain, y_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(y, lb, ub)
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
                    max_x_boundary_values[lb], new_x_grid_value
                )
                max_x_boundary_values[ub] = new_x_grid_value
                max_x_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_x_grid_value)
                )

        self.external_solver.opt_model.set_objective(
            x.solver_variable, sense=lsf.objective_sense_minimize()
        )
        for lb, ub in zip(y_domain, y_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(y, lb, ub)
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
                    min_x_boundary_values[lb], new_x_grid_value
                )
                min_x_boundary_values[ub] = new_x_grid_value
                min_x_conditions.append(
                    (lambda u, f_lb=lb, f_ub=ub: f_lb <= u <= f_ub, new_x_grid_value)
                )
        self.external_solver.opt_model.set_variable_lb(y.solver_variable, y.lb)
        self.external_solver.opt_model.set_variable_ub(y.solver_variable, y.ub)

        x_grid = sorted(x_domain + x_grid_addition)
        y_grid = sorted(y_domain + y_grid_addition)
        feasible_grid_points = []
        for x_grid_point in x_grid:
            for y_grid_point in y_grid:
                y_ub = max_y(x_grid_point)
                if y_grid_point > y_ub:
                    continue
                y_lb = min_y(x_grid_point)
                if y_grid_point < y_lb:
                    continue
                x_ub = max_x(y_grid_point)
                if x_grid_point > x_ub:
                    continue
                x_lb = min_x(y_grid_point)
                if x_grid_point < x_lb:
                    continue
                feasible_grid_points.append((x_grid_point, y_grid_point))
        start_vertex = feasible_grid_points[0]
        current_vertex = start_vertex
        current_direction = -1
        traversed = []
        while True:
            for i in range(4):
                next_direction = (current_direction - 1 + i) % 4
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
            if (current_vertex, next_direction) in traversed:
                break
            if current_direction != next_direction:
                traversed.append((current_vertex, next_direction))
            current_vertex = next_vertex
            current_direction = next_direction

        post_processed_vertices = [point for point, direction in traversed]
        if self.settings.feature_stair_locatelli == 1:
            post_processed_vertices = self._get_convex_hull(post_processed_vertices)
        self.bilinear_projected_domains.append(
            (bilinear_expression, post_processed_vertices)
        )

    @staticmethod
    def _find_closest_entry_in_list(data, target):
        closest_val = min(data, key=lambda x: abs(x - target))
        if abs(closest_val - target) <= s.StaticSettings.feasibility_tolerance:
            return closest_val
        return None

    def _post_process_vertices(
        self, vertices: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Post-process vertices to remove collinear points."""
        vertices = self._filter_equal_vertices(vertices)
        return self._filter_collinear_vertices(vertices)

    @staticmethod
    def _filter_equal_vertices(
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        vertices_filtered_indices = []
        for i, (v1, v2) in enumerate(zip(vertices, vertices[1:] + [vertices[0]])):
            if np.isclose(v1[0], v2[0], atol=1e-3) and np.isclose(
                v1[1], v2[1], atol=1e-3
            ):
                continue
            vertices_filtered_indices.append(i)
        return [vertices[i] for i in vertices_filtered_indices]

    @staticmethod
    def _filter_collinear_vertices(
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        triple_vertices = vertices + vertices + vertices
        non_vertex_point_indices = []

        for i in range(len(vertices)):
            v1, v2, v3 = triple_vertices[i : i + 3]
            vec1 = np.array([v2[0] - v1[0], v2[1] - v1[1]])
            vec2 = np.array([v3[0] - v2[0], v3[1] - v2[1]])

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if (
                norm1 < s.StaticSettings.feasibility_tolerance
                or norm2 < s.StaticSettings.feasibility_tolerance
            ):
                non_vertex_point_indices.append((i + 1) % len(vertices))
                continue
            normalized_vec1 = vec1 / norm1
            normalized_vec2 = vec2 / norm2
            similarity = np.dot(normalized_vec1, normalized_vec2)
            if np.isclose(similarity, 1.0, atol=s.StaticSettings.feasibility_tolerance):
                non_vertex_point_indices.append((i + 1) % len(vertices))
        return [v for i, v in enumerate(vertices) if i not in non_vertex_point_indices]

    @staticmethod
    def _get_convex_hull(vertices):
        """
        Calculates the convex hull of a set of 2D vertices using the Monotone Chain algorithm.

        Args:
            vertices: A list of (x, y) tuples.

        Returns:
            A list of (x, y) tuples representing the vertices of the convex hull
            in counter-clockwise order.
        """

        # Helper function to calculate the cross product (for turn direction)
        def cross_product(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        # 1. Sort vertices lexicographically (by x, then y)
        vertices.sort()

        # 2. Build the lower hull
        lower_hull = []
        for v in vertices:
            # While the last two points and the current point make a clockwise turn...
            while (
                len(lower_hull) >= 2
                and cross_product(lower_hull[-2], lower_hull[-1], v) <= 0
            ):
                lower_hull.pop()  # ...remove the last point.
            lower_hull.append(v)

        # 3. Build the upper hull
        upper_hull = []
        # Iterate in reverse order
        for v in reversed(vertices):
            # While the last two points and the current point make a clockwise turn...
            while (
                len(upper_hull) >= 2
                and cross_product(upper_hull[-2], upper_hull[-1], v) <= 0
            ):
                upper_hull.pop()  # ...remove the last point.
            upper_hull.append(v)

        # 4. Combine hulls and remove duplicates
        # The first and last points of the sorted list are on both hulls.
        # We remove them from the upper hull before combining.
        return lower_hull + upper_hull[1:-1]

    def _get_solution_value_for_interval(
        self, variable: var.Variable, lb: float, ub: float
    ) -> tuple[bool, float]:
        self.external_solver.opt_model.set_variable_lb(variable.solver_variable, lb)
        self.external_solver.opt_model.set_variable_ub(variable.solver_variable, ub)
        self.external_solver.opt_model.optimize()
        if self.external_solver.opt_model.is_infeasible():
            return False, 0.0
        return True, self.external_solver.opt_model.get_objective_value()

    @staticmethod
    def _generate_edge_checkpoints(
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Generates checkpoints along the edges of the polygon defined by vertices.

        Args:
            vertices: A list of (x, y) tuples representing the polygon's vertices.

        Returns:
            A list of (x, y) tuples representing points on the polygon's boundary.
        """
        checkpoints = []
        # Create a closed loop by adding the first vertex to the end
        looped_vertices = vertices + [vertices[0]]
        for (vfx, vfy), (vtx, vty) in zip(looped_vertices, looped_vertices[1:]):
            for i in range(20):
                # Interpolate points along the segment
                point_x = vfx + (vtx - vfx) * (i / 20)
                point_y = vfy + (vty - vfy) * (i / 20)
                checkpoints.append((point_x, point_y))
        return checkpoints

    def _process_vertex_combination(
        self,
        bilinear_expression: ble.BilinearExpression,
        vertex_triple: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        checkpoints: list[tuple[float, float]],
        cut_index: int,
    ) -> bool:
        """
        Processes a single combination of three vertices to potentially add a cut.

        Args:
            bilinear_expression: The bilinear expression object.
            v1, v2, v3: tuples representing the three vertices.
            checkpoints: A list of points to check for feasibility.
            cut_index: The current index for naming the cut.

        Returns:
            True if a cut was added, False otherwise.
        """
        v1, v2, v3 = vertex_triple
        if self._check_if_vertices_are_collinear(v1, v2, v3):
            return False

        check, cut_coefficients = self._calculate_hyperplane_for_vertex_triple(
            v1, v2, v3
        )
        if not check:
            return False
        con_type = self._check_if_hyperplane_is_infeasible(
            cut_coefficients, checkpoints
        )

        if con_type == lsf.empty_string():
            return False

        self._create_and_add_locatelli_cut(
            bilinear_expression, cut_index, cut_coefficients, con_type
        )
        return True

    def _create_and_add_locatelli_cut(
        self,
        bilinear_expression: ble.BilinearExpression,
        cut_index: int,
        cut_coefficients: np.ndarray,
        con_type: str,
    ):
        """
        Creates and adds a single Locatelli cut constraint to the model.

        Args:
            bilinear_expression: The bilinear expression object.
            cut_index: The index for naming the new cut.
            cut_coefficients: The coefficients for the new cut.
            con_type: The type of the constraint (e.g., '<=', '>=', '==').
        """
        x_coeff, y_coeff, const = cut_coefficients
        locatelli_cut = con.LinearConstraint(
            f"locatelli_{bilinear_expression.name}_{cut_index}",
            con_type=con_type,
            rhs=const,
            variables=[
                (-x_coeff, bilinear_expression.variables[0]),
                (-y_coeff, bilinear_expression.variables[1]),
                (1.0, bilinear_expression.representative_variable),
            ],
        )
        self.model_data.add_constraint(locatelli_cut)

    def _add_locatelli_cuts_from_vertices(self):
        """
        The main method to generate and add all Locatelli cuts from projected domains.
        This method now orchestrates calls to helper functions.
        """
        total_nr_of_cuts = 0
        for bilinear_expression, vertices in self.bilinear_projected_domains:
            if len(vertices) < 3:
                continue

            checkpoints = self._generate_edge_checkpoints(vertices)
            cut_index = 0
            for v1, v2, v3 in itertools.combinations(vertices, 3):
                cut_added = self._process_vertex_combination(
                    bilinear_expression, (v1, v2, v3), checkpoints, cut_index
                )
                if cut_added:
                    cut_index += 1
            total_nr_of_cuts += cut_index

        logger.info(lsf.info_total_stair_locatelli_cuts_added(total_nr_of_cuts))

    @staticmethod
    def _calculate_hyperplane_for_vertex_triple(
        v1: tuple[float, float], v2: tuple[float, float], v3: tuple[float, float]
    ) -> tuple[bool, np.ndarray]:
        """Calculate hyperplane coefficients for three points in 2D."""
        matrix = np.array([[v1[0], v1[1], 1], [v2[0], v2[1], 1], [v3[0], v3[1], 1]])
        b = np.array([v1[0] * v1[1], v2[0] * v2[1], v3[0] * v3[1]])
        try:
            return True, np.linalg.solve(matrix, b)
        except np.linalg.LinAlgError:
            return False, np.array([0.0, 0.0, 0.0])

    @staticmethod
    def _check_if_hyperplane_is_infeasible(
        cut_coefficients: np.ndarray,
        checkpoints: list[tuple[float, float]],
    ) -> str:
        """Check if hyperplane is infeasible for all vertices."""
        x_coeff, y_coeff, const = cut_coefficients
        if np.all(
            [
                x_coeff * x + y_coeff * y + const
                <= x * y + s.StaticSettings.feasibility_tolerance
                for x, y in checkpoints
            ]
        ):
            return lsf.constraint_geq()
        if np.all(
            [
                x_coeff * x + y_coeff * y + const
                >= x * y - s.StaticSettings.feasibility_tolerance
                for x, y in checkpoints
            ]
        ):
            return lsf.constraint_leq()
        return lsf.empty_string()

    @staticmethod
    def _check_if_vertices_are_collinear(
        v1: tuple[float, float], v2: tuple[float, float], v3: tuple[float, float]
    ) -> bool:
        """Check if three points are collinear."""
        vec1 = np.array([v2[0] - v1[0], v2[1] - v1[1]])
        vec2 = np.array([v3[0] - v2[0], v3[1] - v2[1]])
        # Check if vectors are collinear using cross product
        if abs(np.cross(vec1, vec2)) < s.StaticSettings.feasibility_tolerance:
            return True
        return False

    def plot_polygons(self):
        """
        Plots all polygons defined by post_processed_vertices and saves the
        plot to a file.
        """
        plot_dir = os.path.join(self.settings.export_path, "stair_locatelli_plots")
        os.makedirs(plot_dir, exist_ok=True)

        for bilinear_expression, vertices in self.bilinear_projected_domains:
            if not vertices:
                continue
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()
            # Unzip vertices into x and y coordinates
            x_coords, y_coords = zip(*vertices)
            # Add the first vertex to the end to close the polygon
            x_coords_closed = x_coords + (x_coords[0],)
            y_coords_closed = y_coords + (y_coords[0],)
            ax.plot(
                x_coords_closed,
                y_coords_closed,
                marker="o",
                linestyle="-",
                label=f"Polygon for {bilinear_expression.name}",
            )
            ax.set_xlabel(bilinear_expression.variables[0].name)
            ax.set_ylabel(bilinear_expression.variables[1].name)
            ax.set_title("Projected Feasible Domains (Polygons)")
            ax.legend()
            ax.grid(True)

            file_path = os.path.join(
                plot_dir, f"polygons_{bilinear_expression.name}.png"
            )
            plt.savefig(file_path)
            plt.close(fig)

    def plot_locatelli_cuts(self):  # pylint: disable=too-many-locals
        """
        Plots the z = x*y surface and the feasible hyperplanes (Locatelli cuts)
        for each bilinear term and saves them to files.
        """
        plot_dir = os.path.join(self.settings.export_path, "stair_locatelli_plots")
        os.makedirs(plot_dir, exist_ok=True)
        view_angles = [
            (30, -60),  # Default-ish view
            (30, 30),  # Another common view
            (60, -120),  # Higher elevation, different azimuth
        ]

        for bilinear_expression, vertices in self.bilinear_projected_domains:
            if len(vertices) < 3:
                continue

            x_var, y_var = bilinear_expression.variables

            # Create a meshgrid for plotting
            x_range = np.linspace(x_var.lb, x_var.ub, 30)
            y_range = np.linspace(y_var.lb, y_var.ub, 30)
            x, y = np.meshgrid(x_range, y_range)
            z = x * y

            for i, (elev, azim) in enumerate(view_angles):
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection="3d")

                # Plot the original z = x*y surface
                ax.plot_surface(x, y, z, alpha=0.5, color="r", label="z = x*y")

                checkpoints = self._generate_edge_checkpoints(vertices)

                # Find and plot feasible hyperplanes
                for v1, v2, v3 in itertools.combinations(vertices, 3):

                    # Re-use logic from the cut generation to find valid planes
                    if self._check_if_vertices_are_collinear(v1, v2, v3):
                        continue

                    check, cut_coefficients = (
                        self._calculate_hyperplane_for_vertex_triple(v1, v2, v3)
                    )
                    if not check:
                        continue
                    con_type = self._check_if_hyperplane_is_infeasible(
                        cut_coefficients, checkpoints
                    )
                    if con_type != lsf.empty_string():
                        color = "green" if con_type == lsf.constraint_leq() else "blue"
                        x_coeff, y_coeff, const = cut_coefficients
                        z_plane = x_coeff * x + y_coeff * y + const
                        # Plot the hyperplane
                        ax.plot_surface(
                            x,
                            y,
                            z_plane,
                            alpha=0.6,
                            rstride=100,
                            cstride=100,
                            color=color,
                        )

                ax.set_xlabel(f"x ({x_var.name})")
                ax.set_ylabel(f"y ({y_var.name})")
                ax.set_zlabel(f"z ({bilinear_expression.representative_variable.name})")
                ax.set_title(f"Feasible Locatelli Cuts for {bilinear_expression.name}")
                ax.view_init(elev=elev, azim=azim)
                filename = f"locatelli_cuts_{bilinear_expression.name}_{i}.png"
                file_path = os.path.join(plot_dir, filename)
                plt.savefig(file_path)
                plt.close(fig)
