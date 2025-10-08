# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import numpy as np

from alpaca.model_data import model_data as md, variable as var, constraint as con
from alpaca.expressions import bilinear_expression as ble
from alpaca.external_solvers import mip_model as mm
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
        )
        self.external_solver.opt_model.hide_output()
        self.bilinear_projected_domains: list[
            tuple[ble.BilinearExpression, list[tuple[float, float]]]
        ] = []
        self._add_stair_locatelli_constraints()

    def _add_stair_locatelli_constraints(self):
        """Add stair locatelli cuts."""
        self._get_vertices_for_bilinear_projections()
        self._add_locatelli_cuts_from_vertices()

    def _get_vertices_for_bilinear_projections(self):
        for expression in self.model_data.expressions.bilinear_expressions.values():
            self._get_projected_domain_vertices_for_bilinear_expression(expression)

    def _get_projected_domain_vertices_for_bilinear_expression(
        self, bilinear_expression: ble.BilinearExpression
    ):
        vertices = []
        x, y = tuple(bilinear_expression.variables)
        x_domain = np.linspace(
            x.lb, x.ub, self.settings.feature_stair_locatelli_grid_size
        ).tolist()
        # y_domain = np.linspace(
        #     y.lb, y.ub, self.settings.feature_stair_locatelli_grid_size
        # ).tolist()
        self.external_solver.opt_model.set_objective(
            y.solver_variable, sense=lsf.objective_sense_maximize()
        )
        for lb, ub in zip(x_domain, x_domain[1:]):
            feasible, solution_value = self._get_solution_value_for_interval(x, lb, ub)
            if feasible:
                vertices.extend([(lb, solution_value), (ub, solution_value)])
        self.external_solver.opt_model.set_variable_lb(x.solver_variable, x.lb)
        self.external_solver.opt_model.set_variable_ub(x.solver_variable, x.ub)

        # self.external_solver.opt_model.set_objective(x.solver_variable,
        #                                              sense=lsf.objective_sense_maximize())
        # for lb, ub in reversed(list(zip(y_domain, y_domain[1:]))):
        #     feasible, solution_value = self._get_solution_value_for_interval(y, lb, ub)
        #     if feasible:
        #         vertices.extend([(solution_value, ub), (solution_value, lb)])
        # self.external_solver.opt_model.set_variable_lb(y.solver_variable, y.lb)
        # self.external_solver.opt_model.set_variable_ub(y.solver_variable, y.ub)

        self.external_solver.opt_model.set_objective(
            y.solver_variable, sense=lsf.objective_sense_minimize()
        )
        for lb, ub in reversed(list(zip(x_domain, x_domain[1:]))):
            feasible, solution_value = self._get_solution_value_for_interval(x, lb, ub)
            if feasible:
                vertices.extend([(ub, solution_value), (lb, solution_value)])
        self.external_solver.opt_model.set_variable_lb(x.solver_variable, x.lb)
        self.external_solver.opt_model.set_variable_ub(x.solver_variable, x.ub)

        # self.external_solver.opt_model.set_objective(x.solver_variable,
        #                                              sense=lsf.objective_sense_minimize())
        # for lb, ub in zip(y_domain, y_domain[1:]):
        #     feasible, solution_value = self._get_solution_value_for_interval(y, lb, ub)
        #     if feasible:
        #         vertices.extend([(solution_value, lb), (solution_value, ub)])
        # self.external_solver.opt_model.set_variable_lb(y.solver_variable, y.lb)
        # self.external_solver.opt_model.set_variable_ub(y.solver_variable, y.ub)
        self.bilinear_projected_domains.append(
            (bilinear_expression, self._post_process_vertices(vertices))
        )

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

            normalized_vec1 = vec1 / np.linalg.norm(vec1)
            normalized_vec2 = vec2 / np.linalg.norm(vec2)
            similarity = np.dot(normalized_vec1, normalized_vec2)
            if np.isclose(similarity, 1.0, atol=1e-3):
                non_vertex_point_indices.append(i + 1 % len(vertices))
        return [v for i, v in enumerate(vertices) if i not in non_vertex_point_indices]

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

        cut_coefficients = self._calculate_hyperplane_for_vertex_triple(v1, v2, v3)
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
    ):
        """Calculate hyperplane coefficients for three points in 2D."""
        matrix = np.array([[v1[0], v1[1], 1], [v2[0], v2[1], 1], [v3[0], v3[1], 1]])
        b = np.array([v1[0] * v1[1], v2[0] * v2[1], v3[0] * v3[1]])
        return np.linalg.solve(matrix, b)

    @staticmethod
    def _check_if_hyperplane_is_infeasible(
        cut_coefficients: np.ndarray,
        checkpoints: list[tuple[float, float]],
    ) -> str:
        """Check if hyperplane is infeasible for all vertices."""
        x_coeff, y_coeff, const = cut_coefficients
        if np.all([x_coeff * x + y_coeff * y + const <= x * y for x, y in checkpoints]):
            return lsf.constraint_geq()
        if np.all([x_coeff * x + y_coeff * y + const >= x * y for x, y in checkpoints]):
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
        if abs(np.cross(vec1, vec2)) < 0.0001:
            return True
        return False
