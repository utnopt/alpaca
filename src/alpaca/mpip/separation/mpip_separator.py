# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from typing import Iterator, Any
import itertools
import numpy as np

import alpaca.settings as s
import alpaca.mpip.mpip as mp
import alpaca.model_data.variable as var
import alpaca.external_solvers.solver_wrapper as sw
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class SeparatedPoint:
    """Non-integer point to be separated with perturbation capabilities."""

    def __init__(self) -> None:
        np.random.seed(0)
        self.implying_values: dict[str, np.ndarray] = {}
        self.implied_values: np.ndarray = np.array([])
        self.implying_values_randomized: dict[str, np.ndarray] = {}
        self.implied_values_randomized: np.ndarray = np.array([])

    def is_integer(self) -> bool:
        """Check if all variables are integer within tolerance."""
        implying_checks = (
            abs(value - round(value)) < s.StaticSettings.feasibility_tolerance
            for values in self.implying_values.values()
            for value in values
        )
        implied_checks = (
            abs(value - round(value)) < s.StaticSettings.feasibility_tolerance
            for value in self.implied_values
        )
        return all(itertools.chain(implying_checks, implied_checks))

    def perturb(self) -> None:
        """Apply random perturbation to variable values."""
        # Perturb implying values
        for key, values in self.implying_values.items():
            self.implying_values_randomized[key] = np.add(
                np.random.uniform(
                    s.StaticSettings.feasibility_tolerance / 100,
                    s.StaticSettings.feasibility_tolerance,
                    len(values),
                ),
                values,
            )

        # Perturb implied values
        self.implied_values_randomized = np.add(
            np.random.uniform(
                s.StaticSettings.feasibility_tolerance / 100,
                s.StaticSettings.feasibility_tolerance,
                len(self.implied_values),
            ),
            self.implied_values,
        )


class MPIPSeparator:  # pylint: disable=too-many-instance-attributes
    """Multipartite Implication Polytope separation handler."""

    def __init__(self, mpip: mp.MPIP, opt_model: sw.SolverWrapper) -> None:
        self.separation_model = sw.SolverWrapper(opt_model.mip_solver)
        self.mpip = mpip
        self._setup_separation_model()
        self.separation_handler: None | sw.ScipSeparation = None
        self.relation_matrix_size = len(mpip.implied_variable.breakpoints) - 1
        self.sep_implying_variables: dict[str, list[Any]] = {}
        self.sep_implied_variables: list[Any] = []
        self.point_to_be_separated = SeparatedPoint()
        self.opt_model = opt_model
        self.nr_of_cuts = 0
        self.nr_of_not_cuts = 0
        self.used = 1
        self.unused = 0
        self.usefulness = 1
        self.separated_cuts = []

    def _setup_separation_model(self) -> None:
        """Configure separation model settings."""
        self.separation_model.hide_output()
        self.separation_model.set_seed(42)
        self.separation_model.enable_reoptimization()

    def reset_useless_counter(self) -> None:
        """Reset the useless cut counter."""
        self.used = 1
        self.unused = 0
        self.usefulness = 1

    def add_mc_cormick_constraints(self) -> None:
        """Add McCormick envelope constraints to optimization model."""
        for (
            implying_indices,
            implying_variables,
        ) in self._generate_implying_combinations():
            implied_variables = tuple(
                self.mpip.implied_variable.pwl.pwl_variables_binary[implied_index]
                for implied_index in self.mpip.relation[implying_indices]
            )
            self._add_mccormick_constraint(
                implying_indices, implying_variables, implied_variables
            )

    def add_corner_constraints(self) -> None:
        """Add corner constraints to optimization model."""
        for z_index in range(len(self.mpip.implied_variable.breakpoints)):
            self._add_corner_constraint_bottom_right_one_entry(z_index)
            self._add_corner_constraint_top_left_one_entry(z_index)

    def _add_corner_constraint_bottom_right_one_entry(self, z_index: int) -> None:
        x_var_name = list(self.mpip.implying_variables.keys())[0]
        y_var_name = list(self.mpip.implying_variables.keys())[1]
        max_x_index = len(self.sep_implying_variables[x_var_name]) - 1
        max_y_index = len(self.sep_implying_variables[y_var_name]) - 1

        current_low_y = 0
        blocks = []
        for i in range(max_x_index, -1, -1):
            for j in range(max_y_index, current_low_y - 1, -1):
                if z_index in self.mpip.relation.get((i, j), (z_index,)):
                    new_low_y = j + 1
                    if i < max_x_index:
                        blocks.append(
                            (
                                tuple(range(i + 1, max_x_index + 1)),  # X: i+1 to max_x
                                tuple(
                                    range(current_low_y, max_y_index + 1)
                                ),  # Y: current_low_y to max_y
                            )
                        )

                    current_low_y = new_low_y
                    break

        if current_low_y <= max_y_index:
            blocks.append(
                (
                    tuple(range(0, max_x_index + 1)),
                    tuple(range(current_low_y, max_y_index + 1)),
                )
            )

        for block in blocks:
            self._add_constraint_from_block(block)

    def _add_corner_constraint_top_left_one_entry(self, z_index: int) -> None:
        x_var_name = list(self.mpip.implying_variables.keys())[0]
        y_var_name = list(self.mpip.implying_variables.keys())[1]
        max_x_index = len(self.sep_implying_variables[x_var_name]) - 1
        max_y_index = len(self.sep_implying_variables[y_var_name]) - 1

        current_h = max_y_index + 1
        blocks = []

        for i in range(max_x_index + 1):
            for j in range(current_h):
                if z_index in self.mpip.relation.get((i, j), (z_index,)):
                    new_h = j
                    if i > 0:
                        blocks.append(
                            (
                                tuple(range(0, i)),  # X: 0 to i-1
                                tuple(range(0, current_h)),  # Y: 0 to current_h-1
                            )
                        )

                    current_h = new_h
                    break

        if current_h > 0:
            blocks.append(
                (
                    tuple(range(0, max_x_index + 1)),
                    tuple(range(0, current_h)),
                )
            )

        for block in blocks:
            self._add_constraint_from_block(block)

    def _add_constraint_from_block(
        self, block: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> None:
        """Add constraint for a specific block."""
        x_indices, y_indices = block
        if x_indices == () or y_indices == ():
            return
        constraint = self.opt_model.add_constraint(
            sum(
                self.mpip.implying_variables[
                    list(self.mpip.implying_variables.keys())[0]
                ]
                .pwl.pwl_variables_binary[x_index]
                .solver_variable
                for x_index in x_indices
            )
            + sum(
                self.mpip.implying_variables[
                    list(self.mpip.implying_variables.keys())[1]
                ]
                .pwl.pwl_variables_binary[y_index]
                .solver_variable
                for y_index in y_indices
            )
            - sum(
                self.mpip.implied_variable.pwl.pwl_variables_binary[
                    implied_index
                ].solver_variable
                for implied_index in set(
                    sum(
                        {
                            self.mpip.relation.get((x_index, y_index), ())
                            for x_index in x_indices
                            for y_index in y_indices
                        },
                        (),
                    )
                )
            )
            <= len(self.mpip.implying_variables) - 1,
            name=f"corner_{block}_{self.mpip.mpip_id}".replace(" ", ""),
        )
        constraint.lazy = 3

    def _generate_implying_combinations(
        self,
    ) -> Iterator[tuple[tuple[int, ...], tuple[var.Variable, ...]]]:
        """Generate all combinations of implying variables."""
        for combination in itertools.product(
            *[
                enumerate(variable.pwl.pwl_variables_binary)
                for variable in self.mpip.implying_variables.values()
            ]
        ):
            indices, variables = zip(*combination)
            yield indices, variables

    def _add_mccormick_constraint(
        self,
        implying_indices: tuple[int, ...],
        implying_variables: tuple[var.Variable, ...],
        implied_variables: tuple[var.Variable, ...],
    ) -> None:
        """Create individual McCormick constraint."""
        self.opt_model.add_constraint(
            -sum(variable.solver_variable for variable in implying_variables)
            + sum(variable.solver_variable for variable in implied_variables)
            >= -len(self.mpip.implying_variables) + 1,
            name=f"corm_{self.mpip.mpip_id}_{implying_indices}".replace(" ", ""),
        )

    def add_stripe_constraints(self) -> None:
        """Add stripe constraints to optimization model."""
        slice_dict = {
            implying_vars_index: {}
            for implying_vars_index in range(len(self.mpip.implying_variables))
        }
        for implying_vars_index, implying_var in enumerate(
            self.mpip.implying_variables.values()
        ):
            for implying_index in range(len(implying_var.pwl.pwl_variables_binary)):
                slice_dict[implying_vars_index][(implying_index,)] = set(
                    sum(
                        {
                            implied_relation_is
                            for implying_is, implied_relation_is in self.mpip.relation.items()
                            if implying_is[implying_vars_index] == implying_index
                        },
                        (),
                    )
                )
        improved_slice_dict = self._improve_slice_dict(slice_dict)
        added_cuts = 0
        for implying_vars_index, slice_dict_values in improved_slice_dict.items():
            for implying_indices, implied_indices in slice_dict_values.items():
                added_cuts += 1
                self.opt_model.add_constraint(
                    sum(
                        list(self.mpip.implying_variables.values())[implying_vars_index]
                        .pwl.pwl_variables_binary[implying_index]
                        .solver_variable
                        for implying_index in implying_indices
                    )
                    + sum(
                        sum(
                            variable.solver_variable
                            for variable in other_implying_var.pwl.pwl_variables_binary
                        )
                        for other_vars_index, other_implying_var in enumerate(
                            self.mpip.implying_variables.values()
                        )
                        if other_vars_index != implying_vars_index
                    )
                    - sum(
                        (
                            self.mpip.implied_variable.pwl.pwl_variables_binary[
                                implied_index
                            ].solver_variable
                            for implied_index in implied_indices
                        )
                    )
                    <= len(self.mpip.implying_variables) - 1,
                    name=f"stripe_{implying_vars_index}_"
                    f"{'_'.join([str(implying_index) for implying_index in implying_indices])}_"
                    f"{self.mpip.mpip_id}",
                )

    def _improve_slice_dict(
        self, slice_dict: dict[int, dict[tuple[int, ...], set[int]]]
    ) -> dict[int, dict[tuple[int, ...], set[int]]]:
        for implying_vars_index, slice_dict_values in slice_dict.items():
            slice_dict[implying_vars_index] = self._improve_slice_dict_for_specific_row(
                slice_dict_values
            )
        return slice_dict

    def _improve_slice_dict_for_specific_row(
        self, slice_dict_values: dict[tuple[int, ...], set[int]]
    ) -> dict[tuple[int, ...], set[int]]:
        improved_slice_dict_values = {}
        slice_dict_values = {
            implying_indices: implied_indices
            for implying_indices, implied_indices in slice_dict_values.items()
            if len(implied_indices) < len(self.sep_implied_variables)
        }
        for implying_indices, implied_indices in slice_dict_values.items():
            new_implying_combination = tuple(
                sorted(
                    set().union(
                        *[
                            set(other_is)
                            for other_is, other_implied_is in slice_dict_values.items()
                            if other_implied_is.issubset(implied_indices)
                        ]
                    )
                )
            )
            improved_slice_dict_values[new_implying_combination] = implied_indices

        return improved_slice_dict_values

    def build_separation_model(self) -> None:
        """Construct separation model with variables and constraints."""
        self._add_separation_variables()
        self._add_separation_constraints()

    def _add_separation_variables(self) -> None:
        """Create variables for separation model."""
        for implying_index, implying_variable in self.mpip.implying_variables.items():
            sep_implying_variables = []
            for i in range(len(implying_variable.breakpoints) - 1):
                variable = self.separation_model.add_variable(
                    f"sep_implying_{implying_index}_{i}", ub=1, obj=1
                )
                sep_implying_variables.append(variable)
            self.sep_implying_variables[implying_index] = sep_implying_variables

        self.sep_implied_variables = [
            self.separation_model.add_variable(f"sep_implied_{i}", ub=1, obj=-1)
            for i in range(len(self.mpip.implied_variable.breakpoints) - 1)
        ]

    def _add_separation_constraints(self) -> None:
        """Add constraints to separation model."""
        for (
            implying_indices,
            implying_variables,
        ) in self._generate_separation_combinations():
            for implied_index in self.mpip.relation[implying_indices]:
                implied_variable = self.sep_implied_variables[implied_index]
                self.separation_model.add_constraint(
                    sum(implying_variables) - implied_variable
                    <= len(self.mpip.implying_variables) - 1
                )

    def _generate_separation_combinations(
        self,
    ) -> Iterator[tuple[tuple[int, ...], tuple[Any, ...]]]:
        """Generate combinations for separation constraints."""
        for combination in itertools.product(
            *[enumerate(d) for d in self.sep_implying_variables.values()]
        ):
            indices, variables = zip(*combination)
            yield indices, variables

    def separate_point(self) -> bool:
        """Perform separation of current point."""
        self._update_separation_objective()
        self.separation_model.optimize()
        return self._generate_cut()

    def _update_separation_objective(self) -> None:
        """Update separation model objective with current point values."""
        new_objective = self.separation_model.get_new_expression()
        for (
            implying_index,
            implying_values,
        ) in self.point_to_be_separated.implying_values_randomized.items():
            for idx, val in enumerate(implying_values):
                new_objective += val * self.sep_implying_variables[implying_index][idx]

        for idx, val in enumerate(self.point_to_be_separated.implied_values_randomized):
            new_objective -= val * self.sep_implied_variables[idx]
        self.separation_model.update_objective(new_objective, sense="maximize")

    def _generate_cut(self) -> bool:
        """Generate cut based on separation solution."""
        if self.mpip.pwl_method == lsf.pwl_method_multiple_choice():
            return self._generate_cut_multiple_choice()
        return self._generate_cut_delta()

    def _generate_cut_multiple_choice(self) -> bool:
        """Generate cut based on separation solution multiple choice method."""
        cut_to_separate = self.opt_model.create_cut(
            self.separation_handler,
            f"{self.mpip.mpip_id}_x_{self.nr_of_cuts}",
            lhs=None,
            rhs=len(self.sep_implying_variables) - 1,
            local=False,
        )
        violation = -len(self.sep_implying_variables) + 1
        for implying_index, implying_variable in self.mpip.implying_variables.items():
            for idx, variable in enumerate(implying_variable.pwl.pwl_variables_binary):
                solution_value = self.separation_model.get_val(
                    self.sep_implying_variables[implying_index][idx]
                )
                self.opt_model.add_var_to_cut(
                    cut_to_separate,
                    variable.solver_variable,
                    solution_value,
                )
                violation += (
                    solution_value
                    * self.point_to_be_separated.implying_values[implying_index][idx]
                )
        for idx, variable in enumerate(
            self.mpip.implied_variable.pwl.pwl_variables_binary
        ):
            solution_value = self.separation_model.get_val(
                self.sep_implied_variables[idx]
            )
            self.opt_model.add_var_to_cut(
                cut_to_separate,
                variable.solver_variable,
                -solution_value,
            )
            violation -= solution_value * self.point_to_be_separated.implied_values[idx]
        if violation > s.StaticSettings.min_cut_violation:
            self.nr_of_cuts += 1
            self.used += 1
            self.usefulness = self.used / (self.used + self.unused)
            self.opt_model.add_cut(cut_to_separate)
            self.separated_cuts.append(cut_to_separate)
            return True
        self.nr_of_not_cuts += 1
        self.unused += 1
        self.usefulness = self.used / (self.used + self.unused)
        return False

    def _generate_cut_delta(self) -> bool:
        """Generate cut based on separation solution delta method."""
        cut_rhs = len(self.sep_implying_variables) - 1
        cut_variables = []
        violation = -len(self.sep_implying_variables) + 1
        for implying_index, implying_variable in self.mpip.implying_variables.items():
            for idx, variable in enumerate(implying_variable.pwl.pwl_variables_binary):
                solution_value = self.separation_model.get_val(
                    self.sep_implying_variables[implying_index][idx]
                )
                if idx == 0:
                    cut_rhs -= solution_value
                    cut_variables.append((-solution_value, variable.solver_variable))
                else:
                    cut_variables.append(
                        (
                            solution_value,
                            implying_variable.pwl.pwl_variables_binary[
                                idx - 1
                            ].solver_variable,
                        )
                    )
                    cut_variables.append((-solution_value, variable.solver_variable))
                violation += (
                    solution_value
                    * self.point_to_be_separated.implying_values[implying_index][idx]
                )
        for idx, variable in enumerate(
            self.mpip.implied_variable.pwl.pwl_variables_binary
        ):
            solution_value = self.separation_model.get_val(
                self.sep_implied_variables[idx]
            )
            if idx == 0:
                cut_rhs += solution_value
                cut_variables.append((solution_value, variable.solver_variable))
            else:
                cut_variables.append(
                    (
                        -solution_value,
                        self.mpip.implied_variable.pwl.pwl_variables_binary[
                            idx - 1
                        ].solver_variable,
                    )
                )
                cut_variables.append((solution_value, variable.solver_variable))
            violation -= solution_value * self.point_to_be_separated.implied_values[idx]
        if violation > s.StaticSettings.min_cut_violation:
            cut_to_separate = self.opt_model.create_cut(
                self.separation_handler,
                f"{self.mpip.mpip_id}_x_{self.nr_of_cuts}",
                lhs=None,
                rhs=cut_rhs,
                local=False,
            )
            for coeff, cut_var in cut_variables:
                self.opt_model.add_var_to_cut(cut_to_separate, cut_var, coeff)
            self.nr_of_cuts += 1
            self.used += 1
            self.usefulness = self.used / (self.used + self.unused)
            self.opt_model.add_cut(cut_to_separate)
            self.separated_cuts.append(cut_to_separate)
            return True
        self.nr_of_not_cuts += 1
        self.unused += 1
        self.usefulness = self.used / (self.used + self.unused)
        return False

    def separate_solution(self) -> bool:
        """Extract solution point and initiate separation."""
        self._build_point_to_be_separated()
        if self.point_to_be_separated.is_integer():
            return False
        self.point_to_be_separated.perturb()
        return self.separate_point()

    def _build_point_to_be_separated(self) -> None:
        """Construct the point to be separated from current solution."""
        if self.mpip.pwl_method == lsf.pwl_method_multiple_choice():
            self._build_point_to_be_separated_multiple_choice()
        else:
            self._build_point_to_be_separated_delta()

    def _build_point_to_be_separated_multiple_choice(self) -> None:
        self.point_to_be_separated.implying_values = {
            implying_index: np.array(
                [
                    self.opt_model.get_val_callback(v.solver_variable)
                    for v in variable.pwl.pwl_variables_binary
                ]
            )
            for implying_index, variable in self.mpip.implying_variables.items()
        }
        self.point_to_be_separated.implied_values = np.array(
            [
                self.opt_model.get_val_callback(v.solver_variable)
                for v in self.mpip.implied_variable.pwl.pwl_variables_binary
            ]
        )

    def _build_point_to_be_separated_delta(self) -> None:
        """Construct the point to be separated from current solution using delta method."""
        self.point_to_be_separated.implying_values = {
            implying_index: np.array(
                [
                    1.0
                    - self.opt_model.get_val_callback(
                        variable.pwl.pwl_variables_binary[0].solver_variable
                    )
                ]
                + [
                    self.opt_model.get_val_callback(prev_v.solver_variable)
                    - self.opt_model.get_val_callback(post_v.solver_variable)
                    for prev_v, post_v in zip(
                        variable.pwl.pwl_variables_binary[:-1],
                        variable.pwl.pwl_variables_binary[1:],
                    )
                ]
            )
            for implying_index, variable in self.mpip.implying_variables.items()
        }
        self.point_to_be_separated.implied_values = np.array(
            [
                1.0
                - self.opt_model.get_val_callback(
                    self.mpip.implied_variable.pwl.pwl_variables_binary[
                        0
                    ].solver_variable
                )
            ]
            + [
                self.opt_model.get_val_callback(prev_v.solver_variable)
                - self.opt_model.get_val_callback(post_v.solver_variable)
                for prev_v, post_v in zip(
                    self.mpip.implied_variable.pwl.pwl_variables_binary[:-1],
                    self.mpip.implied_variable.pwl.pwl_variables_binary[1:],
                )
            ]
        )
