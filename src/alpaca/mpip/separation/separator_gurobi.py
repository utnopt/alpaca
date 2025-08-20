# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
from typing import Iterator
import copy
import dataclasses
import itertools
import random
import gurobipy as gp

import alpaca.settings as s
import alpaca.mpip.mpip as mp


@dataclasses.dataclass
class MPIPCut:
    """Separated callback cut for MPIP separation."""

    lhs: gp.LinExpr | None = None
    rhs: int = 0
    violation: float = 0.0


class SeparatedPoint:
    """Non-integer point to be separated with perturbation capabilities."""

    def __init__(self) -> None:
        random.seed(0)
        self.implying_values: dict[str, list[float]] = {}
        self.original_implying_values: dict[str, list[float]] = {}
        self.implied_values: list[float] = []
        self.original_implied_values: list[float] = []
        self.perturbation_range = (
            s.StaticSettings.feasibility_tolerance / 100,
            10 * s.StaticSettings.feasibility_tolerance,
        )

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
        self.original_implying_values = copy.deepcopy(self.implying_values)
        self.original_implied_values = copy.deepcopy(self.implied_values)

        # Perturb implying values
        for values in self.implying_values.values():
            for i, _ in enumerate(values):
                values[i] += random.uniform(*self.perturbation_range)

        # Perturb implied values
        for i, _ in enumerate(self.implied_values):
            self.implied_values[i] += random.uniform(*self.perturbation_range)


class Separator:  # pylint: disable=too-many-instance-attributes
    """Multipartite Implication Polytope separation handler."""

    def __init__(self, mpip: mp.MPIP, opt_model: gp.Model) -> None:
        self.separation_model = gp.Model()
        self._setup_separation_model()
        self.mpip = mpip
        self.relation_matrix_size = len(mpip.implied_variables)
        self.sep_implying_variables: dict[str, list[gp.Var]] = {}
        self.sep_implied_variables: list[gp.Var] = []
        self.point_to_be_separated = SeparatedPoint()
        self.opt_model = opt_model
        self.cut = MPIPCut()

    def _setup_separation_model(self) -> None:
        """Configure separation model settings."""
        self.separation_model.setParam("OutputFlag", 0)
        self.separation_model.setParam("Seed", 0)

    def add_mc_cormick_constraints(self) -> None:
        """Add McCormick envelope constraints to optimization model."""
        for (
            implying_indices,
            implying_variables,
        ) in self._generate_implying_combinations():
            implied_variables = tuple(
                self.mpip.implied_variables[implied_index]
                for implied_index in self.mpip.relation[implying_indices]
            )
            self._add_mccormick_constraint(
                implying_indices, implying_variables, implied_variables
            )

    def add_stripe_constraints(self) -> None:
        """Add stripe constraints to optimization model."""
        slice_dict = {
            implying_vars_index: {}
            for implying_vars_index in range(len(self.mpip.implying_variables))
        }
        for implying_vars_index, implying_vars in enumerate(
            self.mpip.implying_variables.values()
        ):
            for implying_index in range(len(implying_vars)):
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
        for implying_vars_index, slice_dict_values in improved_slice_dict.items():
            for implying_indices, implied_indices in slice_dict_values.items():
                self.opt_model.addConstr(
                    gp.quicksum(
                        list(self.mpip.implying_variables.values())[
                            implying_vars_index
                        ][implying_index]
                        for implying_index in implying_indices
                    )
                    + gp.quicksum(
                        gp.quicksum(other_implying_vars)
                        for other_vars_index, other_implying_vars in enumerate(
                            self.mpip.implying_variables.values()
                        )
                        if other_vars_index != implying_vars_index
                    )
                    - gp.quicksum(
                        (
                            self.mpip.implied_variables[implied_index]
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
                            for other_is, other_is in slice_dict_values.items()
                            if other_is.issubset(implied_indices)
                        ]
                    )
                )
            )
            improved_slice_dict_values[new_implying_combination] = implied_indices

        return improved_slice_dict_values

    def add_stair_constraints(self) -> None:
        """Add stair constraints to optimization model."""
        if len(self.mpip.implying_breakpoints) != 2:
            return
        self._add_stair_constraints_from_top()
        self._add_stair_constraints_from_bottom()

    def _add_stair_constraints_from_top(self) -> None:
        coeff_dict_z = {}
        for (x_index, y_index), implied_indices in self.mpip.relation.items():
            for z_index in implied_indices:
                coeff_dict_z[z_index] = max(
                    self.relation_matrix_size - x_index - y_index,
                    coeff_dict_z.get(z_index, 0),
                )
        self.opt_model.addConstr(
            gp.quicksum(
                (self.relation_matrix_size - i) * implying_var
                for implying_vars in self.mpip.implying_variables.values()
                for i, implying_var in enumerate(implying_vars)
            )
            - gp.quicksum(
                coeff * self.mpip.implied_variables[z_index]
                for z_index, coeff in coeff_dict_z.items()
            )
            <= self.relation_matrix_size,
            name=f"stair_top_{self.mpip.mpip_id}",
        )

    def _add_stair_constraints_from_bottom(self) -> None:
        coeff_dict_z = {}
        for (x_index, y_index), implied_indices in self.mpip.relation.items():
            for z_index in implied_indices:
                coeff_dict_z[z_index] = max(
                    x_index + y_index + 2 - self.relation_matrix_size,
                    coeff_dict_z.get(z_index, 0),
                )
        self.opt_model.addConstr(
            gp.quicksum(
                (i + 1) * implying_var
                for implying_vars in self.mpip.implying_variables.values()
                for i, implying_var in enumerate(implying_vars)
            )
            - gp.quicksum(
                coeff * self.mpip.implied_variables[z_index]
                for z_index, coeff in coeff_dict_z.items()
            )
            <= self.relation_matrix_size,
            name=f"stair_bottom_{self.mpip.mpip_id}",
        )

    def _generate_implying_combinations(
        self,
    ) -> Iterator[tuple[tuple[int, ...], tuple[gp.Var, ...]]]:
        """Generate all combinations of implying variables."""
        for combination in itertools.product(
            *[enumerate(d) for d in self.mpip.implying_variables.values()]
        ):
            indices, variables = zip(*combination)
            yield indices, variables

    def _add_mccormick_constraint(
        self,
        implying_indices: tuple[int, ...],
        implying_variables: tuple[gp.Var, ...],
        implied_variables: tuple[gp.Var, ...],
    ) -> None:
        """Create individual McCormick constraint."""
        self.opt_model.addConstr(
            -gp.quicksum(implying_variables) + gp.quicksum(implied_variables)
            >= -len(self.mpip.implying_variables) + 1,
            name=f"corm_{self.mpip.mpip_id}_{implying_indices}".replace(" ", ""),
        )

    def add_multiple_choice_constraints(self) -> None:
        """Add all multiple choice constraints to optimization model."""
        self._add_implying_mc_constraints()
        self._add_implied_mc_constraint()

    def _add_implying_mc_constraints(self) -> None:
        """Add multiple choice constraints for implying variables."""
        for implying_index, implying_variables in self.mpip.implying_variables.items():
            self.opt_model.addConstr(
                gp.quicksum(implying_variables) == 1,
                name=f"mc_implying_{implying_index}_{self.mpip.mpip_id}",
            )

    def _add_implied_mc_constraint(self) -> None:
        """Add multiple choice constraint for implied variables."""
        self.opt_model.addConstr(
            gp.quicksum(self.mpip.implied_variables) == 1,
            name=f"mc_implied_{self.mpip.mpip_id}",
        )

    def build_separation_model(self) -> None:
        """Construct separation model with variables and constraints."""
        self._add_separation_variables()
        self._add_separation_constraints()

    def _add_separation_variables(self) -> None:
        """Create variables for separation model."""
        for implying_index, implying_variables in self.mpip.implying_variables.items():
            sep_implying_variables = []
            for i in range(len(implying_variables)):
                var = self.separation_model.addVar(
                    name=f"sep_implying_{implying_index}_{i}", ub=1, obj=-1
                )
                sep_implying_variables.append(var)
            self.sep_implying_variables[implying_index] = sep_implying_variables

        self.sep_implied_variables = [
            self.separation_model.addVar(name=f"sep_implied_{i}", ub=1, obj=1)
            for i in range(len(self.mpip.implied_variables))
        ]

    def _add_separation_constraints(self) -> None:
        """Add constraints to separation model."""
        for (
            implying_indices,
            implying_variables,
        ) in self._generate_separation_combinations():
            for implied_index in self.mpip.relation[implying_indices]:
                implied_variable = self.sep_implied_variables[implied_index]
                self.separation_model.addConstr(
                    sum(implying_variables) - implied_variable
                    <= len(self.mpip.implying_variables) - 1
                )

    def _generate_separation_combinations(
        self,
    ) -> Iterator[tuple[tuple[int, ...], tuple[gp.Var, ...]]]:
        """Generate combinations for separation constraints."""
        for combination in itertools.product(
            *[enumerate(d) for d in self.sep_implying_variables.values()]
        ):
            indices, variables = zip(*combination)
            yield indices, variables

    def separate_point(self) -> None:
        """Perform separation of current point."""
        self._update_separation_objective()
        self.separation_model.optimize()
        self._generate_cut()

    def _update_separation_objective(self) -> None:
        """Update separation model objective with current point values."""

        for (
            implying_index,
            implying_values,
        ) in self.point_to_be_separated.implying_values.items():
            for idx, val in enumerate(implying_values):
                self.sep_implying_variables[implying_index][idx].obj = -val

        for idx, val in enumerate(self.point_to_be_separated.implied_values):
            self.sep_implied_variables[idx].obj = val

    def _generate_cut(self) -> None:
        """Generate cut based on separation solution."""
        # Calculate violation
        implying_sum = sum(
            var.X
            * self.point_to_be_separated.original_implying_values[implying_index][idx]
            for implying_index, variables in self.sep_implying_variables.items()
            for idx, var in enumerate(variables)
        )

        implied_sum = sum(
            var.X * self.point_to_be_separated.original_implied_values[idx]
            for idx, var in enumerate(self.sep_implied_variables)
        )

        violation = implying_sum - implied_sum - len(self.sep_implying_variables) + 1
        self.cut.violation = violation

        # Create cut if violation is significant
        if violation > s.StaticSettings.min_cut_violation:
            lhs = gp.quicksum(
                self.sep_implying_variables[implying_index][idx].X * var
                for implying_index, implying_variables in self.mpip.implying_variables.items()
                for idx, var in enumerate(implying_variables)
            ) - gp.quicksum(
                self.sep_implied_variables[idx].X * var
                for idx, var in enumerate(self.mpip.implied_variables)
            )
            self.cut.lhs = lhs
            self.cut.rhs = len(self.sep_implying_variables) - 1
        else:
            self.cut.rhs = 0

    def separate_solution(self) -> None:
        """Extract solution point and initiate separation."""
        self.point_to_be_separated.implying_values = {
            implying_index: [self.opt_model.cbGetNodeRel(var) for var in variables]
            for implying_index, variables in self.mpip.implying_variables.items()
        }
        self.point_to_be_separated.implied_values = [
            self.opt_model.cbGetNodeRel(var) for var in self.mpip.implied_variables
        ]
        if self.point_to_be_separated.is_integer():
            self.cut.rhs = 0
        else:
            self.point_to_be_separated.perturb()
            self.separate_point()
