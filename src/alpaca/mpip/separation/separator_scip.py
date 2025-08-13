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
import pyscipopt as scip

import alpaca.settings as s
import alpaca.mpip.mpip as mp


@dataclasses.dataclass
class MPIPCut:
    """Separated callback cut for MPIP separation."""

    lhs: scip.Expr | None = None
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


class Separator:
    """Multipartite Implication Polytope separation handler."""

    def __init__(self, mpip: mp.MPIP, opt_model: scip.Model) -> None:
        self.separation_model = scip.Model()
        self._setup_separation_model()
        self.mpip = mpip
        self.sep_implying_variables: dict[str, list[scip.Variable]] = {}
        self.sep_implied_variables: list[scip.Variable] = []
        self.point_to_be_separated = SeparatedPoint()
        self.opt_model = opt_model
        self.cut = MPIPCut()

    def _setup_separation_model(self) -> None:
        """Configure separation model settings."""
        self.separation_model.hideOutput()
        self.separation_model.setParam("randomization/randomseedshift", 0)
        self.separation_model.setParam("reoptimization/enable", True)
        self.separation_model.setMaximize()

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

    def _generate_implying_combinations(
        self,
    ) -> Iterator[tuple[tuple[int, ...], tuple[scip.Variable, ...]]]:
        """Generate all combinations of implying variables."""
        for combination in itertools.product(
            *[enumerate(d) for d in self.mpip.implying_variables.values()]
        ):
            indices, variables = zip(*combination)
            yield indices, variables

    def _add_mccormick_constraint(
        self,
        implying_indices: tuple[int, ...],
        implying_variables: tuple[scip.Variable, ...],
        implied_variables: tuple[scip.Variable, ...],
    ) -> None:
        """Create individual McCormick constraint."""
        self.opt_model.addCons(
            -sum(implying_variables) + sum(implied_variables)
            >= -len(self.mpip.implying_variables) + 1,
            name=f"corm_{self.mpip.mpip_id}_{implying_indices}".replace(" ", ""),
        )

    def add_stair_constraints(self) -> None:
        """Add stair constraints to optimization model."""

    def add_multiple_choice_constraints(self) -> None:
        """Add all multiple choice constraints to optimization model."""
        self._add_implying_mc_constraints()
        self._add_implied_mc_constraint()

    def _add_implying_mc_constraints(self) -> None:
        """Add multiple choice constraints for implying variables."""
        for implying_index, implying_variables in self.mpip.implying_variables.items():
            self.opt_model.addCons(
                sum(implying_variables) == 1,
                name=f"mc_implying_{implying_index}_{self.mpip.mpip_id}",
            )

    def _add_implied_mc_constraint(self) -> None:
        """Add multiple choice constraint for implied variables."""
        self.opt_model.addCons(
            sum(self.mpip.implied_variables) == 1,
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
                    f"sep_implying_{implying_index}_{i}", ub=1, obj=1
                )
                sep_implying_variables.append(var)
            self.sep_implying_variables[implying_index] = sep_implying_variables

        self.sep_implied_variables = [
            self.separation_model.addVar(f"sep_implied_{i}", ub=1, obj=-1)
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
                self.separation_model.addCons(
                    sum(implying_variables) - implied_variable
                    <= len(self.mpip.implying_variables) - 1
                )

    def _generate_separation_combinations(
        self,
    ) -> Iterator[tuple[tuple[int, ...], tuple[scip.Variable, ...]]]:
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
        self.separation_model.freeReoptSolve()
        new_objective = scip.Expr()
        for (
            implying_index,
            implying_values,
        ) in self.point_to_be_separated.implying_values.items():
            for idx, val in enumerate(implying_values):
                new_objective += val * self.sep_implying_variables[implying_index][idx]

        for idx, val in enumerate(self.point_to_be_separated.implied_values):
            new_objective -= val * self.sep_implied_variables[idx]
        self.separation_model.chgReoptObjective(new_objective, "maximize")

    def _generate_cut(self) -> None:
        """Generate cut based on separation solution."""
        # Calculate violation
        implying_sum = sum(
            self.separation_model.getVal(var)
            * self.point_to_be_separated.original_implying_values[implying_index][idx]
            for implying_index, variables in self.sep_implying_variables.items()
            for idx, var in enumerate(variables)
        )

        implied_sum = sum(
            self.separation_model.getVal(var)
            * self.point_to_be_separated.original_implied_values[idx]
            for idx, var in enumerate(self.sep_implied_variables)
        )

        violation = implying_sum - implied_sum - len(self.sep_implying_variables) + 1
        self.cut.violation = violation

        # Create cut if violation is significant
        if violation > s.StaticSettings.min_cut_violation:
            lhs = sum(
                self.separation_model.getVal(
                    self.sep_implying_variables[implying_index][idx]
                )
                * var
                for implying_index, implying_variables in self.mpip.implying_variables.items()
                for idx, var in enumerate(implying_variables)
            ) - sum(
                self.separation_model.getVal(self.sep_implied_variables[idx]) * var
                for idx, var in enumerate(self.mpip.implied_variables)
            )
            self.cut.lhs = lhs
            self.cut.rhs = len(self.sep_implying_variables) - 1
        else:
            self.cut.rhs = 0

    def separate_solution(self) -> None:
        """Extract solution point and initiate separation."""
        self.point_to_be_separated.implying_values = {
            implying_index: [self.opt_model.getVal(var) for var in variables]
            for implying_index, variables in self.mpip.implying_variables.items()
        }
        self.point_to_be_separated.implied_values = [
            self.opt_model.getVal(var) for var in self.mpip.implied_variables
        ]
        if self.point_to_be_separated.is_integer():
            self.cut.rhs = 0
        else:
            self.point_to_be_separated.perturb()
            self.separate_point()
