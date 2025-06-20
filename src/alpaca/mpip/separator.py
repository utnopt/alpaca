# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""

import copy
import dataclasses
import itertools
import random
import pyscipopt as scip

import alpaca.settings as s  # pylint: disable=import-error


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
        self.implying_values: list[dict[int, float]] = []
        self.original_implying_values: list[dict[int, float]] = []
        self.implied_values: dict[int, float] = {}
        self.original_implied_values: dict[int, float] = {}
        self.perturbation_range = (
            s.StaticSettings.feasibility_tolerance / 100,
            10 * s.StaticSettings.feasibility_tolerance,
        )

    def is_integer(self) -> bool:
        """Check if all variables are integer within tolerance."""
        implying_checks = (
            abs(value - round(value)) < s.StaticSettings.feasibility_tolerance
            for values in self.implying_values
            for value in values.values()
        )
        implied_checks = (
            abs(value - round(value)) < s.StaticSettings.feasibility_tolerance
            for value in self.implied_values.values()
        )
        return all(itertools.chain(implying_checks, implied_checks))

    def perturb(self) -> None:
        """Apply random perturbation to variable values."""
        self.original_implying_values = copy.deepcopy(self.implying_values)
        self.original_implied_values = copy.deepcopy(self.implied_values)

        # Perturb implying values
        for values in self.implying_values:
            for key in values:
                values[key] += random.uniform(*self.perturbation_range)

        # Perturb implied values
        for key in self.implied_values:
            self.implied_values[key] += random.uniform(*self.perturbation_range)


class Separator:  # pylint: disable=too-many-instance-attributes
    """Multipartite Implication Polytope separation handler."""

    def __init__(self, mpip_id: str, opt_model: scip.Model) -> None:
        self.relation_matrix: dict[tuple[int, ...], int] = {}
        self.separation_model = scip.Model()
        self._setup_separation_model()
        self.mpip_id = mpip_id
        self.implying_variables: list[dict[int, scip.Variable]] = []
        self.implied_variables: dict[int, scip.Variable] = {}
        self.sep_implying_variables: list[dict[int, scip.Variable]] = []
        self.sep_implied_variables: dict[int, scip.Variable] = {}
        self.point_to_separate = SeparatedPoint()
        self.opt_model = opt_model
        self.cut = MPIPCut()

    def _setup_separation_model(self) -> None:
        """Configure separation model settings."""
        self.separation_model.hideOutput()
        self.separation_model.setParam("randomization/randomseedshift", 0)
        self.separation_model.setMaximize()

    def add_mc_cormick_constraints(self) -> None:
        """Add McCormick envelope constraints to optimization model."""
        for indices, variables in self._generate_implying_combinations():
            z_var = self.implied_variables.get(self.relation_matrix.get(indices, -1), 0)
            if z_var:
                self._add_mccormick_constraint(indices, variables, z_var)

    def _generate_implying_combinations(self) -> tuple[tuple[int, ...], scip.Variable]:
        """Generate all combinations of implying variables."""
        for combination in itertools.product(
            *[d.items() for d in self.implying_variables]
        ):
            indices, variables = zip(*combination)
            yield indices, variables

    def _add_mccormick_constraint(
        self,
        indices: tuple[int, ...],
        variables: tuple[scip.Variable, ...],
        z_var: scip.Variable,
    ) -> None:
        """Create individual McCormick constraint."""
        self.opt_model.addCons(
            -sum(variables) + z_var >= -len(self.implying_variables) + 1,
            name=f"corm_{self.mpip_id}_{indices}".replace(" ", ""),
        )

    def add_multiple_choice_constraints(self) -> None:
        """Add all multiple choice constraints to optimization model."""
        self._add_implying_mc_constraints()
        self._add_implied_mc_constraint()

    def _add_implying_mc_constraints(self) -> None:
        """Add multiple choice constraints for implying variables."""
        for i, variables in enumerate(self.implying_variables):
            self.opt_model.addCons(
                sum(variables.values()) == 1, name=f"mc_implying_{i}_{self.mpip_id}"
            )

    def _add_implied_mc_constraint(self) -> None:
        """Add multiple choice constraint for implied variables."""
        self.opt_model.addCons(
            sum(self.implied_variables.values()) == 1, name=f"mc_implied_{self.mpip_id}"
        )

    def build_separation_model(self) -> None:
        """Construct separation model with variables and constraints."""
        self._add_separation_variables()
        self._add_separation_constraints()

    def _add_separation_variables(self) -> None:
        """Create variables for separation model."""
        self.sep_implying_variables = []
        for i, variables in enumerate(self.implying_variables):
            var_dict = {}
            for name in variables:
                var = self.separation_model.addVar(
                    f"sep_implying_{i}_{name}", ub=1, obj=1
                )
                var_dict[name] = var
            self.sep_implying_variables.append(var_dict)

        self.sep_implied_variables = {
            name: self.separation_model.addVar(f"sep_implied_{name}", ub=1, obj=-1)
            for name in self.implied_variables
        }

    def _add_separation_constraints(self) -> None:
        """Add constraints to separation model."""
        for indices, variables in self._generate_separation_combinations():
            if indices in self.relation_matrix:
                z_var = self.sep_implied_variables[self.relation_matrix[indices]]
                self.separation_model.addCons(
                    sum(variables) - z_var <= len(self.implying_variables) - 1
                )

    def _generate_separation_combinations(
        self,
    ) -> tuple[tuple[int, ...], scip.Variable]:
        """Generate combinations for separation constraints."""
        for combination in itertools.product(
            *[d.items() for d in self.sep_implying_variables]
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
        for i, values in enumerate(self.point_to_separate.implying_values):
            for idx, val in values.items():
                self.sep_implying_variables[i][idx].obj = val

        for idx, val in self.point_to_separate.implied_values.items():
            self.sep_implied_variables[idx].obj = -val

    def _generate_cut(self) -> None:
        """Generate cut based on separation solution."""
        # Calculate violation
        implying_sum = sum(
            self.separation_model.getVal(var)
            * self.point_to_separate.original_implying_values[i][idx]
            for i, variables in enumerate(self.implying_variables)
            for idx, var in variables.items()
        )

        implied_sum = sum(
            self.separation_model.getVal(self.sep_implied_variables[idx])
            * self.point_to_separate.original_implied_values[idx]
            for idx in self.implied_variables
        )

        violation = implying_sum - implied_sum - len(self.implying_variables) + 1
        self.cut.violation = violation

        # Create cut if violation is significant
        if violation > s.StaticSettings.min_cut_violation:
            lhs = sum(
                self.separation_model.getVal(self.sep_implying_variables[i][idx]) * var
                for i, variables in enumerate(self.implying_variables)
                for idx, var in variables.items()
            ) - sum(
                self.separation_model.getVal(self.sep_implied_variables[idx]) * var
                for idx, var in self.implied_variables.items()
            )
            self.cut.lhs = lhs
            self.cut.rhs = len(self.implying_variables) - 1
        else:
            self.cut.rhs = 0

    def separate_solution(self) -> None:
        """Extract solution point and initiate separation."""
        # Extract current solution values
        self.point_to_separate.implying_values = [
            {idx: self.opt_model.getVal(var) for idx, var in variables.items()}
            for variables in self.implying_variables
        ]
        self.point_to_separate.implied_values = {
            idx: self.opt_model.getVal(var)
            for idx, var in self.implied_variables.items()
        }

        # Skip separation if solution is integer
        if self.point_to_separate.is_integer():
            self.cut.rhs = 0
        else:
            self.point_to_separate.perturb()
            self.separate_point()


class MPIPHandler:
    """Handler for multiple MPIP separation routines."""

    def __init__(self, opt_model: scip.Model) -> None:
        self.mpip_dict: dict[str, Separator] = {}
        self.iteration = 0
        self.cut_pool: list[MPIPCut] = []
        self.opt_model = opt_model
        self.nr_added_cuts = 0

    def add_mc_cormick_constraints(self) -> None:
        """Add McCormick constraints for all MPIPs."""
        for mpip in self.mpip_dict.values():
            mpip.add_mc_cormick_constraints()

    def add_multiple_choice_constraints(self) -> None:
        """Add multiple choice constraints for all MPIPs."""
        for mpip in self.mpip_dict.values():
            mpip.add_multiple_choice_constraints()

    def build_separation_models(self) -> None:
        """Build separation models for all MPIPs."""
        for mpip in self.mpip_dict.values():
            mpip.build_separation_model()

    def separate_solution(self) -> int:
        """Perform separation for current solution."""
        self.iteration += 1
        self.cut_pool = []

        for mpip in self.mpip_dict.values():
            mpip.separate_solution()
            if mpip.cut.rhs:  # Non-zero rhs indicates valid cut
                self.cut_pool.append(mpip.cut)

        if self.cut_pool:
            self._add_cuts_to_model()

        return self.nr_added_cuts

    def _add_cuts_to_model(self) -> None:
        """Add cuts meeting violation threshold to model."""
        max_violation = max(cut.violation for cut in self.cut_pool)
        min_violation = s.StaticSettings.max_violation_relation * max_violation
        self.nr_added_cuts = 0

        for cut in self.cut_pool:
            if cut.violation >= min_violation:
                self.opt_model.addCons(cut.lhs <= cut.rhs)
                self.nr_added_cuts += 1


def mpip_callback(model: scip.Model, where: scip.SCIP_STAGE) -> None:
    """Callback function for MPIP separation during solving."""
    if where == scip.SCIP_STAGE.SOLVING:
        if model.getStatus() == "optimal":
            model._mpip_handler.separate_solution()  # pylint: disable=protected-access
