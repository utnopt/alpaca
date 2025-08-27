# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import bisect
import pyscipopt as scip
import numpy as np

import alpaca.settings as s
import alpaca.model_data.variable as var


class MPIP:  # pylint: disable=too-many-instance-attributes
    """Multipartite Implication Polytope."""

    def __init__(self, mpip_id) -> None:
        """Initialize MPIP instance."""
        self.mpip_id = mpip_id
        self.implying_variables: dict[str, var.Variable] = {}
        self.implied_variable: var.Variable | None = None
        self.relation = {}
        self.relation_ratio = 0.0
        self.implying_function = scip.Expr()
        self.interval_lp = scip.Model()
        self.interval_lp.hideOutput()
        self.interval_lp_implying_vars = {}
        self.interval_lp_implied_var = 0
        self.separator = None
        self.feasible = True

    def build_mpip(self) -> None:
        """Build MPIP structure."""
        self._calculate_relation_function()

    def calculate_relation_ratio(self) -> None:
        """Calculate relation ratio."""
        self.relation_ratio = sum(
            len(implied_indices) for implied_indices in self.relation.values()
        ) / len(self.relation)

    def build_implied_values_list(self, num_grid_points=100) -> None:
        """Build list of implied values for domain of implying variables."""
        implying_domains = [
            np.linspace(variable.lb, variable.ub, num_grid_points)
            for variable in self.implying_variables.values()
        ]

        # Create a grid of all possible combinations of values from the domains
        value_grid = itertools.product(*implying_domains)

        # Get the SCIP variables in the correct order to match the domains
        scip_vars = [
            self.interval_lp_implying_vars[name] for name in self.implying_variables
        ]
        # Iterate over each point (combination of values) in the grid
        for point in value_grid:
            # Fix the implying variables to the values in the current grid point
            for variable, val in zip(scip_vars, point):
                self.interval_lp.chgVarLb(variable, val)
                self.interval_lp.chgVarUb(variable, val)

            # Solve the trivial problem to calculate the value of the implied variable
            self.interval_lp.setObjective(self.interval_lp_implied_var, "minimize")
            self.interval_lp.optimize()

            if self.interval_lp.getStatus() == "optimal":
                evaluated_value = self.interval_lp.getObjVal()
                self.implied_variable.implied_values_list.append(evaluated_value)
            self.interval_lp.freeTransform()

    def add_implied_id(
        self,
        variable: var.Variable,
    ) -> None:
        """Add implied variable information."""
        self.implied_variable = variable
        self.interval_lp_implied_var = self.interval_lp.addVar(
            f"x_{variable.name}", lb=-s.StaticSettings.infinity
        )

    def add_implying_id(
        self,
        variable: var.Variable,
    ) -> None:
        """Add implying variable information."""
        self.implying_variables.update({variable.name: variable})
        self.interval_lp_implying_vars[variable.name] = self.interval_lp.addVar(
            f"x_{variable.name}"
        )

    def _calculate_relation_function(self) -> None:
        breakpoint_ranges = []
        for variable in self.implying_variables.values():
            bp = variable.breakpoints
            intervals = [(i, (bp[i], bp[i + 1])) for i in range(len(bp) - 1)]
            breakpoint_ranges.append(intervals)

        for combo in itertools.product(*breakpoint_ranges):
            key = tuple(implying_index for implying_index, _ in combo)
            intervals = tuple(interval for _, interval in combo)
            interval_dict = dict(zip(list(self.implying_variables.keys()), intervals))
            feasible, lb, ub = self._implied_interval_scip(interval_dict)
            if feasible:
                self.relation[key] = self._calculate_implied_relation_from_interval(
                    lb, ub
                )
            else:
                self.relation[key] = ()

    def _calculate_implied_relation_from_interval(self, lb: float, ub: float) -> tuple:
        if self.implied_variable.breakpoints[0] == ub:
            return (len(self.implied_variable.breakpoints) - 2,)
        if (
            self.implied_variable.breakpoints[0] > ub
            or self.implied_variable.breakpoints[-1] < lb
        ):
            return ()
        idx1 = max(0, bisect.bisect_left(self.implied_variable.breakpoints, lb) - 1)
        idx2 = min(
            bisect.bisect_left(self.implied_variable.breakpoints, ub) - 1,
            len(self.implied_variable.breakpoints) - 2,
        )
        return tuple(range(idx1, idx2 + 1))

    def _implied_interval_scip(
        self, intervals: dict[str, tuple[float, float]]
    ) -> tuple[bool, float, float]:
        self.interval_lp.freeTransform()
        for implying_index, (low, high) in intervals.items():
            implying_var = self.interval_lp_implying_vars[implying_index]
            self.interval_lp.chgVarLb(implying_var, low)
            self.interval_lp.chgVarUb(implying_var, high)

        self.interval_lp.setObjective(self.interval_lp_implied_var, "minimize")
        self.interval_lp.optimize()
        if self.interval_lp.getStatus() == "infeasible":
            return False, 0.0, 0.0
        lower_bound = round(
            self.interval_lp.getObjVal(), s.StaticSettings.rounding_precision
        )
        self.interval_lp.freeTransform()

        self.interval_lp.setObjective(self.interval_lp_implied_var, "maximize")
        self.interval_lp.optimize()
        if self.interval_lp.getStatus() == "infeasible":
            return False, 0.0, 0.0
        upper_bound = round(
            self.interval_lp.getObjVal(), s.StaticSettings.rounding_precision
        )
        return True, lower_bound, upper_bound
