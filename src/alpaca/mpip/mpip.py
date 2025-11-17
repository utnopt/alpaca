# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import bisect

import alpaca.settings as s
import alpaca.model_data.variable as var
import alpaca.external_solvers.solver_wrapper as sw
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class MPIP:  # pylint: disable=too-many-instance-attributes
    """
    Multipartite Implication Polytope (MPIP) structure.
    The underlying mathematical structure for the logical conditions introduced
     in this class can be described by implication polytopes. When a
    continuous variable's domain is discretized, a set of binary variables is
    introduced to represent which interval the continuous variable falls into.
    The constraints that dictate that exactly one binary variable can be active
    (an SOS1 constraint) and the subsequent implications on the model's
    functions and other variables form a system of conditional relations.
    For a deeper theoretical understanding of the polytopes governing such
    conditional relationships between sets of binary variables, see:
    Burlacu, Gemander and Kuen (2024): The Bipartite Implication Polytope:
    Conditional Relations over Multiple Sets of Binary Variables.
    Link: https://optimization-online.org/?p=26208
    """

    def __init__(self, mpip_id: str, pwl_method: str) -> None:
        """Initialize MPIP instance."""
        self.mpip_id = mpip_id
        self.implying_variables: dict[str, var.Variable] = {}
        self.implied_variable: var.Variable | None = None
        self.relation = {}
        self.interval_lp = sw.SolverWrapper(mip_solver=lsf.solver_name_gurobi())
        self.implying_function = self.interval_lp.nonlinear_expression()
        self.interval_lp.hide_output()
        self.interval_lp_implying_vars = {}
        self.interval_lp_implied_var = 0
        self.separator = None
        self.feasible = True
        self.pwl_method = pwl_method

    def build_mpip(self) -> None:
        """Build MPIP structure."""
        self._add_implying_function_to_interval_lp()
        self._calculate_relation_function()

    def add_implied_id(
        self,
        variable: var.Variable,
    ) -> None:
        """Add implied variable information."""
        self.implied_variable = variable
        self.interval_lp_implied_var = self.interval_lp.add_variable(
            lsf.mpip_interval_lp_var_name(variable.name), lb=-s.StaticSettings.infinity
        )

    def add_implying_id(
        self,
        variable: var.Variable,
    ) -> None:
        """Add implying variable information."""
        self.implying_variables.update({variable.name: variable})
        self.interval_lp_implying_vars[variable.name] = self.interval_lp.add_variable(
            lsf.mpip_interval_lp_var_name(variable.name)
        )

    def calculate_relation_ratio(self) -> float:
        """Calculate relation ratio."""
        return sum(
            len(implied_indices) for implied_indices in self.relation.values()
        ) / len(self.relation)

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

    def _add_implying_function_to_interval_lp(self) -> None:
        self.interval_lp.add_constraint(
            self.interval_lp_implied_var == self.implying_function
        )

    def _implied_interval_scip(
        self, intervals: dict[str, tuple[float, float]]
    ) -> tuple[bool, float, float]:
        for implying_index, (low, high) in intervals.items():
            implying_var = self.interval_lp_implying_vars[implying_index]
            self.interval_lp.set_variable_lb(implying_var, low)
            self.interval_lp.set_variable_ub(implying_var, high)

        self.interval_lp.set_objective(
            self.interval_lp_implied_var, sense=lsf.objective_sense_minimize()
        )
        self.interval_lp.optimize()
        if self.interval_lp.is_infeasible():
            return False, 0.0, 0.0
        lower_bound = round(
            self.interval_lp.get_objective_value(), s.StaticSettings.rounding_precision
        )

        self.interval_lp.set_objective(
            self.interval_lp_implied_var, lsf.objective_sense_maximize()
        )
        self.interval_lp.optimize()
        if self.interval_lp.is_infeasible():
            return False, 0.0, 0.0
        upper_bound = round(
            self.interval_lp.get_objective_value(), s.StaticSettings.rounding_precision
        )
        return True, lower_bound, upper_bound
