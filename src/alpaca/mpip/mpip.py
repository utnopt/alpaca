# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import itertools
import bisect
import pyscipopt as scip

import alpaca.settings as s


class MPIP:  # pylint: disable=too-many-instance-attributes
    """Multipartite Implication Polytope."""

    def __init__(self, mpip_id: str) -> None:
        """Initialize MPIP instance."""
        self.mpip_id = mpip_id
        self.implying_breakpoints: dict[str, list[float]] = {}
        self.implied_breakpoints: list[float] = []
        self.implied_id = ""
        self.implying_variables: dict[str, list[scip.Variable]] = {}
        self.implied_variables: list[scip.Variable] = []
        self.relation = {}
        self.implying_function = scip.Expr()
        self.interval_lp = scip.Model()
        self.interval_lp.hideOutput()
        self.interval_lp_implying_vars = {}
        self.interval_lp_implied_var = 0
        self.separator = None
        self.feasible = True

    def build_mpip(self) -> None:
        """Build MPIP structure."""
        self._add_implying_function_to_interval_lp()
        self._calculate_relation_function()

    def add_implied_id(self, implied_id: str, breakpoints: list[float]) -> None:
        """Add implied variable information."""
        self.implied_id = implied_id
        self.implied_breakpoints = breakpoints
        self.interval_lp_implied_var = self.interval_lp.addVar(
            f"x_{self.implied_id}", lb=-s.StaticSettings.infinity
        )

    def add_implying_id(self, implying_id: str, breakpoints: list[float]) -> None:
        """Add implying variable information."""
        self.implying_variables.update({implying_id: []})
        self.implying_breakpoints[implying_id] = breakpoints
        var_name = f"x_{implying_id}"
        self.interval_lp_implying_vars[implying_id] = self.interval_lp.addVar(var_name)

    def _calculate_relation_function(self) -> None:
        breakpoint_ranges = []
        for bp in self.implying_breakpoints.values():
            intervals = [(bp[i], bp[i + 1]) for i in range(len(bp) - 1)]
            breakpoint_ranges.append(intervals)

        for combo in itertools.product(*breakpoint_ranges):
            key = tuple(low for low, _ in combo)
            interval_dict = dict(zip(list(self.implying_breakpoints.keys()), combo))
            feasible, lb, ub = self._implied_interval_scip(interval_dict)
            if feasible:
                self.relation[key] = self._calculate_implied_relation_from_interval(
                    lb, ub
                )
            else:
                self.relation[key] = ()

    def _calculate_implied_relation_from_interval(self, lb: float, ub: float) -> tuple:
        if self.implied_breakpoints[0] == ub:
            return (ub,)
        if self.implied_breakpoints[0] > ub or self.implied_breakpoints[-1] < lb:
            return ()
        idx1 = max(0, bisect.bisect_left(self.implied_breakpoints, lb) - 1)
        idx2 = bisect.bisect_left(self.implied_breakpoints, ub) - 1
        return tuple(self.implied_breakpoints[idx1 : idx2 + 1])

    def _add_implying_function_to_interval_lp(self) -> None:
        self.interval_lp.addCons(self.interval_lp_implied_var == self.implying_function)

    def _implied_interval_scip(
        self, intervals: dict[str, tuple[float, float]]
    ) -> tuple[bool, float, float]:
        self.interval_lp.freeTransform()
        for implying_index, (low, high) in intervals.items():
            var = self.interval_lp_implying_vars[implying_index]
            self.interval_lp.chgVarLb(var, low)
            self.interval_lp.chgVarUb(var, high)

        self.interval_lp.setObjective(self.interval_lp_implied_var, "minimize")
        self.interval_lp.optimize()
        if self.interval_lp.getStatus() == scip.SCIP_STATUS_INFEASIBLE:
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
