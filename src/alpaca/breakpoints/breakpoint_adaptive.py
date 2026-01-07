# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np

import alpaca.model_data.variable as var
import alpaca.settings as s


class BreakpointAdaptive:
    """Generates breakpoints adaptive to match relaxation accuracy."""

    def __init__(self, variable: var.Variable, settings: s.UserSettings) -> None:
        self.variable = variable
        self.breakpoints = np.linspace(
            self.variable.lb, self.variable.ub, num=settings.number_of_breakpoints
        ).tolist()
        self.settings = settings
        self._build_breakpoints()

    def _build_breakpoints(self) -> None:
        refinement_intervals = self._find_refinement_intervals()
        while refinement_intervals:
            self._refine_intervals(refinement_intervals)
            refinement_intervals = self._find_refinement_intervals()

    def _find_refinement_intervals(self) -> set[int]:
        refinement_intervals = []
        for expression in self.variable.occurring_in.values():
            for i, bp in enumerate(self.breakpoints[:-1]):
                slope, intercept = (
                    expression.get_linear_approximation_function_parameters_for_segment(
                        bp, self.breakpoints[i + 1]
                    )
                )
                min_deviation, max_deviation = expression.get_min_max_deviation(
                    bp, self.breakpoints[i + 1], slope, intercept
                )
                if (
                    abs(min_deviation) > self.settings.relaxation_tolerance
                    or abs(max_deviation) > self.settings.relaxation_tolerance
                ):
                    refinement_intervals.append(i)
        return set(refinement_intervals)

    def _refine_intervals(self, refinement_intervals: set[int]) -> None:
        new_breakpoints = []
        for i, bp in enumerate(self.breakpoints[:-1]):
            new_breakpoints.append(bp)
            if i in refinement_intervals:
                new_breakpoint = (bp + self.breakpoints[i + 1]) / 2
                new_breakpoints.append(new_breakpoint)
        new_breakpoints.append(self.breakpoints[-1])
        self.breakpoints = sorted(new_breakpoints)
