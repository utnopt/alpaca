# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses
import numpy as np

import alpaca.settings as s


@dataclasses.dataclass
class Variable:
    """Variable."""

    def __init__(self, name: str, lb=0, ub=s.StaticSettings.infinity, var_type="C"):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.var_type = var_type
        self.is_discretized = False
        self.breakpoints = []

    def discretize_variable(self, number_of_breakpoints: int):
        """Discretize variable."""
        if self.is_discretized:
            return
        self.breakpoints = np.linspace(self.lb, self.ub, number_of_breakpoints)
        self.is_discretized = True

    def __repr__(self):
        return self.name

    def __hash__(self):
        return self.name
