# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses
import numpy as np


@dataclasses.dataclass
class Variable:
    """Variable."""

    def __init__(self, name: str, lb=0, ub=np.inf, var_type="C"):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.var_type = var_type

    def __repr__(self):
        return self.name

    def __hash__(self):
        return self.name
