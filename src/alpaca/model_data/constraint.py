# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses

from alpaca.model_data import variable as var


@dataclasses.dataclass
class Constraint:
    """Constraint."""

    def __init__(self, name: str):
        self.name = name
        self.type = "<="
        self.rhs = 0.0
        self.variables: list[tuple[float, var.Variable]] = []

    def __repr__(self):
        return self.name
