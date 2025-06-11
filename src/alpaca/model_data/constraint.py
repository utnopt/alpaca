# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses

from alpaca.model_data import variable as var


@dataclasses.dataclass
class Constraint:
    """Constraint."""

    def __init__(self, name: str, con_type="<=", rhs=0.0, variables=None):
        self.name = name
        self.con_type = con_type
        self.rhs = rhs
        self.variables: list[tuple[float, var.Variable]] = (
            [] if variables is None else variables
        )

    def __repr__(self):
        return self.name
