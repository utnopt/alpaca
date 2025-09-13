# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class BoundPropagator:
    """Handles bound propagation for constraints and expressions."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data

    def propagate_linear_constraints(self):
        """Performs bound propagation on linear equality constraints."""
        for _ in range(self.model_data.settings.bound_propagation_rounds):
            for constraint in self.model_data.constraints.values():
                if constraint.con_type == "==":
                    constraint.propagate_variable_bounds()

    def propagate_expressions(self):
        """Performs bound propagation on all expressions for a set number of rounds."""
        sorted_expressions = sorted(
            self.model_data.expressions.all_low_dim_expressions(),
            key=lambda e: -e.level,
        )
        for _ in range(self.model_data.settings.bound_propagation_rounds):
            for expression in sorted_expressions:
                expression.propagate_variable_bounds()
