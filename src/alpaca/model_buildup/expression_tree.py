# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class ExpressionTree:
    """Handles the decomposition of complex nonlinear expression trees."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data

    def decompose(self):
        """Decomposes complex expression trees into simpler, low-dimensional functions."""
        self.model_data.expressions.first_level_nonlinear_expression_keys = list(
            self.model_data.expressions.nonlinear_expressions.keys()
        )
        self._grow_nonlinear_expression_trees()
        self._fragment_expression_trees_to_low_dimensional_functions()

    def _grow_nonlinear_expression_trees(self) -> None:
        """Builds expression trees for all top-level nonlinear expressions."""
        for (
            nonlinear_expression_key
        ) in self.model_data.expressions.first_level_nonlinear_expression_keys:
            nl_expression = self.model_data.expressions.nonlinear_expressions[
                nonlinear_expression_key
            ]
            nl_expression.grow_expression_tree()

    def _fragment_expression_trees_to_low_dimensional_functions(self) -> None:
        """Decomposes complex expression trees into simpler, low-dimensional functions."""
        for (
            nonlinear_expression_key
        ) in self.model_data.expressions.first_level_nonlinear_expression_keys:
            nl_expression = self.model_data.expressions.nonlinear_expressions[
                nonlinear_expression_key
            ]
            nl_expression.fragment_expression_tree_to_low_dimensional_functions(1)
