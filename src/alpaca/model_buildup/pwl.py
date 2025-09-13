# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class PWL:
    """Applies Piecewise Linear relaxations to expressions."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data
        self.settings = model_data.settings

    def apply_relaxations(self):
        """Applies all PWL relaxations based on model settings."""
        self._apply_piecewise_linear_relaxation_for_one_dim_expressions()
        self._apply_piecewise_constant_relaxation_for_multilinear_expressions()

    def _apply_piecewise_linear_relaxation_for_one_dim_expressions(self) -> None:
        """Applies piecewise linear relaxation to all one-dimensional expressions."""
        for expression in self.model_data.expressions.one_dim_expressions.values():
            expression.apply_piecewise_linear_relaxation(
                approximation=self.settings.approximation
            )

    def _apply_piecewise_constant_relaxation_for_multilinear_expressions(
        self,
    ) -> None:
        """Applies piecewise constant relaxation to bilinear and multilinear expressions."""
        if self.settings.bilinear_handling == 2:
            for expression in self.model_data.expressions.bilinear_expressions.values():
                expression.apply_piecewise_constant_relaxation(
                    approximation=self.settings.approximation,
                )
        if not self.settings.reformulate_multilinear_to_bilinear:
            for (
                expression
            ) in self.model_data.expressions.multilinear_expressions.values():
                expression.apply_piecewise_constant_relaxation(
                    approximation=self.settings.approximation,
                )
