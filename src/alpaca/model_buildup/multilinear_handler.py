# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class MultilinearHandler:
    """Manages reformulation and handling of multilinear and bilinear expressions."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data

    def handle(self):
        """Applies reformulations based on user settings."""
        if self.model_data.settings.reformulate_multilinear_to_bilinear:
            self._reformulate_multilinear_to_bilinear()
        else:
            self._mark_representative_variables_of_multilinear_expressions_as_discretized()

        if self.model_data.settings.bilinear_handling == 1:
            self._reformulate_bilinear_to_sum_of_squares()
        elif self.model_data.settings.bilinear_handling == 2:
            self._mark_representative_variables_of_bilinear_expressions_as_discretized()

    def add_mccormick_envelopes(self) -> None:
        """Adds McCormick envelope constraints to bilinear expressions."""
        if self.model_data.settings.bilinear_handling == 0:
            self._add_mccormick_envelope_to_bilinear_expressions()

    def _reformulate_multilinear_to_bilinear(self) -> None:
        """Reformulates multilinear expressions into bilinear expressions."""
        for expression in self.model_data.expressions.multilinear_expressions.values():
            expression.reformulate_to_bilinear_expressions()

    def _mark_representative_variables_of_multilinear_expressions_as_discretized(
        self,
    ) -> None:
        """Marks representative variables of multilinear expressions for discretization."""
        for expression in self.model_data.expressions.multilinear_expressions.values():
            expression.representative_variable.add_nonlinearity_to_occurring_in(
                f"multilinearimplied{len(expression.variables)}"
            )

    def _mark_representative_variables_of_bilinear_expressions_as_discretized(
        self,
    ) -> None:
        """Marks representative variables of bilinear expressions for discretization."""
        for expression in self.model_data.expressions.bilinear_expressions.values():
            expression.representative_variable.add_nonlinearity_to_occurring_in(
                "multilinearimplied2"
            )

    def _add_mccormick_envelope_to_bilinear_expressions(self) -> None:
        """Adds McCormick envelope constraints to bilinear expressions."""
        for expression in self.model_data.expressions.bilinear_expressions.values():
            expression.add_mccormick_envelope()

    def _reformulate_bilinear_to_sum_of_squares(self) -> None:
        """Reformulates bilinear expressions into a sum of squares."""
        for expression in self.model_data.expressions.bilinear_expressions.values():
            expression.reformulate_to_sum_of_squares()
