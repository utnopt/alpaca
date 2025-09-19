# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from alpaca.model_data import variable as var
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class Expression:
    """Represents an expression z = ...
    The expression can be linear, bilinear, multilinear or one-dimensional.

    Attributes:
        name: Identifier for the expression
        representative_variable: Variable representing the product (z)
        level: Level of the expression in expression tree
    """

    def __init__(
        self,
        name: str,
        model_data: ModelData,
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        """Initialize bilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            level: Level of the expression in expression tree
            representative_variable: Optional existing variable to represent product
        """
        self.name = name
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.add_variable(
                var.Variable(lsf.representative_variable_name(name))
            )
        )
        self.level = level
        self.solver_constraint = None

    def __repr__(self):
        return self.name
