# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from dataclasses import dataclass, field

from alpaca.expressions import (
    linear_expression as lie,
    bilinear_expression as ble,
    multilinear_expression as mle,
    one_dim_expression as ode,
    nonlinear_expression as nle,
)


@dataclass
class ExpressionContainer:
    """Container class for expressions."""

    nonlinear_expressions: dict[str, nle.NonlinearExpression] = field(
        default_factory=dict
    )
    first_level_nonlinear_expression_keys: list[str] = field(default_factory=list)
    one_dim_expressions: dict[str, ode.OneDimExpression] = field(default_factory=dict)
    linear_expressions: dict[str, lie.LinearExpression] = field(default_factory=dict)
    bilinear_expressions: dict[str, ble.BilinearExpression] = field(
        default_factory=dict
    )
    multilinear_expressions: dict[str, mle.MultilinearExpression] = field(
        default_factory=dict
    )

    def all_low_dim_expressions(self) -> list:
        """Return all low dimensional expressions."""
        return (
            list(self.one_dim_expressions.values())
            + list(self.linear_expressions.values())
            + list(self.bilinear_expressions.values())
            + list(self.multilinear_expressions.values())
        )
