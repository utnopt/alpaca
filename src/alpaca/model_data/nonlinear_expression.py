# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations

import alpaca.utils.datahandling as udh
from alpaca.model_data import variable as var


class NonlinearExpression:
    """Nonlinear expression."""

    def __init__(self, name: str, expression_tag, model_data=None):
        self.name = name
        self.expression_type = expression_tag.name
        self.expression_tag = expression_tag
        self.model_data: "ModelData" | None = model_data
        self.child_expressions = []
        self.representative_variable = var.Variable(f"r_{self.name}")

    def grow_expression_tree(self):
        """Grow expression tree."""
        for child_expression_tag in self.expression_tag.contents:
            child_expression_tag_name = udh.hash_nonlinearity(str(child_expression_tag))
            if child_expression_tag.name == "variable":
                coeff = (
                    1.0
                    if child_expression_tag.get("coef") is None
                    else float(child_expression_tag.get("coef"))
                )
                variable = self.model_data.variables[
                    f"x_{child_expression_tag.get('idx')}"
                ]
                self.child_expressions.append((coeff, variable))
            elif child_expression_tag.name == "number":
                self.child_expressions.append(float(child_expression_tag.get("value")))
            else:
                self._add_nonlinear_expression_child(
                    child_expression_tag_name, child_expression_tag
                )

    def _add_nonlinear_expression_child(
        self, child_expression_tag_name, child_expression_tag
    ):
        if child_expression_tag_name in self.model_data.nonlinear_expressions:
            self.child_expressions.append(
                self.model_data.nonlinear_expressions[child_expression_tag_name]
            )
            return
        child_expression = NonlinearExpression(
            child_expression_tag_name, child_expression_tag, self.model_data
        )
        self.model_data.nonlinear_expressions[child_expression_tag_name] = (
            child_expression
        )
        self.child_expressions.append(child_expression)
        child_expression.grow_expression_tree()

    def __repr__(self):
        return self.expression_type + "_" + self.name
