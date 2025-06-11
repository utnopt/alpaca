# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var


class MultilinearExpression:
    """Multilinear expression."""

    def __init__(
        self,
        name: str,
        variables: list[var.Variable],
        model_data,
        representative_variable=None,
    ):
        self.name = name
        self.variables = variables
        # pylint: disable=duplicate-code
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.variables.setdefault(f"r_{name}", var.Variable(f"r_{name}"))
        )

    def __repr__(self):
        return self.name
