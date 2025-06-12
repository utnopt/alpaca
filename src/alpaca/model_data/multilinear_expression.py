# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var
import alpaca.settings as s


class MultilinearExpression:
    """Multilinear expression."""

    def __init__(
        self,
        name: str,
        model_data,
        variables: list[var.Variable],
        representative_variable=None,
    ):
        self.name = name
        self.variables = variables
        # pylint: disable=duplicate-code
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.variables.setdefault(
                f"r_{name}", var.Variable(f"r_{name}", lb=-s.StaticSettings.infinity)
            )
        )
        self.representative_variable.discretize_variable(
            model_data.settings.number_of_breakpoints
        )

    def apply_piecewise_linear_approximation(self):
        """Apply piecewise linear approximation."""

    def __repr__(self):
        return self.name
