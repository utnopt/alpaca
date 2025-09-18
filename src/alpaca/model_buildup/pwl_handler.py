# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from alpaca.utils.logger import logger
from alpaca.pwl import multiple_choice_method as mcm, delta_method as dem
import alpaca.expressions.one_dim_expression as ode
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class PWLHandler:
    """Applies Piecewise Linear relaxations to expressions."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data

    def apply_relaxations(self):
        """Applies all PWL relaxations based on model settings."""
        logger.info(lsf.info_create_piecewise_linear_relaxation())
        self._discretize_variable_domains()
        self._apply_piecewise_linear_relaxation_for_one_dim_expressions()
        self._apply_piecewise_constant_relaxation_for_multilinear_expressions()

    def _discretize_variable_domains(self) -> None:
        pwl_variables = []
        pwl_constraints = []
        for variable in self.model_data.variables.values():
            if variable.is_discretized and variable.var_type != lsf.var_type_binary():
                variable.pwl = (
                    mcm.MultipleChoiceMethod(variable)
                    if self.model_data.settings.pwl_method
                    == lsf.pwl_method_multiple_choice()
                    else dem.DeltaMethod(variable)
                )
                pwl_variables.extend(
                    variable.pwl.pwl_variables_binary
                    + variable.pwl.pwl_variables_continuous
                )
                pwl_constraints.extend(variable.pwl.pwl_constraints)
        for variable in pwl_variables:
            self.model_data.add_variable(variable)
        for constraint in pwl_constraints:
            self.model_data.add_constraint(constraint)

    def _apply_piecewise_linear_relaxation_for_one_dim_expressions(self) -> None:
        """Applies piecewise linear relaxation to all one-dimensional expressions."""
        for expression in self.model_data.expressions.one_dim_expressions.values():
            if isinstance(expression, ode.AbsExpression):
                expression.handle_abs_expression(self.model_data)
                continue
            constraints = (
                expression.variable.pwl.couple_domain_to_function_value_one_dim(
                    expression, approximation=self.model_data.settings.approximation
                )
            )
            for constraint in constraints:
                self.model_data.add_constraint(constraint)

    def _apply_piecewise_constant_relaxation_for_multilinear_expressions(
        self,
    ) -> None:
        """Applies piecewise constant relaxation to bilinear and multilinear expressions."""
        if self.model_data.settings.bilinear_handling == 2:
            for expression in self.model_data.expressions.bilinear_expressions.values():
                constraints = (
                    expression.representative_variable.pwl.apply_pwc_relaxation(
                        expression,
                        approximation=self.model_data.settings.approximation,
                    )
                )
                for constraint in constraints:
                    self.model_data.add_constraint(constraint)
        if not self.model_data.settings.reformulate_multilinear_to_bilinear:
            for (
                expression
            ) in self.model_data.expressions.multilinear_expressions.values():
                constraints = (
                    expression.representative_variable.pwl.apply_pwc_relaxation(
                        expression,
                        approximation=self.model_data.settings.approximation,
                    )
                )
                for constraint in constraints:
                    self.model_data.add_constraint(constraint)
