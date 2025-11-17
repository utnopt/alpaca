# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from alpaca.utils.logger import logger
import alpaca.model_data.variable as var
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
import alpaca.breakpoints.breakpoint_neural_network as bnn

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class BreakpointGenerator:
    """Generates variable breakpoints if they get discretized."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data

    def generate_breakpoints(self) -> None:
        """Creates piecewise linear approximations for variables marked for discretization."""
        logger.info(lsf.info_generate_breakpoints())
        for variable in self.model_data.variables.values():
            if variable.is_discretized and variable.var_type != lsf.var_type_binary():
                variable.breakpoints = self._get_breakpoints_for_variable(variable)

    def _get_breakpoints_for_variable(self, variable: var.Variable) -> list[float]:
        if variable.ub == variable.lb:
            variable.is_discretized = False
            return []
        if (
            not variable.occurring_in
            or self.model_data.settings.breakpoint_generation == 0
        ):
            return np.linspace(
                variable.lb, variable.ub, self.model_data.settings.number_of_breakpoints
            ).tolist()
        return bnn.BreakpointNeuralNetwork(
            variable, self.model_data.settings
        ).breakpoints.tolist()
