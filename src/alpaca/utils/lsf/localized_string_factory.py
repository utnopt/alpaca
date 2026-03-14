# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.utils.lsf.general import General
from alpaca.utils.lsf.math_model import MathModel
from alpaca.utils.lsf.messages import Messages
from alpaca.utils.lsf.external_solvers import ExternalSolvers
from alpaca.utils.lsf.reformulations import Reformulations
from alpaca.utils.lsf.model_reading import ModelReading
from alpaca.utils.lsf.statistics import Statistics


class LocalizedStringFactory(
    General,
    MathModel,
    Messages,
    ExternalSolvers,
    Reformulations,
    ModelReading,
    Statistics,
):
    """
    Localized string factory.
    This class provides static methods to retrieve standardized strings used
    throughout the application, such as error messages, variable names,
    solver parameters, and more.
    """
