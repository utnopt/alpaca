# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses
import numpy as np

import alpaca.settings as s


@dataclasses.dataclass
class Variable:
    """Represents a variable in an optimization model.

    This class encapsulates a decision variable with attributes such as bounds and type.
    Variables can be discretized into breakpoints for piecewise linear approximations.

    Attributes:
        name: Unique identifier for the variable.
        lb: Lower bound of the variable's domain. Defaults to 0.
        ub: Upper bound of the variable's domain. Defaults to infinity.
        var_type: Variable type. Defaults to 'C'.
        is_discretized: Flag indicating if the variable has been discretized.
        breakpoints: Discretization points within the variable's domain.
    """

    def __init__(
        self,
        name: str,
        lb: float = 0,
        ub: float = s.StaticSettings.infinity,
        var_type: str = "C",
    ):
        """Initialize a Variable instance.

        Args:
            name: Unique identifier for the variable.
            lb: Lower bound of the variable's domain. Defaults to 0.
            ub: Upper bound of the variable's domain. Defaults to infinity.
            var_type: Variable type. Defaults to 'C'.
        """
        self.name = name
        self.lb = lb
        self.ub = ub
        self.var_type = var_type
        self.is_discretized = False
        self.breakpoints: list[float] = []

    def discretize_variable(self, number_of_breakpoints: int) -> None:
        """Discretize the variable into equally spaced breakpoints.

        Creates a set of equally spaced points between the lower and upper bounds
        of the variable. These breakpoints are used for piecewise linear approximations
        of nonlinear functions involving this variable.

        Args:
            number_of_breakpoints: Number of discretization points to create.
        """
        if self.is_discretized:
            return
        self.breakpoints = np.linspace(self.lb, self.ub, number_of_breakpoints)
        self.is_discretized = True

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)
