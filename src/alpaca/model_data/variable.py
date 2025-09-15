# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import dataclasses

import alpaca.settings as s

if TYPE_CHECKING:
    from alpaca.pwl import pwl_method as pwm


@dataclasses.dataclass
class Variable:  # pylint: disable=too-many-instance-attributes
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
        lb: float = -s.StaticSettings.infinity,
        ub: float = s.StaticSettings.infinity,
        var_type: str = "C",
    ):
        """Initialize a Variable instance.

        Args:
            name: Unique identifier for the variable.
            lb: Lower bound of the variable's domain. Defaults to 0.
            ub: Upper bound of the variable's domain. Defaults to infinity.
            var_type: Variable type. Defaults to 'C'. Binary: 'B'
        """
        self.name = name
        self.lb = lb
        self.ub = ub
        self.var_type = var_type
        self.solver_variable = None
        self.is_discretized = False
        self.breakpoints: list[float] = []
        self.occurring_in: list[str] = []
        self.pwl: pwm.PWLMethod | None = None

    def add_nonlinearity_to_occurring_in(self, nonlinearity_type: str) -> None:
        """Save in which types of nonlinearities the variable occurs.
        To determine the optimal breakpoint locations.
        """
        if nonlinearity_type not in self.occurring_in:
            self.occurring_in.append(nonlinearity_type)
            self.is_discretized = True

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)
