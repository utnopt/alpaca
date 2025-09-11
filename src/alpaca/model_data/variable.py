# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses
import numpy as np

import alpaca.settings as s
import alpaca.model_data.constraint as con  # pylint: disable=cyclic-import


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
        self.pwl_variables_binary = []
        self.pwl_variables_continuous = []
        self.pwl_constraints = []
        self.implied_values_list: list[float] = []

    def add_nonlinearity_to_occurring_in(self, nonlinearity_type: str) -> None:
        """Save in which types of nonlinearities the variable occurs.
        To determine the optimal breakpoint locations.
        """
        if nonlinearity_type not in self.occurring_in:
            self.occurring_in.append(nonlinearity_type)
            self.is_discretized = True

    def add_binary_pwl(self, number_of_breakpoints: int, pwl_method: str) -> None:
        """Discretize the variable into breakpoints.

        Creates a set of points between the lower and upper bounds
        of the variable. These breakpoints are used for piecewise linear approximations
        of nonlinear functions involving this variable.

        Args:
            number_of_breakpoints: Number of discretization points to create.
            pwl_method: PWL method to use.
        """
        self.breakpoints = self._get_breakpoints(number_of_breakpoints)
        if pwl_method == "multiple-choice":
            self._add_binaries_multiple_choice()

    def _get_breakpoints(self, nr_of_breakpoints: int) -> list[float]:
        if self.ub == self.lb:
            return [self.lb]
        return np.linspace(self.lb, self.ub, nr_of_breakpoints).tolist()

    def add_continuous_pwl(self, pwl_method: str) -> None:
        """Link continuous variables to the breakpoints.

        Args:
            pwl_method: PWL method to use.
        """
        if pwl_method == "multiple-choice":
            self._add_pwl_approximation_multiple_choice()

    def _add_pwl_approximation_multiple_choice(self):
        self._add_continuous_variables_multiple_choice()
        self._add_pwl_constraints_multiple_choice()

    def _add_binaries_multiple_choice(self) -> None:
        for breakpoint_index in range(len(self.breakpoints) - 1):
            self.pwl_variables_binary.append(
                Variable(f"{self.name}_bp_{breakpoint_index}", var_type="B")
            )

    def _add_continuous_variables_multiple_choice(self) -> None:
        for breakpoint_index in range(len(self.breakpoints) - 1):
            self.pwl_variables_continuous.append(
                Variable(
                    f"{self.name}_c_{breakpoint_index}",
                    var_type="C",
                    lb=min(0.0, self.lb),
                    ub=max(0.0, self.ub),
                )
            )

    def _add_pwl_constraints_multiple_choice(self) -> None:
        self.pwl_constraints.append(
            con.Constraint(
                f"mc_varlink_cont_{self.name}",
                con_type="==",
                variables=[
                    (1.0, variable) for variable in self.pwl_variables_continuous
                ]
                + [(-1.0, self)],
            )
        )
        self.pwl_constraints.append(
            con.Constraint(
                f"mc_varlink_{self.name}",
                con_type="==",
                variables=[(1.0, variable) for variable in self.pwl_variables_binary],
                rhs=1.0,
            )
        )
        for breakpoint_index in range(len(self.breakpoints) - 1):
            self.pwl_constraints.append(
                con.Constraint(
                    f"mc_lb_{self.name}_{breakpoint_index}",
                    con_type="<=",
                    variables=[
                        (
                            self.breakpoints[breakpoint_index],
                            self.pwl_variables_binary[breakpoint_index],
                        ),
                        (-1.0, self.pwl_variables_continuous[breakpoint_index]),
                    ],
                )
            )
            self.pwl_constraints.append(
                con.Constraint(
                    f"mc_ub_{self.name}_{breakpoint_index}",
                    con_type=">=",
                    variables=[
                        (
                            self.breakpoints[breakpoint_index + 1],
                            self.pwl_variables_binary[breakpoint_index],
                        ),
                        (-1.0, self.pwl_variables_continuous[breakpoint_index]),
                    ],
                )
            )

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)
