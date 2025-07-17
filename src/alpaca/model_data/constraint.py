# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var


class Constraint:
    """Represents a linear constraint in an optimization model.

    Constraints are defined by their name, type (e.g., inequality/equality),
    right-hand side value, and a collection of variables with associated coefficients.

    Attributes:
        name: Identifier for the constraint.
        con_type: Constraint type ('<=', '==', '>='). Defaults to '<='.
        rhs: Right-hand side constant term. Defaults to 0.0.
        variables: Coefficients paired with variables in the constraint expression.
    """

    def __init__(self, name: str, con_type="<=", rhs=0.0, variables=None):
        """Initializes a Constraint instance.

        Args:
            name: Identifier for the constraint.
            con_type: Constraint type ('<=', '==', '>='). Defaults to '<='.
            rhs: Right-hand side constant term. Defaults to 0.0.
            variables: Coefficients paired with variables. None initializes an empty list.
        """
        self.name = name
        self.con_type = con_type
        self.solver_constraint = None
        self.rhs = rhs
        self.variables: list[tuple[float, var.Variable]] = (
            [] if variables is None else variables
        )

    def propagate_variable_bounds(self):
        """Propagate bounds of variables based on the constraint."""
        total_min = total_max = 0
        for a_j, x_j in self.variables:
            if a_j > 0:
                total_min += a_j * x_j.lb
                total_max += a_j * x_j.ub
            else:
                total_min += a_j * x_j.ub
                total_max += a_j * x_j.lb

        for a_i, x_i in self.variables:
            # Remove x_i's contribution to get sum of other variables
            if a_i > 0:
                other_min = total_min - a_i * x_i.lb
                other_max = total_max - a_i * x_i.ub
            else:
                other_min = total_min - a_i * x_i.ub
                other_max = total_max - a_i * x_i.lb

            # Update bounds for x_i
            if self.con_type in ("<=", "=="):
                if a_i > 0:
                    x_i.ub = min(x_i.ub, (self.rhs - other_min) / a_i)
                else:  # a_i < 0
                    x_i.lb = max(x_i.lb, (self.rhs - other_min) / a_i)
            if self.con_type in (">=", "=="):
                if a_i > 0:
                    x_i.lb = max(x_i.lb, (self.rhs - other_max) / a_i)
                else:  # a_i < 0
                    x_i.ub = min(x_i.ub, (self.rhs - other_max) / a_i)

    def __repr__(self):
        return self.name
