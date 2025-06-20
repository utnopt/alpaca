# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses

from alpaca.model_data import variable as var


@dataclasses.dataclass
class Constraint:
    """Represents a mathematical constraint in an optimization model.

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
        self.rhs = rhs
        self.variables: list[tuple[float, var.Variable]] = (
            [] if variables is None else variables
        )

    def __repr__(self):
        return self.name
