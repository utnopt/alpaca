# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.model_data import variable as var, constraint as con


class BilinearBinaryExpression:
    """Represents a bilinear expression z = x * y where x and y are binary variables.

    Attributes:
        name: Identifier for the expression
        first_var: First variable (x) in the expression
        second_var: Second variable (y) in the expression
        representative_variable: Variable representing the product (z)
    """

    def __init__(
        self,
        name: str,
        model_data: "ModelData",
        variables: tuple[var.Variable, var.Variable],
        representative_variable: var.Variable | None = None,
    ):
        """Initialize bilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            variables: Tuple containing the two input variables (x, y)
            representative_variable: Optional existing variable to represent product
        """
        self.name = name
        self.first_var, self.second_var = variables
        self.model_data = model_data
        self.representative_variable = (
            representative_variable
            if representative_variable
            else model_data.add_variable(var.Variable(f"r_{name}"))
        )
        self.representative_variable.var_type = "B"
        self.representative_variable.lb = 0.0
        self.representative_variable.ub = 1.0
        self._add_mc_cormick_constraints()

    def _add_mc_cormick_constraints(self):
        """Add McCormick constraints for bilinear binary expressions."""
        self.model_data.add_constraint(
            con.Constraint(
                f"mc_cormick_{self.name}_1",
                con_type=">=",
                variables=[
                    (-1.0, self.first_var),
                    (-1.0, self.second_var),
                    (1.0, self.representative_variable),
                ],
                rhs=-1.0,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"mc_cormick_{self.name}_2",
                con_type=">=",
                variables=[(1.0, self.first_var), (-1.0, self.representative_variable)],
                rhs=0.0,
            )
        )
        self.model_data.add_constraint(
            con.Constraint(
                f"mc_cormick_{self.name}_3",
                con_type=">=",
                variables=[
                    (1.0, self.second_var),
                    (-1.0, self.representative_variable),
                ],
                rhs=0.0,
            )
        )
