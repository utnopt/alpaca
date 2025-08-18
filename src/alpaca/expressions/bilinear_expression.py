# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import bisect

from alpaca.model_data import variable as var, constraint as con
from alpaca.expressions import (
    expression as exn,
    one_dim_expression as ode,
)


class BilinearExpression(exn.Expression):
    """Represents a bilinear expression z = x * y.

    Attributes:
        name: Identifier for the expression
        first_var: First variable (x) in the expression
        second_var: Second variable (y) in the expression
        level: Level of expression in expression tree
        representative_variable: Variable representing the product (z)
    """

    def __init__(
        self,
        name: str,
        model_data,
        variables: tuple[var.Variable, var.Variable],
        level: int,
        representative_variable: var.Variable | None = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Initialize bilinear expression.

        Args:
            name: Expression identifier
            model_data: Container for model components
            variables: Tuple containing the two input variables (x, y)
            level: Level of expression in expression tree
            representative_variable: Optional existing variable to represent product
        """
        super().__init__(name, model_data, level, representative_variable)
        self.first_var, self.second_var = variables
        self.model_data = model_data
        self.representative_variable.add_nonlinearity_to_occurring_in("bilinear")
        self.first_var.add_nonlinearity_to_occurring_in("bilinear")
        self.second_var.add_nonlinearity_to_occurring_in("bilinear")
        self.piecewise_constant_relation = {}

    def apply_piecewise_constant_relaxation(
        self, approximation: bool = False, reformulated: int = 0
    ):
        """Apply piecewise constant relaxation for bilinear expressions."""
        first_segment_midpoints = self._compute_segment_midpoints(
            self.first_var.breakpoints
        )
        second_segment_midpoints = self._compute_segment_midpoints(
            self.second_var.breakpoints
        )

        for i, first_mid in enumerate(first_segment_midpoints):
            for j, second_mid in enumerate(second_segment_midpoints):
                if approximation:
                    implied_indices = self._compute_approximate_product_indices(
                        first_mid, second_mid
                    )
                else:
                    implied_indices = self._compute_exact_product_indices(i, j)

                self._store_bilinear_relation(i, j, implied_indices)
                if not reformulated:
                    self._add_bilinear_constraint(i, j, implied_indices)

    @staticmethod
    def _compute_segment_midpoints(breakpoints):
        return [(breakpoints[i + 1] + bp) / 2 for i, bp in enumerate(breakpoints[:-1])]

    def _compute_approximate_product_indices(self, first_mid, second_mid):
        product_value = first_mid * second_mid
        product_index = self._get_implied_index_from_implied_value(product_value)
        return (product_index,)

    def _compute_exact_product_indices(self, first_seg_index, second_seg_index):
        lb_first = self.first_var.breakpoints[first_seg_index]
        ub_first = self.first_var.breakpoints[first_seg_index + 1]
        lb_second = self.second_var.breakpoints[second_seg_index]
        ub_second = self.second_var.breakpoints[second_seg_index + 1]
        min_product, max_product = self._get_implied_lb_and_ub(
            lb_first, ub_first, lb_second, ub_second
        )
        min_index = self._get_implied_index_from_implied_value(min_product)
        max_index = self._get_implied_index_from_implied_value(max_product)
        return tuple(range(min_index, max_index + 1))

    def _store_bilinear_relation(
        self, first_seg_index, second_seg_index, product_indices
    ):
        self.piecewise_constant_relation[(first_seg_index, second_seg_index)] = (
            product_indices
        )

    def _add_bilinear_constraint(
        self, first_seg_index, second_seg_index, product_indices
    ):
        constraint_vars = [
            (-1.0, self.representative_variable.pwl_variables_binary[idx])
            for idx in product_indices
        ]
        constraint_vars += [
            (1.0, self.first_var.pwl_variables_binary[first_seg_index]),
            (1.0, self.second_var.pwl_variables_binary[second_seg_index]),
        ]

        constraint = con.Constraint(
            name=f"bilinear_rel_{self.name}_{first_seg_index}_{second_seg_index}",
            con_type="<=",
            variables=constraint_vars,
            rhs=1.0,
        )
        self.model_data.add_constraint(constraint)

    def _get_implied_index_from_implied_value(self, implied_value: float) -> int:
        return min(
            bisect.bisect_left(self.representative_variable.breakpoints, implied_value)
            - 1,
            len(self.representative_variable.breakpoints) - 2,
        )

    def reformulate_to_sum_of_squares(self):
        """Reformulate bilinear expression to sum of squares.
        xy = 0.5 (x² + y² − p²), p = x - y."""
        master_linear_expression = self.model_data.add_linear_expression(
            f"le_{self.name}_master",
            self.level,
            representative_variable=self.representative_variable,
        )
        square_first_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            f"fvs_{self.name}",
            self.first_var,
            self.level + 1,
        )
        square_second_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            f"svs_{self.name}",
            self.second_var,
            self.level + 1,
        )
        sub_linear_expression = self.model_data.add_linear_expression(
            f"le_{self.name}_sub",
            self.level + 2,
        )
        sub_linear_expression.variables = [
            (1.0, self.first_var),
            (-1.0, self.second_var),
        ]
        square_helper_var = self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            f"hvs_{self.name}",
            sub_linear_expression.representative_variable,
            self.level + 1,
        )
        master_linear_expression.variables = [
            (0.5, square_first_var.representative_variable),
            (0.5, square_second_var.representative_variable),
            (-0.5, square_helper_var.representative_variable),
        ]

    def propagate_variable_bounds(self):
        """Propagate variable bounds for bilinear expression."""
        implied_lb, implied_ub = self._get_implied_lb_and_ub(
            self.first_var.lb, self.first_var.ub, self.second_var.lb, self.second_var.ub
        )
        self.representative_variable.lb = max(
            self.representative_variable.lb, implied_lb
        )
        self.representative_variable.ub = min(
            self.representative_variable.ub, implied_ub
        )

    @staticmethod
    def _get_implied_lb_and_ub(
        first_lb: float, first_ub: float, second_lb: float, second_ub: float
    ) -> tuple[float, float]:
        p1 = first_lb * second_lb
        p2 = first_lb * second_ub
        p3 = first_ub * second_lb
        p4 = first_ub * second_ub
        return min(p1, p2, p3, p4), max(p1, p2, p3, p4)
