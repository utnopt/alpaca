# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import alpaca.utils.datahandling as udh
from alpaca.model_data import variable as var
from alpaca.expressions import (
    one_dim_expression as ode,
)
from alpaca.utils.logger import logger
import alpaca.settings as s

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class NonlinearExpression:
    """Represents a general nonlinear expression in an optimization model.

    This class handles the parsing, representation, and transformation of nonlinear expressions
    into piecewise linear approximations that can be solved by linear solvers. It recursively
    processes expression trees and decomposes them into simpler one-dimensional, bilinear, or
    multilinear expressions.

    Attributes:
        name: Unique identifier for the expression
        expression_type: Type of the nonlinear operation (e.g., 'sum', 'product', 'exp')
        expression_tag: Original expression tag from the OSiL format
        model_data: Reference to the containing model data object
        child_expressions: List of child expressions in the expression tree
        representative_variable: Variable representing the result of the expression
        fragmented: Flag indicating if the expression has been decomposed
    """

    def __init__(self, name: str, expression_tag, model_data: ModelData):
        """Initialize a nonlinear expression with its XML tag and optional model data.

        Args:
            name: Unique identifier for the expression
            expression_tag: XML tag containing the expression from OSiL format
            model_data: Reference to the containing model data object
        """
        self.name = name
        self.expression_type = expression_tag.name
        self.expression_tag = expression_tag
        self.model_data = model_data
        self.child_expressions: list = []
        self.representative_variable = model_data.add_variable(
            var.Variable(f"r_{self.name}", lb=-s.StaticSettings.infinity)
        )
        self.fragmented = False

    def grow_expression_tree(self) -> None:
        """Build the expression tree recursively from OSiL XML tags.

        Parses the expression_tag's contents and creates child nodes for variables, numbers,
        and nested nonlinear expressions. Each child is processed and added to the
        child_expressions list.
        """
        for child_expression_tag in self.expression_tag.contents:
            child_expression_tag_name = udh.hash_nonlinearity(str(child_expression_tag))
            if child_expression_tag.name == "variable":
                coeff = (
                    1.0
                    if child_expression_tag.get("coef") is None
                    else float(child_expression_tag.get("coef"))
                )
                variable = self.model_data.variables[
                    f"x_{child_expression_tag.get('idx')}"
                ]
                self.child_expressions.append((coeff, variable))
            elif child_expression_tag.name == "number":
                self.child_expressions.append(float(child_expression_tag.get("value")))
            else:
                self._add_nonlinear_expression_child(
                    child_expression_tag_name, child_expression_tag
                )

    def fragment_expression_tree_to_low_dimensional_functions(self, level: int) -> None:
        # pylint: disable=too-many-branches
        """Decompose complex expressions into simple low-dimensional functions.

        Based on the expression type, this method breaks down the nonlinear expression into
        simpler one-dimensional, bilinear, or multilinear expressions that can be approximated
        using piecewise linear functions. It handles various expression types such as products,
        sums, exponentials, logarithms, trigonometric functions, etc.

        The method recursively processes child expressions and marks them as fragmented.
        """
        if self.expression_type == "product":
            next_level = self._fragment_product_expression(level)
        elif self.expression_type == "sum":
            next_level = self._fragment_sum_expression(level)
        elif self.expression_type == "divide":
            next_level = self._fragment_division_expression(level)
        elif self.expression_type == "square":
            next_level = self._fragment_one_dim_expression(ode.SquareExpression, level)
        elif self.expression_type == "exp":
            next_level = self._fragment_one_dim_expression(
                ode.ExponentialExpression, level
            )
        elif self.expression_type == "ln":
            next_level = self._fragment_one_dim_expression(ode.LnExpression, level)
        elif self.expression_type == "sqrt":
            next_level = self._fragment_one_dim_expression(
                ode.SquareRootExpression, level
            )
        elif self.expression_type == "sin":
            next_level = self._fragment_one_dim_expression(ode.SineExpression, level)
        elif self.expression_type == "cos":
            next_level = self._fragment_one_dim_expression(ode.CosineExpression, level)
        elif self.expression_type == "log10":
            next_level = self._fragment_one_dim_expression(ode.LogExpression, level)
        elif self.expression_type == "tanh":
            next_level = self._fragment_one_dim_expression(
                ode.TangensHExpression, level
            )
        elif self.expression_type == "min":
            next_level = logger.warning("Expression type min not supported yet!")
        elif self.expression_type == "inverse":
            next_level = self._fragment_one_dim_expression(ode.InverseExpression, level)
        elif self.expression_type == "power":
            if self.child_expressions[1] == 2.0:
                next_level = self._fragment_one_dim_expression(
                    ode.SquareExpression, level
                )
            elif self.child_expressions[1] == 0.5:
                next_level = self._fragment_one_dim_expression(
                    ode.SquareRootExpression, level
                )
            else:
                next_level = logger.warning("Expression type power not supported yet!")
        elif self.expression_type == "xabsx":
            next_level = self._fragment_one_dim_expression(ode.AbsExpression, level)
        elif self.expression_type == "negate":
            next_level = self._fragment_negate_expression(level)
        else:
            raise KeyError(f"Expression type {self.expression_type} not supported yet!")
        for child_expression in self.child_expressions:
            if (
                isinstance(child_expression, NonlinearExpression)
                and not child_expression.fragmented
            ):
                child_expression.fragment_expression_tree_to_low_dimensional_functions(
                    next_level
                )
                child_expression.fragmented = True

    def _fragment_product_expression(self, level: int) -> int:
        float_children, variable_children, nonlinear_children = (
            self._classify_children()
        )
        product_coeff = self._calculate_product_coeff(float_children, variable_children)
        variables_in_product = self._get_variables_in_product(
            variable_children, nonlinear_children
        )

        if product_coeff != 1.0:
            if len(variables_in_product) == 1:
                lin_expression = self.model_data.add_linear_expression(
                    f"le_{self.name}",
                    level,
                    representative_variable=self.representative_variable,
                )
                lin_expression.variables = [(product_coeff, variables_in_product[0])]
            else:
                helper_variable = self._create_coeff_intermediate_expression(
                    product_coeff, level
                )
                level += 1
                self._create_product_expression(
                    variables_in_product, helper_variable, level
                )
        else:
            self._create_product_expression(
                variables_in_product, self.representative_variable, level
            )
        return level + 1

    def _classify_children(
        self,
    ) -> tuple[
        list[float], list[tuple[float, var.Variable]], list[NonlinearExpression]
    ]:
        float_children = [
            child for child in self.child_expressions if isinstance(child, float)
        ]
        variable_children = [
            child for child in self.child_expressions if isinstance(child, tuple)
        ]
        nonlinear_children = [
            child
            for child in self.child_expressions
            if isinstance(child, NonlinearExpression)
        ]
        return float_children, variable_children, nonlinear_children

    @staticmethod
    def _calculate_product_coeff(
        float_children: list[float], variable_children: list[tuple[float, var.Variable]]
    ) -> float:
        product_coeff = 1.0
        for coeff in float_children:
            product_coeff *= coeff
        for var_coeff, _ in variable_children:
            product_coeff *= var_coeff
        return product_coeff

    @staticmethod
    def _get_variables_in_product(
        variable_children: list[tuple[float, var.Variable]],
        nonlinear_children: list[NonlinearExpression],
    ) -> list[var.Variable]:
        variables = [variable for _, variable in variable_children]
        variables.extend(child.representative_variable for child in nonlinear_children)
        return variables

    def _create_coeff_intermediate_expression(
        self, product_coeff: float, level: int
    ) -> var.Variable:
        helper_variable = self.model_data.add_variable(var.Variable(f"h_{self.name}"))
        lin_expression = self.model_data.add_linear_expression(
            f"le_h_{self.name}",
            level,
            representative_variable=self.representative_variable,
        )
        lin_expression.variables = [(product_coeff, helper_variable)]
        return helper_variable

    def _create_product_expression(
        self,
        variables_in_product: list[var.Variable],
        representative_variable: var.Variable,
        level: int,
    ) -> None:
        num_vars = len(variables_in_product)
        if num_vars > 2:
            self.model_data.add_multilinear_expression(
                f"ml_{representative_variable.name}",
                variables_in_product,
                level,
                representative_variable=representative_variable,
            )
        elif num_vars == 2 and not variables_in_product[0] is variables_in_product[1]:
            self.model_data.add_bilinear_expression(
                f"bl_{representative_variable.name}",
                tuple(variables_in_product),
                level,
                representative_variable=representative_variable,
            )
        elif num_vars == 2 and variables_in_product[0] is variables_in_product[1]:
            self._create_square_expression(
                variables_in_product, representative_variable, level
            )
        else:
            raise AssertionError("Product containing 1 variable not allowed!")

    def _create_square_expression(
        self,
        variables: list[var.Variable],
        representative_variable: var.Variable,
        level: int,
    ) -> None:
        expr_name = f"square_{representative_variable.name}"
        self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            expr_name,
            variables[0],
            level,
            representative_variable=representative_variable,
        )

    def _fragment_sum_expression(self, level: int) -> int:
        lin_expression = self.model_data.add_linear_expression(
            f"le_{self.expression_type}_{self.name}",
            level,
            representative_variable=self.representative_variable,
        )
        for child_expression in self.child_expressions:
            if isinstance(child_expression, tuple):
                child_expression: tuple[float, var.Variable]
                lin_expression.variables.append(child_expression)
            elif isinstance(child_expression, float):
                lin_expression.constant += child_expression
            else:
                child_expression: NonlinearExpression
                lin_expression.variables.append(
                    (1.0, child_expression.representative_variable)
                )
        return level + 1

    def _fragment_one_dim_expression(self, expression_class: type, level: int) -> int:
        if len(self.child_expressions) == 2:
            self._fragment_one_dim_expression_with_coefficient(expression_class, level)
            return level + 2
        self._fragment_one_dim_expression_without_coefficient(expression_class, level)
        return level + 1

    def _fragment_one_dim_expression_with_coefficient(
        self, expression_class: type, level: int
    ) -> None:
        if isinstance(self.child_expressions[0], float):
            coeff = self.child_expressions[0]
            variable = (
                self.child_expressions[1][1]
                if isinstance(self.child_expressions[1], tuple)
                else self.child_expressions[1].representative_variable
            )

        else:
            coeff = self.child_expressions[1]
            variable = (
                self.child_expressions[0][1]
                if isinstance(self.child_expressions[0], tuple)
                else self.child_expressions[0].representative_variable
            )
        helper_variable = self.model_data.add_variable(var.Variable(f"h_{self.name}"))
        lin_expression = self.model_data.add_linear_expression(
            f"le_{self.name}",
            level + 1,
            representative_variable=helper_variable,
        )
        lin_expression.variables.append((coeff, variable))
        self.model_data.add_one_dim_expression(
            expression_class,
            f"{self.expression_type}_{helper_variable.name}",
            helper_variable,
            level,
            representative_variable=self.representative_variable,
        )

    def _fragment_one_dim_expression_without_coefficient(
        self, expression_class: type, level: int
    ) -> None:
        variable = (
            self.child_expressions[0][1]
            if isinstance(self.child_expressions[0], tuple)
            else self.child_expressions[0].representative_variable
        )
        self.model_data.add_one_dim_expression(
            expression_class,
            f"{self.expression_type}_{variable.name}",
            variable,
            level,
            representative_variable=self.representative_variable,
        )

    def _fragment_negate_expression(self, level: int) -> int:
        lin_expression = self.model_data.add_linear_expression(
            f"le_{self.name}",
            level,
            representative_variable=self.representative_variable,
        )
        lin_expression.variables.append(
            (-1.0, self.child_expressions[0].representative_variable)
        )
        return level + 1

    def _fragment_division_expression(self, level: int) -> int:
        denominator_variable = (
            self.child_expressions[1][1]
            if isinstance(self.child_expressions[1], tuple)
            else self.child_expressions[1].representative_variable
        )
        denominator_coeff = (
            1.0
            if not isinstance(self.child_expressions[1], tuple)
            else 1 / self.child_expressions[1][0]
        )
        helper_inverse_expression = self.model_data.add_one_dim_expression(
            ode.InverseExpression, f"iv_{self.name}", denominator_variable, 0
        )
        if isinstance(self.child_expressions[0], float):
            self._handle_division_float_numerator(
                helper_inverse_expression.representative_variable,
                denominator_coeff,
                level,
            )
            helper_inverse_expression.level = level + 1
            return level + 2
        level = self._handle_division_nonlinear_numerator(
            helper_inverse_expression.representative_variable,
            denominator_coeff,
            level,
        )
        helper_inverse_expression.level = level
        return level + 1

    def _handle_division_float_numerator(
        self, variable: var.Variable, denominator_coeff: float, level: int
    ):
        lin_expression = self.model_data.add_linear_expression(
            f"le_{self.name}",
            level,
            representative_variable=self.representative_variable,
        )
        lin_expression.variables.append(
            (denominator_coeff * self.child_expressions[0], variable)
        )

    def _handle_division_nonlinear_numerator(
        self, variable: var.Variable, denominator_coeff: float, level: int
    ) -> int:
        numerator_variable = (
            self.child_expressions[0][1]
            if isinstance(self.child_expressions[0], tuple)
            else self.child_expressions[0].representative_variable
        )
        product_coeff = (
            denominator_coeff
            if not isinstance(self.child_expressions[0], tuple)
            else denominator_coeff * self.child_expressions[0][0]
        )
        if product_coeff != 1.0:
            helper_variable = self._create_coeff_intermediate_expression(
                product_coeff, level
            )
            level += 1
            self._create_product_expression(
                [numerator_variable, variable], helper_variable, level
            )
        else:
            self._create_product_expression(
                [numerator_variable, variable], self.representative_variable, level
            )
        return level + 1

    def _add_nonlinear_expression_child(
        self, child_expression_tag_name: str, child_expression_tag
    ) -> None:
        if (
            child_expression_tag_name
            in self.model_data.expressions.nonlinear_expressions
        ):
            self.child_expressions.append(
                self.model_data.expressions.nonlinear_expressions[
                    child_expression_tag_name
                ]
            )
            return
        child_expression = self.model_data.add_nonlinear_expression(
            child_expression_tag_name, child_expression_tag
        )
        self.child_expressions.append(child_expression)
        child_expression.grow_expression_tree()

    def __repr__(self) -> str:
        return self.expression_type + "_" + self.name
