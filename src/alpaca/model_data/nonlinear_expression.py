# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations

import alpaca.utils.datahandling as udh
from alpaca.model_data import (
    variable as var,
    bilinear_expression as ble,
    multilinear_expression as mle,
    one_dim_expression as ode,
    constraint as con,
)
from alpaca.utils.logger import logger
import alpaca.settings as s


class NonlinearExpression:
    """Nonlinear expression."""

    def __init__(self, name: str, expression_tag, model_data=None):
        self.name = name
        self.expression_type = expression_tag.name
        self.expression_tag = expression_tag
        self.model_data: "ModelData" | None = model_data
        self.child_expressions: list = []
        self.representative_variable = var.Variable(
            f"r_{self.name}", lb=-s.StaticSettings.infinity
        )
        if not model_data is None:
            model_data.variables[f"r_{self.name}"] = self.representative_variable
        self.fragmented = False

    def grow_expression_tree(self):
        """Grow expression tree."""
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

    def fragment_expression_tree_to_low_dimensional_functions(self):
        # pylint: disable=too-many-branches
        """Fragment expression tree to one dim, bilinear or multilinear functions."""
        if self.expression_type == "product":
            self._fragment_product_expression()
        elif self.expression_type == "sum":
            self._fragment_sum_expression()
        elif self.expression_type == "divide":
            logger.warning("Expression type divide not supported yet!")
        elif self.expression_type == "square":
            self._fragment_one_dim_expression(ode.SquareExpression)
        elif self.expression_type == "exp":
            self._fragment_one_dim_expression(ode.ExponentialExpression)
        elif self.expression_type == "ln":
            self._fragment_one_dim_expression(ode.LnExpression)
        elif self.expression_type == "sqrt":
            self._fragment_one_dim_expression(ode.SquareRootExpression)
        elif self.expression_type == "sin":
            self._fragment_one_dim_expression(ode.SinusExpression)
        elif self.expression_type == "cos":
            self._fragment_one_dim_expression(ode.CosinusExpression)
        elif self.expression_type == "log10":
            self._fragment_one_dim_expression(ode.LogExpression)
        elif self.expression_type == "tanh":
            self._fragment_one_dim_expression(ode.TangensHExpression)
        elif self.expression_type == "min":
            self._fragment_one_dim_expression(ode.MinExpression)
        elif self.expression_type == "inverse":
            self._fragment_one_dim_expression(ode.InverseExpression)
        elif self.expression_type == "power":
            logger.warning("Expression type power not supported yet!")
        elif self.expression_type == "xabsx":
            self._fragment_one_dim_expression(ode.AbsExpression)
        elif self.expression_type == "negate":
            self._fragment_negate_expression()
        else:
            raise KeyError(f"Expression type {self.expression_type} not supported yet!")
        for child_expression in self.child_expressions:
            if (
                isinstance(child_expression, NonlinearExpression)
                and not child_expression.fragmented
            ):
                child_expression.fragment_expression_tree_to_low_dimensional_functions()
                child_expression.fragmented = True

    def _fragment_product_expression(self):
        float_children, variable_children, nonlinear_children = (
            self._separate_children()
        )
        product_coeff = self._calculate_product_coeff(float_children, variable_children)
        variables_in_product = self._get_variables_in_product(
            variable_children, nonlinear_children
        )

        if product_coeff != 1.0:
            helper_variable = self._create_helper_variable_and_constraint(product_coeff)
            self._create_product_expression(variables_in_product, helper_variable)
        else:
            self._create_product_expression(
                variables_in_product, self.representative_variable
            )

    def _separate_children(self):
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
    def _calculate_product_coeff(float_children, variable_children):
        product_coeff = 1.0
        for coeff in float_children:
            product_coeff *= coeff
        for var_coeff, _ in variable_children:
            product_coeff *= var_coeff
        return product_coeff

    @staticmethod
    def _get_variables_in_product(variable_children, nonlinear_children):
        variables = [variable for _, variable in variable_children]
        variables.extend(child.representative_variable for child in nonlinear_children)
        return variables

    def _create_helper_variable_and_constraint(self, product_coeff):
        lb = self.representative_variable.lb / product_coeff
        ub = self.representative_variable.ub / product_coeff
        helper_var_lb, helper_var_ub = min(lb, ub), max(lb, ub)

        helper_variable = var.Variable(
            f"h_{self.name}", lb=helper_var_lb, ub=helper_var_ub
        )

        self.model_data.variables[f"h_{self.name}"] = helper_variable
        self.model_data.constraints[f"c_{self.name}"] = con.Constraint(
            f"c_{self.name}",
            variables=[
                (-1.0, self.representative_variable),
                (product_coeff, helper_variable),
            ],
            con_type="==",
        )
        return helper_variable

    def _create_product_expression(self, variables_in_product, representative_variable):
        num_vars = len(variables_in_product)
        if num_vars > 2:
            self._create_multilinear_expression(
                variables_in_product, representative_variable
            )
        elif num_vars == 2 and not variables_in_product[0] is variables_in_product[1]:
            self._create_bilinear_expression(
                variables_in_product, representative_variable
            )
        elif num_vars == 2 and variables_in_product[0] is variables_in_product[1]:
            self._create_square_expression(
                variables_in_product, representative_variable
            )
        else:
            self.model_data.constraints[f"cl_{self.name}"] = con.Constraint(
                f"cl_{self.name}",
                con_type="==",
                variables=[
                    (1.0, representative_variable),
                    (-1.0, variables_in_product[0]),
                ],
            )

    def _create_multilinear_expression(self, variables, representative_variable):
        expr_name = f"ml_{representative_variable.name}"
        self.model_data.multilinear_expressions[expr_name] = mle.MultilinearExpression(
            expr_name,
            self.model_data,
            variables,
            representative_variable=representative_variable,
        )

    def _create_bilinear_expression(self, variables, representative_variable):
        expr_name = f"bl_{representative_variable.name}"
        self.model_data.bilinear_expressions[expr_name] = ble.BilinearExpression(
            expr_name,
            self.model_data,
            variables,
            representative_variable=representative_variable,
        )

    def _create_square_expression(self, variables, representative_variable):
        expr_name = f"square_{representative_variable.name}"
        self.model_data.one_dim_expressions[expr_name] = ode.SquareExpression(
            expr_name,
            self.model_data,
            variables[0],
            representative_variable=representative_variable,
        )

    def _fragment_sum_expression(self):
        constraint = con.Constraint(
            f"c_{self.expression_type}_{self.name}",
            con_type="==",
        )
        self.model_data.constraints[f"c_{self.expression_type}_{self.name}"] = (
            constraint
        )
        constraint.variables.append((-1.0, self.representative_variable))
        for child_expression in self.child_expressions:
            if isinstance(child_expression, tuple):
                child_expression: tuple[float, var.Variable]
                constraint.variables.append(child_expression)
            elif isinstance(child_expression, float):
                constraint.rhs -= child_expression
            else:
                child_expression: NonlinearExpression
                constraint.variables.append(
                    (1.0, child_expression.representative_variable)
                )

    def _fragment_one_dim_expression(self, expression_class):
        if len(self.child_expressions) == 2:
            self._fragment_one_dim_expression_with_coefficient(expression_class)
        elif len(self.child_expressions) == 1:
            self._fragment_one_dim_expression_without_coefficient(expression_class)

    def _fragment_one_dim_expression_with_coefficient(self, expression_class):
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
        helper_var_lb, helper_var_ub = tuple(
            sorted([variable.lb * coeff, variable.ub * coeff])
        )
        helper_variable = var.Variable(
            f"h_{self.name}", lb=helper_var_lb, ub=helper_var_ub
        )
        self.model_data.variables[f"h_{self.name}"] = helper_variable
        self.model_data.constraints[f"c_{self.name}"] = con.Constraint(
            f"c_{self.name}",
            variables=[(coeff, variable), (-1.0, helper_variable)],
            con_type="==",
        )
        self.model_data.one_dim_expressions[
            f"{self.expression_type}_{helper_variable.name}"
        ] = expression_class(
            f"{self.expression_type}_{helper_variable.name}",
            self.model_data,
            helper_variable,
            representative_variable=self.representative_variable,
        )

    def _fragment_one_dim_expression_without_coefficient(self, expression_class):
        variable = (
            self.child_expressions[0][1]
            if isinstance(self.child_expressions[0], tuple)
            else self.child_expressions[0].representative_variable
        )
        self.model_data.one_dim_expressions[
            f"{self.expression_type}_{variable.name}"
        ] = expression_class(
            f"{self.expression_type}_{variable.name}",
            self.model_data,
            variable,
            representative_variable=self.representative_variable,
        )

    def _fragment_negate_expression(self):
        self.model_data.constraints[f"c_{self.name}"] = con.Constraint(
            f"c_{self.name}",
            variables=[
                (1.0, self.representative_variable),
                (1.0, self.child_expressions[0].representative_variable),
            ],
            con_type="==",
        )

    def _add_nonlinear_expression_child(
        self, child_expression_tag_name, child_expression_tag
    ):
        if child_expression_tag_name in self.model_data.nonlinear_expressions:
            self.child_expressions.append(
                self.model_data.nonlinear_expressions[child_expression_tag_name]
            )
            return
        child_expression = NonlinearExpression(
            child_expression_tag_name, child_expression_tag, self.model_data
        )
        self.model_data.nonlinear_expressions[child_expression_tag_name] = (
            child_expression
        )
        self.child_expressions.append(child_expression)
        child_expression.grow_expression_tree()

    def __repr__(self):
        return self.expression_type + "_" + self.name
