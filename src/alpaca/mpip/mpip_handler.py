# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pyscipopt as scip

from alpaca.utils.logger import logger
import alpaca.external_solvers.solver_wrapper as sw
import alpaca.model_data.model_data as mda
from alpaca.expressions import (
    nonlinear_expression as nle,
    bilinear_expression as ble,
    multilinear_expression as mle,
)
import alpaca.mpip.mpip as mp
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class MPIPHandler:  # pylint: disable=too-many-instance-attributes
    """Multipartite Implication Polytope handler."""

    def __init__(self, model_data: mda.ModelData) -> None:
        """Initialize MPIP instance."""
        logger.info(lsf.info_init_mpip_handler())
        self.model_data = model_data
        self.first_level_nonlinear_expression_values = [
            model_data.expressions.nonlinear_expressions[expr_key]
            for expr_key in model_data.expressions.first_level_nonlinear_expression_keys
        ]
        self.bilinear_expressions = model_data.expressions.bilinear_expressions
        self.multilinear_expressions = model_data.expressions.multilinear_expressions
        self.mpip_dict: dict[str, mp.MPIP] = {}
        self.mpip_counter: int = 0
        self._extract_mpip_instances_in_nonlinear_expressions()
        self._find_mpip_instances_in_multilinear_and_bilinear_expressions()
        self._build_mpip_instances()

    def _build_mpip_instances(self) -> None:
        for mpip in self.mpip_dict.values():
            if not mpip.relation:
                mpip.build_mpip()

    def _extract_mpip_instances_in_nonlinear_expressions(self) -> None:
        """Extract mpip instances from nonlinear expression trees."""
        for nonlinear_expression in self.first_level_nonlinear_expression_values:
            self._process_expression_tree(nonlinear_expression)

    def _find_mpip_instances_in_multilinear_and_bilinear_expressions(self) -> None:
        """Extract mpip instances from multilinear and bilinear expressions."""
        for multilinear_expression in list(
            self.multilinear_expressions.values()
        ) + list(self.bilinear_expressions.values()):
            if multilinear_expression.representative_variable.is_discretized:
                multilinear_expression.extract_mpip_relation(
                    approximation=self.model_data.settings.approximation
                )
                self._add_mpip_instance_from_multilinear_expression(
                    multilinear_expression
                )

    def _add_mpip_instance_from_bilinear_expression(
        self, bilinear_expression: ble.BilinearExpression
    ) -> None:
        self.mpip_counter += 1
        mpip_id = lsf.mpip_id(self.mpip_counter)
        mpip = mp.MPIP(mpip_id, self.model_data.settings.pwl_method)
        representative_variable = bilinear_expression.representative_variable
        mpip.add_implied_id(representative_variable)
        first_var = bilinear_expression.variables[0]
        mpip.add_implying_id(first_var)
        second_var = bilinear_expression.variables[1]
        mpip.add_implying_id(second_var)
        mpip.relation = bilinear_expression.piecewise_constant_relation
        self.mpip_dict[mpip_id] = mpip

    def _add_mpip_instance_from_multilinear_expression(
        self, multilinear_expression: mle.MultilinearExpression
    ) -> None:
        self.mpip_counter += 1
        mpip_id = lsf.mpip_id(self.mpip_counter)
        mpip = mp.MPIP(mpip_id, self.model_data.settings.pwl_method)
        representative_variable = multilinear_expression.representative_variable
        mpip.add_implied_id(representative_variable)
        for var in multilinear_expression.variables:
            mpip.add_implying_id(var)
        mpip.relation = multilinear_expression.piecewise_constant_relation
        self.mpip_dict[mpip_id] = mpip

    def _process_expression_tree(
        self, nonlinear_expression: nle.NonlinearExpression
    ) -> None:
        if not isinstance(nonlinear_expression, nle.NonlinearExpression):
            return
        if nonlinear_expression.representative_variable.is_discretized:
            self._start_new_mpip_instance(nonlinear_expression)
        else:
            for child_nonlinear_expression in nonlinear_expression.child_expressions:
                self._process_expression_tree(child_nonlinear_expression)

    def _start_new_mpip_instance(
        self, nonlinear_expression: nle.NonlinearExpression
    ) -> None:
        self.mpip_counter += 1
        mpip_id = lsf.mpip_id(self.mpip_counter)
        mpip = mp.MPIP(mpip_id, self.model_data.settings.pwl_method)
        representative_variable = nonlinear_expression.representative_variable
        mpip.add_implied_id(representative_variable)
        mpip.implying_function = self._nonlinear_expression_to_scip_expression(
            nonlinear_expression, mpip
        )
        if mpip.feasible:
            self.mpip_dict[mpip_id] = mpip

    def _nonlinear_expression_to_scip_expression(
        self, nonlinear_expression: nle.NonlinearExpression, mpip: mp.MPIP
    ) -> scip.Expr:
        if nonlinear_expression.expression_type == lsf.expression_type_product():
            implying_function = 1
            for child_nonlinear_expression in nonlinear_expression.child_expressions:
                implying_function *= self._continue_mpip_instance(
                    child_nonlinear_expression, mpip
                )
            return implying_function
        if nonlinear_expression.expression_type == lsf.expression_type_sum():
            implying_function = 0
            for child_nonlinear_expression in nonlinear_expression.child_expressions:
                implying_function += self._continue_mpip_instance(
                    child_nonlinear_expression, mpip
                )
            return implying_function
        if nonlinear_expression.expression_type == lsf.expression_type_divide():
            return self._continue_mpip_instance(
                nonlinear_expression.child_expressions[0], mpip
            ) / self._continue_mpip_instance(
                nonlinear_expression.child_expressions[1], mpip
            )
        scip_function = sw.get_nonlinear_function_scip(
            nonlinear_expression.expression_type
        )
        return scip_function(
            self._continue_mpip_instance(
                nonlinear_expression.child_expressions[0], mpip
            )
        )

    def _continue_mpip_instance(
        self,
        nonlinear_expression: nle.NonlinearExpression | tuple | float,
        mpip: mp.MPIP,
    ) -> scip.Expr | float:
        if isinstance(nonlinear_expression, tuple):
            coeff, variable = nonlinear_expression
            if variable.is_discretized:
                mpip.add_implying_id(variable)
                return coeff * mpip.interval_lp_implying_vars[variable.name]
            mpip.feasible = False
            return 1.0
        if isinstance(nonlinear_expression, float):
            return nonlinear_expression
        representative_variable = nonlinear_expression.representative_variable
        if representative_variable.is_discretized:
            mpip.add_implying_id(representative_variable)
            self._start_new_mpip_instance(nonlinear_expression)
            return mpip.interval_lp_implying_vars[representative_variable.name]
        return self._nonlinear_expression_to_scip_expression(nonlinear_expression, mpip)
