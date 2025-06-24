# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pyscipopt as scip

from alpaca.utils.logger import logger
import alpaca.utils.datahandling as udh
from alpaca.model_data import nonlinear_expression as nle
import alpaca.mpip.mpip as mp


class MPIPHandler:  # pylint: disable=too-many-instance-attributes
    """Multipartite Implication Polytope handler."""

    def __init__(
        self, first_level_nonlinear_expressions: dict[str, nle.NonlinearExpression]
    ) -> None:
        """Initialize MPIP instance."""
        self.first_level_nonlinear_expressions = first_level_nonlinear_expressions
        self.mpip_dict: dict[str, mp.MPIP] = {}
        self.mpip_counter: int = 0
        self._find_mpip_instances_in_nonlinear_expression()

    def _find_mpip_instances_in_nonlinear_expression(self) -> None:
        """Extract mpip instances from nonlinear expression trees."""
        logger.info("Add feature mpip..")
        for nonlinear_expression in self.first_level_nonlinear_expressions.values():
            self._process_expression_tree(nonlinear_expression)

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
        mpip_id = f"mpip_{self.mpip_counter}"
        mpip = mp.MPIP(mpip_id)
        representative_variable = nonlinear_expression.representative_variable
        mpip.add_implied_id(
            representative_variable.name, representative_variable.breakpoints
        )
        mpip.implying_function = self._nonlinear_expression_to_scip_expression(
            nonlinear_expression, mpip
        )
        if mpip.feasible:
            self.mpip_dict[mpip_id] = mpip
            mpip.build_mpip()

    def _nonlinear_expression_to_scip_expression(
        self, nonlinear_expression: nle.NonlinearExpression, mpip: mp.MPIP
    ) -> scip.Expr:
        if nonlinear_expression.expression_type == "product":
            implying_function = 1
            for child_nonlinear_expression in nonlinear_expression.child_expressions:
                implying_function *= self._continue_mpip_instance(
                    child_nonlinear_expression, mpip
                )
            return implying_function
        if nonlinear_expression.expression_type == "sum":
            implying_function = 0
            for child_nonlinear_expression in nonlinear_expression.child_expressions:
                implying_function += self._continue_mpip_instance(
                    child_nonlinear_expression, mpip
                )
            return implying_function
        if nonlinear_expression.expression_type == "divide":
            return self._continue_mpip_instance(
                nonlinear_expression.child_expressions[0], mpip
            ) / self._continue_mpip_instance(
                nonlinear_expression.child_expressions[1], mpip
            )
        scip_function = udh.pyscipopt_nonlinearity(nonlinear_expression.expression_type)
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
                mpip.add_implying_id(variable.name, variable.breakpoints)
                return coeff * mpip.interval_lp_implying_vars[variable.name]
            mpip.feasible = False
            return 1.0
        if isinstance(nonlinear_expression, float):
            return nonlinear_expression
        representative_variable = nonlinear_expression.representative_variable
        if representative_variable.is_discretized:
            mpip.add_implying_id(
                representative_variable.name, representative_variable.breakpoints
            )
            self._start_new_mpip_instance(nonlinear_expression)
            return mpip.interval_lp_implying_vars[representative_variable.name]
        return self._nonlinear_expression_to_scip_expression(nonlinear_expression, mpip)
