# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from bs4 import BeautifulSoup

from alpaca.utils.logger import logger
from alpaca.settings import StaticSettings
import alpaca.utils.data_handling as udh
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.model_data import (
    variable as var,
    constraint as con,
)
from alpaca.expressions import one_dim_expression as ode

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class OsilReader:
    """Reads and parses an OSiL file to populate the model data."""

    def __init__(self, model_data: ModelData, path: str):
        self.model_data = model_data
        self.path = path

    def build_from_osil(self):
        """Reads the OSiL file and builds the initial model structure."""
        logger.info(lsf.info_read_osil_data())
        osil_data = self._read_osil_file()
        self._add_variables_from_osil_data(osil_data)
        self._add_objective_from_osil_data(osil_data)
        self._add_constraints_from_osil_data(osil_data)
        self._add_linear_expressions_from_osil_data(osil_data)
        self._add_quadratic_expressions_from_osil_data(osil_data)
        self._add_nonlinear_expressions_from_osil_data(osil_data)

    def _read_osil_file(self) -> BeautifulSoup:
        """Reads the OSiL file and returns its parsed XML content."""
        with open(
            self.path,
            lsf.file_mode_read(),
            encoding=lsf.file_encoding_utf8(),
        ) as f:
            data = f.read()
        return BeautifulSoup(data, lsf.osil_input_format_xml())

    def _add_variables_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses and adds variables from the OSiL data."""
        var_tags = osil_data.find(lsf.osil_tag_variables()).find_all(lsf.osil_tag_var())
        for v in var_tags:
            variable = self.model_data.add_variable(
                var.Variable(lsf.var_name(len(self.model_data.variables) - 1))
            )
            lb = v.get(lsf.osil_attr_lb())
            variable.lb = (
                0
                if lb is None
                else -StaticSettings.infinity if lb == lsf.osil_inf_neg() else float(lb)
            )
            ub = v.get(lsf.osil_attr_ub())
            variable.ub = (
                StaticSettings.infinity
                if ub is None
                else StaticSettings.infinity if ub == lsf.osil_inf_pos() else float(ub)
            )
            var_type = v.get(lsf.osil_attr_type())
            variable.var_type = (
                lsf.var_type_continuous() if var_type is None else var_type
            )

    def _add_constraints_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses and adds constraints from the OSiL data."""
        try:
            cons_tags = osil_data.find(lsf.osil_tag_constraints()).find_all(
                lsf.osil_tag_con()
            )
        except AttributeError:
            return
        for c in cons_tags:
            constraint = self.model_data.add_constraint(
                con.LinearConstraint(lsf.con_name(len(self.model_data.constraints) - 1))
            )
            lb = c.get(lsf.osil_attr_lb())
            ub = c.get(lsf.osil_attr_ub())
            constraint.con_type = (
                lsf.constraint_leq()
                if lb is None
                else lsf.constraint_geq() if ub is None else lsf.constraint_eq()
            )
            constraint.rhs = float(ub) if lb is None else float(lb)

    def _add_objective_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses the objective function from OSiL and adds it as a constraint."""
        objective = osil_data.find(lsf.osil_tag_objectives()).find_all(
            lsf.osil_tag_obj()
        )[0]
        constraint = self.model_data.add_constraint(
            con.LinearConstraint(lsf.obj_con_name(), con_type=lsf.constraint_leq())
        )
        constraint.variables.append(
            (-1.0, self.model_data.variables[lsf.objective_var()])
        )
        coeff_tags = objective.find_all(lsf.osil_attr_coef())
        for c in coeff_tags:
            constraint.variables.append(
                (
                    float(c.string),
                    self.model_data.variables[lsf.var_name(c.get(lsf.osil_attr_idx()))],
                )
            )

    @staticmethod
    def _expand_osil_elements(parent_element: BeautifulSoup, dtype=float) -> list:
        """Expands compressed OSiL element tags into a full list of values."""
        el_tags = parent_element.find_all(lsf.osil_tag_el())
        if el_tags is None:
            return []
        expanded = []
        for el in el_tags:
            if lsf.osil_attr_mult() in el.attrs:
                count = int(el.attrs[lsf.osil_attr_mult()])
                base_val = dtype(el.text)
                incr = dtype(el.attrs.get(lsf.osil_attr_incr(), 0))
                expanded.extend(base_val + i * incr for i in range(count))
            else:
                expanded.append(dtype(el.text))
        return expanded

    def _add_linear_expressions_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses linear constraint coefficients and adds them to constraints."""
        lin_con = osil_data.find(lsf.osil_tag_linear_constraint_coefficients())
        if lin_con is None:
            return
        try:
            num_vals = int(lin_con.get(lsf.osil_attr_number_of_values()))
        except (TypeError, ValueError):
            return
        start_elem = lin_con.find(lsf.osil_tag_start())
        if start_elem is None:
            return
        start_vals = self._expand_osil_elements(start_elem, int)
        start_vals.append(num_vals)
        var_indices = self._expand_osil_elements(
            lin_con.find(lsf.osil_tag_col_idx()), int
        )
        coeff_vals = self._expand_osil_elements(
            lin_con.find(lsf.osil_tag_value()), float
        )
        if len(var_indices) != num_vals or len(coeff_vals) != num_vals:
            return
        for con_idx in range(len(start_vals) - 1):
            start = start_vals[con_idx]
            end = start_vals[con_idx + 1]
            for pos in range(start, end):
                self.model_data.constraints[lsf.con_name(con_idx)].variables.append(
                    (
                        coeff_vals[pos],
                        self.model_data.variables[lsf.var_name(var_indices[pos])],
                    )
                )

    def _add_quadratic_expressions_from_osil_data(
        self, osil_data: BeautifulSoup
    ) -> None:
        """Parses quadratic terms and adds them as expressions to constraints."""
        try:
            quad_tags = osil_data.find(lsf.osil_tag_quadratic_coefficients()).find_all(
                lsf.osil_tag_qterm()
            )
        except AttributeError:
            return
        for q in quad_tags:
            first_var_index = q.get(lsf.osil_attr_idx_one())
            second_var_index = q.get(lsf.osil_attr_idx_two())
            constraint_index = q.get(lsf.osil_attr_idx())
            coeff = float(q.get(lsf.osil_attr_coef()))
            if first_var_index == second_var_index:
                self._add_square_expression_to_constraint(
                    first_var_index, constraint_index, coeff
                )
            else:
                self._add_bilinear_expression_to_constraint(
                    first_var_index, second_var_index, constraint_index, coeff
                )

    def _add_square_expression_to_constraint(
        self, var_index: str, constraint_index: str, coeff: float
    ):
        """Creates and adds a square expression to a specified constraint."""
        expr_hash = lsf.expression_hash_square(var_index)
        square_expression = self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            expr_hash,
            self.model_data.variables[lsf.var_name(var_index)],
            0,
        )
        self.model_data.constraints[lsf.con_name(constraint_index)].variables.append(
            (coeff, square_expression.representative_variable)
        )

    def _add_bilinear_expression_to_constraint(
        self,
        first_var_index: str,
        second_var_index: str,
        constraint_index: str,
        coeff: float,
    ):
        """Creates and adds a bilinear expression to a specified constraint."""
        expr_hash = lsf.expression_hash_bilinear(first_var_index, second_var_index)
        bilinear_expression = self.model_data.add_bilinear_expression(
            expr_hash,
            [
                self.model_data.variables[lsf.var_name(first_var_index)],
                self.model_data.variables[lsf.var_name(second_var_index)],
            ],
            0,
        )
        self.model_data.constraints[lsf.con_name(constraint_index)].variables.append(
            (coeff, bilinear_expression.representative_variable)
        )

    def _add_nonlinear_expressions_from_osil_data(
        self, osil_data: BeautifulSoup
    ) -> None:
        """Parses nonlinear expressions from OSiL and adds them to constraints."""
        try:
            nonlinear_tags = osil_data.find(
                lsf.osil_tag_nonlinear_expressions()
            ).find_all(lsf.osil_tag_nl())
        except AttributeError:
            return
        for n in nonlinear_tags:
            coeff = (
                1.0
                if n.get(lsf.osil_attr_coef()) is None
                else float(n.get(lsf.osil_attr_coef()))
            )
            expr_hash = udh.hash_nonlinearity(str(n.next))
            nl_expression = self.model_data.add_nonlinear_expression(expr_hash, n.next)
            self.model_data.constraints[
                lsf.con_name(n.get(lsf.osil_attr_idx()))
            ].variables.append((coeff, nl_expression.representative_variable))
