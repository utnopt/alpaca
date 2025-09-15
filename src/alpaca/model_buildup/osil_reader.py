# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from bs4 import BeautifulSoup

from alpaca.settings import StaticSettings
import alpaca.utils.datahandling as udh
from alpaca.model_data import (
    variable as var,
    constraint as con,
)
from alpaca.expressions import one_dim_expression as ode

if TYPE_CHECKING:
    from alpaca.model_data.model_data import ModelData


class OsilReader:
    """Reads and parses an OSiL file to populate the model data."""

    def __init__(self, model_data: ModelData):
        self.model_data = model_data

    def build_from_osil(self):
        """Reads the OSiL file and builds the initial model structure."""
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
            StaticSettings.instances_path
            + self.model_data.settings.osil_file_name
            + ".osil",
            "r",
            encoding="utf-8",
        ) as f:
            data = f.read()
        return BeautifulSoup(data, "xml")

    def _add_variables_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses and adds variables from the OSiL data."""
        var_tags = osil_data.find("variables").find_all("var")
        for v in var_tags:
            variable = self.model_data.add_variable(
                var.Variable(f"x_{len(self.model_data.variables) - 1}")
            )
            lb = v.get("lb")
            variable.lb = (
                0
                if lb is None
                else -StaticSettings.infinity if lb == "-INF" else float(lb)
            )
            ub = v.get("ub")
            variable.ub = (
                StaticSettings.infinity
                if ub is None
                else StaticSettings.infinity if ub == "INF" else float(ub)
            )
            var_type = v.get("type")
            variable.var_type = "C" if var_type is None else var_type

    def _add_constraints_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses and adds constraints from the OSiL data."""
        try:
            cons_tags = osil_data.find("constraints").find_all("con")
        except AttributeError:
            return
        for c in cons_tags:
            constraint = self.model_data.add_constraint(
                con.Constraint(f"c_{len(self.model_data.constraints) - 1}")
            )
            lb = c.get("lb")
            ub = c.get("ub")
            constraint.con_type = "<=" if lb is None else ">=" if ub is None else "=="
            constraint.rhs = float(ub) if lb is None else float(lb)

    def _add_objective_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses the objective function from OSiL and adds it as a constraint."""
        objective = osil_data.find("objectives").find_all("obj")[0]
        constraint = con.Constraint("c_-1", con_type="<=")
        self.model_data.add_constraint(constraint)
        constraint.variables.append((-1.0, self.model_data.variables["x_-1"]))
        coeff_tags = objective.find_all("coef")
        for c in coeff_tags:
            constraint.variables.append(
                (float(c.string), self.model_data.variables[f"x_{c.get('idx')}"])
            )

    @staticmethod
    def _expand_osil_elements(parent_element: BeautifulSoup, dtype=float) -> list:
        """Expands compressed OSiL element tags into a full list of values."""
        el_tags = parent_element.find_all("el")
        if el_tags is None:
            return []
        expanded = []
        for el in el_tags:
            if "mult" in el.attrs:
                count = int(el.attrs["mult"])
                base_val = dtype(el.text)
                incr = dtype(el.attrs.get("incr", 0))
                expanded.extend(base_val + i * incr for i in range(count))
            else:
                expanded.append(dtype(el.text))
        return expanded

    def _add_linear_expressions_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        """Parses linear constraint coefficients and adds them to constraints."""
        lin_con = osil_data.find("linearConstraintCoefficients")
        if lin_con is None:
            return
        try:
            num_vals = int(lin_con.get("numberOfValues"))
        except (TypeError, ValueError):
            return
        start_elem = lin_con.find("start")
        if start_elem is None:
            return
        start_vals = self._expand_osil_elements(start_elem, int)
        start_vals.append(num_vals)
        var_indices = self._expand_osil_elements(lin_con.find("colIdx"), int)
        coeff_vals = self._expand_osil_elements(lin_con.find("value"), float)
        if len(var_indices) != num_vals or len(coeff_vals) != num_vals:
            return
        for con_idx in range(len(start_vals) - 1):
            start = start_vals[con_idx]
            end = start_vals[con_idx + 1]
            for pos in range(start, end):
                self.model_data.constraints[f"c_{con_idx}"].variables.append(
                    (
                        coeff_vals[pos],
                        self.model_data.variables[f"x_{var_indices[pos]}"],
                    )
                )

    def _add_quadratic_expressions_from_osil_data(
        self, osil_data: BeautifulSoup
    ) -> None:
        """Parses quadratic terms and adds them as expressions to constraints."""
        try:
            quad_tags = osil_data.find("quadraticCoefficients").find_all("qTerm")
        except AttributeError:
            return
        for q in quad_tags:
            first_var_index = q.get("idxOne")
            second_var_index = q.get("idxTwo")
            constraint_index = q.get("idx")
            coeff = float(q.get("coef"))
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
        expr_hash = f"q_{var_index}"
        square_expression = self.model_data.add_one_dim_expression(
            ode.SquareExpression,
            expr_hash,
            self.model_data.variables[f"x_{var_index}"],
            0,
        )
        self.model_data.constraints[f"c_{constraint_index}"].variables.append(
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
        expr_hash = "b" + "_".join(sorted([first_var_index, second_var_index]))
        bilinear_expression = self.model_data.add_bilinear_expression(
            expr_hash,
            [
                self.model_data.variables[f"x_{first_var_index}"],
                self.model_data.variables[f"x_{second_var_index}"],
            ],
            0,
        )
        self.model_data.constraints[f"c_{constraint_index}"].variables.append(
            (coeff, bilinear_expression.representative_variable)
        )

    def _add_nonlinear_expressions_from_osil_data(
        self, osil_data: BeautifulSoup
    ) -> None:
        """Parses nonlinear expressions from OSiL and adds them to constraints."""
        try:
            nonlinear_tags = osil_data.find("nonlinearExpressions").find_all("nl")
        except AttributeError:
            return
        for n in nonlinear_tags:
            coeff = 1.0 if n.get("coef") is None else float(n.get("coef"))
            expr_hash = udh.hash_nonlinearity(str(n.next))
            nl_expression = self.model_data.add_nonlinear_expression(expr_hash, n.next)
            self.model_data.constraints[f"c_{n.get('idx')}"].variables.append(
                (coeff, nl_expression.representative_variable)
            )
