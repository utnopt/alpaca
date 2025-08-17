# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from bs4 import BeautifulSoup

from alpaca.settings import UserSettings, StaticSettings
from alpaca.utils.logger import logger
import alpaca.utils.datahandling as udh
from alpaca.model_data import (
    variable as var,
    constraint as con,
)
from alpaca.expressions import (
    expression_container as eco,
    bilinear_expression as ble,
    multilinear_expression as mle,
    one_dim_expression as ode,
    nonlinear_expression as nle,
    linear_expression as lie,
)


class ModelData:  # pylint: disable=too-many-instance-attributes
    """A comprehensive data container for optimization models based on OSiL format.

    This class maintains collections of variables, constraints, and various types of expressions
    (nonlinear, bilinear, multilinear, and one-dimensional) that comprise an optimization model.
    It provides methods to build and manipulate model components from OSiL data.
    """

    def __init__(self, settings: UserSettings):
        """Initialize a ModelData instance with user settings.

        Args:
            settings: User configuration settings for the model.
        """
        self.settings = settings
        self.variables: dict[str, var.Variable] = {"x_-1": var.Variable("x_-1")}
        self.constraints: dict[str, con.Constraint] = {}
        self.expressions = eco.ExpressionContainer()
        self._build_model_from_osil_data()

    def add_constraint(self, constraint: con.Constraint) -> con.Constraint:
        """Add a constraint to the model.

        Args:
            constraint: The constraint to be added.
        """
        constraint_name = constraint.name
        assert (
            constraint_name not in self.constraints
        ), f"Duplicate constraint name {constraint_name}."
        self.constraints[constraint_name] = constraint
        return constraint

    def add_variable(self, variable: var.Variable) -> var.Variable:
        """Add a variable to the model.

        Args:
            variable: The variable to be added.
        """
        variable_name = variable.name
        assert (
            variable_name not in self.variables
        ), f"Duplicate variable name {variable_name}."
        self.variables[variable_name] = variable
        return variable

    def add_bilinear_expression(
        self, bilinear_expression: ble.BilinearExpression
    ) -> ble.BilinearExpression:
        """Add a bilinear expression to the model.

        Args:
            bilinear_expression: The bilinear expression to be added.
        """
        bilinear_expression_name = bilinear_expression.name
        if bilinear_expression_name in self.expressions.bilinear_expressions:
            return self.expressions.bilinear_expressions[bilinear_expression_name]
        self.expressions.bilinear_expressions[bilinear_expression_name] = (
            bilinear_expression
        )
        return bilinear_expression

    def add_multilinear_expression(
        self, multilinear_expression: mle.MultilinearExpression
    ) -> mle.MultilinearExpression:
        """Add a multilinear expression to the model.

        Args:
            multilinear_expression: The multilinear expression to be added.
        """
        multilinear_expression_name = multilinear_expression.name
        if multilinear_expression_name in self.expressions.multilinear_expressions:
            return self.expressions.multilinear_expressions[multilinear_expression_name]
        self.expressions.multilinear_expressions[multilinear_expression_name] = (
            multilinear_expression
        )
        return multilinear_expression

    def add_one_dim_expression(
        self, one_dim_expression: ode.OneDimExpression
    ) -> ode.OneDimExpression:
        """Add a one-dimensional expression to the model.

        Args:
            one_dim_expression: The one-dimensional expression to be added.
        """
        one_dim_expression_name = one_dim_expression.name
        if one_dim_expression_name in self.expressions.one_dim_expressions:
            return self.expressions.one_dim_expressions[one_dim_expression_name]
        self.expressions.one_dim_expressions[one_dim_expression_name] = (
            one_dim_expression
        )
        return one_dim_expression

    def add_nonlinear_expression(
        self, nonlinear_expression: nle.NonlinearExpression
    ) -> nle.NonlinearExpression:
        """Add a nonlinear expression to the model.

        Args:
            nonlinear_expression: The nonlinear expression to be added.
        """
        nonlinear_expression_name = nonlinear_expression.name
        if nonlinear_expression_name in self.expressions.nonlinear_expressions:
            return self.expressions.nonlinear_expressions[nonlinear_expression_name]
        self.expressions.nonlinear_expressions[nonlinear_expression_name] = (
            nonlinear_expression
        )
        return nonlinear_expression

    def add_linear_expression(
        self, linear_expression: lie.LinearExpression
    ) -> lie.LinearExpression:
        """Add a linear expression to the model.

        Args:
            linear_expression: The linear expression to be added.
        """
        linear_expression_name = linear_expression.name
        if linear_expression_name in self.expressions.linear_expressions:
            return self.expressions.linear_expressions[linear_expression_name]
        self.expressions.linear_expressions[linear_expression_name] = linear_expression
        return linear_expression

    def _build_model_from_osil_data(self) -> None:
        """Create a complete model from OSiL data file.

        Reads the OSiL file specified in settings, builds variables, constraints,
        and different types of expressions, and applies piecewise linear approximations
        to nonlinear functions.
        """
        logger.info("Reading data..")
        osil_data = self._read_osil_file()
        self._add_variables_from_osil_data(osil_data)
        self._add_objective_from_osil_data(osil_data)
        self._add_constraints_from_osil_data(osil_data)
        self._add_linear_expressions_from_osil_data(osil_data)
        self._add_quadratic_expressions_from_osil_data(osil_data)
        self._add_nonlinear_expressions_from_osil_data(osil_data)
        self._propagate_bounds_linear_constraints()
        self.expressions.first_level_nonlinear_expression_keys = list(
            self.expressions.nonlinear_expressions.keys()
        )
        self._grow_nonlinear_expression_trees()
        self._fragment_expression_trees_to_low_dimensional_functions()
        if self.settings.reformulate_multilinear:
            self._reformulate_multilinear_and_bilinear_expressions()
        self._propagate_bounds_expressions()
        self._discretize_variables()
        self._translate_expressions_to_constraints()

    def _read_osil_file(self) -> BeautifulSoup:
        with open(
            StaticSettings.instances_path + self.settings.osil_file_name + ".osil",
            "r",
            encoding="utf-8",
        ) as f:
            data = f.read()
        data = BeautifulSoup(data, "xml")
        return data

    def _add_variables_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        var_tags = osil_data.find("variables").find_all("var")
        for v in var_tags:
            variable = self.add_variable(var.Variable(f"x_{len(self.variables) - 1}"))
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
        try:
            cons_tags = osil_data.find("constraints").find_all("con")
        except AttributeError:
            return
        for c in cons_tags:
            constraint = self.add_constraint(
                con.Constraint(f"c_{len(self.constraints) - 1}")
            )
            lb = c.get("lb")
            ub = c.get("ub")
            constraint.con_type = "<=" if lb is None else ">=" if ub is None else "=="
            constraint.rhs = float(ub) if lb is None else float(lb)

    def _add_objective_from_osil_data(self, osil_data: BeautifulSoup) -> None:
        objective = osil_data.find("objectives").find_all("obj")[0]
        constraint = con.Constraint(f"c_{-1}", con_type="<=")
        self.add_constraint(constraint)
        constraint.variables.append((-1.0, self.variables["x_-1"]))
        coeff_tags = objective.find_all("coef")
        for c in coeff_tags:
            constraint.variables.append(
                (float(c.string), self.variables[f"x_{c.get('idx')}"])
            )

    @staticmethod
    def _expand_osil_elements(parent_element: BeautifulSoup, dtype=float) -> list:
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
                self.constraints[f"c_{con_idx}"].variables.append(
                    (coeff_vals[pos], self.variables[f"x_{var_indices[pos]}"])
                )

    def _add_quadratic_expressions_from_osil_data(
        self, osil_data: BeautifulSoup
    ) -> None:
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
        expr_hash = f"q_{var_index}"
        square_expression = self.add_one_dim_expression(
            ode.SquareExpression(expr_hash, self, self.variables[f"x_{var_index}"], 0)
        )
        self.constraints[f"c_{constraint_index}"].variables.append(
            (coeff, square_expression.representative_variable)
        )

    def _add_bilinear_expression_to_constraint(
        self,
        first_var_index: str,
        second_var_index: str,
        constraint_index: str,
        coeff: float,
    ):
        expr_hash = "b" + "_".join(sorted([first_var_index, second_var_index]))
        bilinear_expression = self.add_bilinear_expression(
            ble.BilinearExpression(
                expr_hash,
                self,
                (
                    self.variables[f"x_{first_var_index}"],
                    self.variables[f"x_{second_var_index}"],
                ),
                0,
            )
        )
        self.constraints[f"c_{constraint_index}"].variables.append(
            (coeff, bilinear_expression.representative_variable)
        )

    def _add_nonlinear_expressions_from_osil_data(
        self, osil_data: BeautifulSoup
    ) -> None:
        try:
            nonlinear_tags = osil_data.find("nonlinearExpressions").find_all("nl")
        except AttributeError:
            return
        for n in nonlinear_tags:
            coeff = 1.0 if n.get("coef") is None else float(n.get("coef"))
            expr_hash = udh.hash_nonlinearity(str(n.next))
            nonlinear_expression = self.add_nonlinear_expression(
                nle.NonlinearExpression(expr_hash, n.next, self)
            )
            self.constraints[f"c_{n.get('idx')}"].variables.append(
                (coeff, nonlinear_expression.representative_variable)
            )

    def _grow_nonlinear_expression_trees(self) -> None:
        for (
            nonlinear_expression_key
        ) in self.expressions.first_level_nonlinear_expression_keys:
            nonlinear_expression = self.expressions.nonlinear_expressions[
                nonlinear_expression_key
            ]
            nonlinear_expression.grow_expression_tree()

    def _fragment_expression_trees_to_low_dimensional_functions(self) -> None:
        for (
            nonlinear_expression_key
        ) in self.expressions.first_level_nonlinear_expression_keys:
            nonlinear_expression = self.expressions.nonlinear_expressions[
                nonlinear_expression_key
            ]
            nonlinear_expression.fragment_expression_tree_to_low_dimensional_functions(
                1
            )

    def _reformulate_multilinear_and_bilinear_expressions(self) -> None:
        for expression in self.expressions.multilinear_expressions.values():
            expression.reformulate_to_bilinear_expressions()
        for expression in self.expressions.bilinear_expressions.values():
            expression.reformulate_to_sum_of_squares()

    def _propagate_bounds_expressions(self):
        sorted_expressions = sorted(
            self.expressions.all_low_dim_expressions(),
            key=lambda e: -e.level,
        )
        for _ in range(self.settings.bound_propagation_rounds):
            for expression in sorted_expressions:
                expression.propagate_variable_bounds()

    def _propagate_bounds_linear_constraints(self):
        for _ in range(self.settings.bound_propagation_rounds):
            for constraint in self.constraints.values():
                if constraint.con_type == "==":
                    constraint.propagate_variable_bounds()

    def _discretize_variables(
        self,
    ) -> None:
        pwl_variables = []
        pwl_constraints = []
        for variable in self.variables.values():
            if variable.is_discretized:
                variable.add_binary_pwl(
                    self.settings.number_of_breakpoints, self.settings.pwl_method
                )
                variable.add_continuous_pwl(self.settings.pwl_method)
                pwl_variables.extend(
                    variable.pwl_variables_binary + variable.pwl_variables_continuous
                )
                pwl_constraints.extend(variable.pwl_constraints)
        for variable in pwl_variables:
            self.add_variable(variable)
        for constraint in pwl_constraints:
            self.add_constraint(constraint)

    def _translate_expressions_to_constraints(
        self,
    ) -> None:
        self._apply_piecewise_linear_approximation()
        if not self.settings.reformulate_multilinear:
            self._apply_piecewise_constant_approximation()
        for expression in self.expressions.linear_expressions.values():
            expression.add_constraint_from_linear_expression()

    def _apply_piecewise_linear_approximation(self) -> None:
        for expression in self.expressions.one_dim_expressions.values():
            expression.apply_piecewise_linear_relaxation(
                approximation=self.settings.approximation
            )

    def _apply_piecewise_constant_approximation(self) -> None:
        for expression in self.expressions.bilinear_expressions.values():
            expression.apply_piecewise_constant_relaxation(
                approximation=self.settings.approximation
            )
        for expression in self.expressions.multilinear_expressions.values():
            expression.apply_piecewise_constant_relaxation(
                approximation=self.settings.approximation
            )
