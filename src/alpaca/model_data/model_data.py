# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.settings import UserSettings
from alpaca.model_data import (
    variable as var,
    constraint as con,
)
from alpaca.expressions import (
    expression_container as eco,
    bilinear_expression as ble,
    bilinear_binary_expression as bbe,
    bilinear_mixed_binary_expression as bme,
    multilinear_expression as mle,
    one_dim_expression as ode,
    nonlinear_expression as nle,
    linear_expression as lie,
)
from alpaca.model_buildup import (
    bound_propagator as bpr,
    breakpoint_generator as dis,
    expression_tree as etr,
    multilinear_handler as mlh,
    osil_reader as osr,
    pwl_handler as pwh,
)
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


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
        self.variables: dict[str, var.Variable] = {
            lsf.objective_var(): var.Variable(lsf.objective_var())
        }
        self.constraints: dict[str, con.LinearConstraint] = {}
        self.expressions = eco.ExpressionContainer()
        self._build_model_from_osil_data()

    def add_constraint(self, constraint: con.LinearConstraint) -> con.LinearConstraint:
        """Add a constraint to the model.

        Args:
            constraint: The constraint to be added.

        Returns:
            The added constraint object.
        """
        constraint_name = constraint.name
        assert constraint_name not in self.constraints, lsf.error_duplicate_con_name(
            constraint_name
        )
        self.constraints[constraint_name] = constraint
        return constraint

    def add_variable(
        self,
        variable: var.Variable,
        representative_variable: var.Variable | None = None,
    ) -> var.Variable:
        """Add a variable to the model.

        Args:
            variable: The variable to be added.
            representative_variable: The variable to be added represents an expression.

        Returns:
            The added variable object.
        """
        if representative_variable:
            return representative_variable
        variable_name = variable.name
        assert variable_name not in self.variables, lsf.error_duplicate_var_name(
            variable_name
        )
        self.variables[variable_name] = variable
        return variable

    def add_bilinear_expression(
        self,
        name: str,
        variables: list[var.Variable],
        level: int,
        representative_variable: var.Variable | None = None,
    ) -> (
        ble.BilinearExpression
        | bbe.BilinearBinaryExpression
        | bme.BilinearMixedBinaryExpression
    ):
        """Add a bilinear expression to the model, or return it if it already exists.

        Args:
            name: The unique name or hash for the expression.
            variables: A tuple of two variable objects involved in the expression.
            level: The nesting level of the expression in the model hierarchy.
            representative_variable: An optional variable that represents this expression.

        Returns:
            The newly created or existing BilinearExpression object.
        """
        if name in self.expressions.bilinear_expressions:
            return self.expressions.bilinear_expressions[name]
        if name in self.expressions.bilinear_binary_expressions:
            return self.expressions.bilinear_binary_expressions[name]
        if (
            variables[0].var_type == lsf.var_type_binary()
            and variables[1].var_type == lsf.var_type_binary()
        ):
            bilinear_expression = bbe.BilinearBinaryExpression(
                name,
                self,
                variables,
                representative_variable=representative_variable,
            )
            self.expressions.bilinear_binary_expressions[name] = bilinear_expression
            return bilinear_expression
        if variables[0].var_type == lsf.var_type_binary():
            bilinear_expression = bme.BilinearMixedBinaryExpression(
                name,
                self,
                variables,
                level,
                representative_variable=representative_variable,
            )
            self.expressions.bilinear_mixed_binary_expressions[name] = (
                bilinear_expression
            )
            return bilinear_expression
        if variables[1].var_type == lsf.var_type_binary():
            bilinear_expression = bme.BilinearMixedBinaryExpression(
                name,
                self,
                [variables[1], variables[0]],
                level,
                representative_variable=representative_variable,
            )
            self.expressions.bilinear_mixed_binary_expressions[name] = (
                bilinear_expression
            )
            return bilinear_expression
        bilinear_expression = ble.BilinearExpression(
            name,
            self,
            variables,
            level,
            representative_variable=representative_variable,
        )
        self.expressions.bilinear_expressions[name] = bilinear_expression
        return bilinear_expression

    def add_multilinear_expression(
        self,
        name: str,
        variables: list[var.Variable],
        level: int,
        representative_variable: var.Variable | None = None,
    ) -> mle.MultilinearExpression:
        """Add a multilinear expression to the model, or return it if it already exists.

        Args:
            name: The unique name or hash for the expression.
            variables: A list of variable objects involved in the expression.
            level: The nesting level of the expression in the model hierarchy.
            representative_variable: An optional variable that represents this expression.

        Returns:
            The newly created or existing MultilinearExpression object.
        """
        if name in self.expressions.multilinear_expressions:
            return self.expressions.multilinear_expressions[name]
        multilinear_expression = mle.MultilinearExpression(
            name,
            self,
            variables,
            level,
            representative_variable=representative_variable,
        )
        self.expressions.multilinear_expressions[name] = multilinear_expression
        return multilinear_expression

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def add_one_dim_expression(
        self,
        expression_class,
        name: str,
        variable: var.Variable,
        level: int,
        representative_variable: var.Variable | None = None,
    ) -> ode.OneDimExpression:
        """Add a one-dimensional expression to the model, or return it if it already exists.

        Args:
            expression_class: The specific class of the one-dimensional expression.
            name: The unique name or hash for the expression.
            variable: The variable object involved in the expression.
            level: The nesting level of the expression in the model hierarchy.
            representative_variable: An optional variable that represents this expression.

        Returns:
            The newly created or existing OneDimExpression object.
        """
        if name in self.expressions.one_dim_expressions:
            return self.expressions.one_dim_expressions[name]
        one_dim_expression = expression_class(
            name,
            self,
            variable,
            level,
            representative_variable=representative_variable,
        )
        self.expressions.one_dim_expressions[name] = one_dim_expression
        return one_dim_expression

    def add_nonlinear_expression(
        self,
        name: str,
        expression_tag,
    ) -> nle.NonlinearExpression:
        """Add a nonlinear expression to the model, or return it if it already exists.

        Args:
            name: The unique name or hash for the expression.
            expression_tag: The BeautifulSoup XML tag representing the nonlinear expression.

        Returns:
            The newly created or existing NonlinearExpression object.
        """
        if name in self.expressions.nonlinear_expressions:
            return self.expressions.nonlinear_expressions[name]
        nl_expression = nle.NonlinearExpression(name, expression_tag, self)
        self.expressions.nonlinear_expressions[name] = nl_expression
        return nl_expression

    def add_linear_expression(
        self,
        name: str,
        level: int,
        representative_variable: var.Variable | None = None,
    ) -> lie.LinearExpression:
        """Add a linear expression to the model, or return it if it already exists.

        Args:
            name: The unique name or hash for the expression.
            level: The nesting level of the expression in the model hierarchy.
            representative_variable: An optional variable that represents this expression.

        Returns:
            The newly created or existing LinearExpression object.
        """
        if name in self.expressions.linear_expressions:
            return self.expressions.linear_expressions[name]
        linear_expression = lie.LinearExpression(
            name,
            self,
            level,
            representative_variable=representative_variable,
        )
        self.expressions.linear_expressions[name] = linear_expression
        return linear_expression

    def _build_model_from_osil_data(self) -> None:
        """Create a complete model from an OSiL data file.

        This method orchestrates the entire process of reading an OSiL file,
        building variables, constraints, and expressions, propagating bounds,
        and preparing the model for optimization using dedicated handler classes.
        """

        # Step 1: Read OSiL file and build the basic model structure.
        # This includes variables, objective, constraints, and all expressions.
        osr.OsilReader(self).build_from_osil()

        # Step 3: Decompose complex nonlinear expression trees.
        etr.ExpressionTree(self).decompose()

        # Step 4: Handle multilinear and bilinear terms based on settings.
        multilinear_handler = mlh.MultilinearHandler(self)
        multilinear_handler.handle()

        # Step 5: Propagate variable bounds through the expression structures.
        bpr.BoundPropagator(self).propagate_bounds()

        if self.settings.pwl_method == lsf.pwl_method_none():
            self._translate_linear_expressions_to_constraints()
            return

        # Step 6: Add McCormick envelopes for bilinear terms if specified.
        multilinear_handler.add_mccormick_envelopes()

        # Step 7: Discretize variables that are part of nonlinear terms.
        dis.BreakpointGenerator(self).generate_breakpoints()

        # Step 8: Apply piecewise relaxations for remaining nonlinear expressions.
        pwh.PWLHandler(self).apply_relaxations()

        # Step 9: Convert any remaining linear expression objects into standard constraints.
        self._translate_linear_expressions_to_constraints()

    def _translate_linear_expressions_to_constraints(
        self,
    ) -> None:
        """Converts expression objects into their equivalent constraint representations."""
        for expression in self.expressions.linear_expressions.values():
            expression.add_constraint_from_linear_expression()
