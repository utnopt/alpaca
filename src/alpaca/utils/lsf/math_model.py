# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, too-many-public-methods
"""
@authors: kuen,
"""


class MathModel:
    """Mathematical elements, objectives, variable types, and naming."""

    # --- Mathematical & Model Elements ---

    @classmethod
    def constraint_leq(cls) -> str:
        return "<="

    @classmethod
    def constraint_geq(cls) -> str:
        return ">="

    @classmethod
    def constraint_eq(cls) -> str:
        return "=="

    @classmethod
    def numpy_infinity(cls) -> str:
        return "inf"

    # --- Objective & Optimization Status ---

    @classmethod
    def objective_var(cls) -> str:
        return "obj_var"

    @classmethod
    def objective_sense_minimize(cls) -> str:
        return "minimize"

    @classmethod
    def objective_sense_maximize(cls) -> str:
        return "maximize"

    @classmethod
    def opt_model_status_infeasible(cls) -> str:
        return "infeasible"

    @classmethod
    def opt_model_status_optimal(cls) -> str:
        return "optimal"

    # --- Variable & Constraint Naming Conventions ---

    @classmethod
    def var_name(cls, index: int | str) -> str:
        return f"x_{index}"

    @classmethod
    def con_name(cls, index: int | str) -> str:
        return f"c_{index}"

    @classmethod
    def obj_con_name(cls) -> str:
        return cls.con_name(-1)

    @classmethod
    def representative_variable_name(cls, name: str) -> str:
        return f"rep_{name}"

    @classmethod
    def helper_variable_name(cls, name: str) -> str:
        return f"helper_{name}"

    @classmethod
    def mpip_id(cls, counter: int) -> str:
        return f"mpip_{counter}"

    @classmethod
    def mpip_interval_lp_var_name(cls, name: str) -> str:
        return f"x_{name}"

    # --- Variable Types ---

    @classmethod
    def var_type_binary(cls) -> str:
        return "B"

    @classmethod
    def var_type_integer(cls) -> str:
        return "I"

    @classmethod
    def var_type_continuous(cls) -> str:
        return "C"

    # --- Expression Types & Hashes ---

    @classmethod
    def expression_type_product(cls) -> str:
        return "product"

    @classmethod
    def expression_type_sum(cls) -> str:
        return "sum"

    @classmethod
    def expression_type_divide(cls) -> str:
        return "divide"

    @classmethod
    def expression_hash_bilinear(cls, idx1: str, idx2: str) -> str:
        return "bl" + "_".join(sorted([idx1, idx2]))

    @classmethod
    def expression_hash_multilinear(cls, names: list[str]) -> str:
        return "ml" + "_".join(sorted(names))

    @classmethod
    def expression_hash_linear(cls, name: str) -> str:
        return f"le_{name}"

    @classmethod
    def expression_hash_square(cls, name: str) -> str:
        return f"square_{name}"

    @classmethod
    def expression_hash_generic_nonlinear(cls, name: str, nonlinearity: str) -> str:
        return f"{nonlinearity}_{name}"

    # --- Nonlinearity Types ---

    @classmethod
    def nonlinearity_type_square(cls) -> str:
        return "square"

    @classmethod
    def nonlinearity_type_exp(cls) -> str:
        return "exp"

    @classmethod
    def nonlinearity_type_ln(cls) -> str:
        return "ln"

    @classmethod
    def nonlinearity_type_sqrt(cls) -> str:
        return "sqrt"

    @classmethod
    def nonlinearity_type_sin(cls) -> str:
        return "sin"

    @classmethod
    def nonlinearity_type_cos(cls) -> str:
        return "cos"

    @classmethod
    def nonlinearity_type_log10(cls) -> str:
        return "log10"

    @classmethod
    def nonlinearity_type_tanh(cls) -> str:
        return "tanh"

    @classmethod
    def nonlinearity_type_inverse(cls) -> str:
        return "inverse"

    @classmethod
    def nonlinearity_type_xabsx(cls) -> str:
        return "xabsx"

    @classmethod
    def nonlinearity_type_abs(cls) -> str:
        return "abs"

    @classmethod
    def nonlinearity_type_negate(cls) -> str:
        return "negate"

    @classmethod
    def nonlinearity_type_min(cls) -> str:
        return "min"

    @classmethod
    def nonlinearity_type_power(cls) -> str:
        return "power"

    @classmethod
    def nonlinearity_type_multilinear_implied(cls, nr_of_implying_vars: int) -> str:
        return f"multilinear_implied_{nr_of_implying_vars}"

    @classmethod
    def nonlinearity_type_multilinear(cls, nr_of_implying_vars: int) -> str:
        return f"multilinear_{nr_of_implying_vars}"

    @classmethod
    def power_expression_class_name(cls) -> str:
        return "PowerExpression"
