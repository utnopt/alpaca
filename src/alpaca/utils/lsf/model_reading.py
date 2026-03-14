# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, too-many-public-methods
"""
@authors: kuen,
"""


class ModelReading:
    """Model input reading."""

    # --- Osil Specific ---

    @classmethod
    def osil_file_suffix(cls) -> str:
        return ".osil"

    @classmethod
    def osil_input_format_xml(cls) -> str:
        return "xml"

    @classmethod
    def osil_inf_neg(cls) -> str:
        return "-INF"

    @classmethod
    def osil_inf_pos(cls) -> str:
        return "INF"

    @classmethod
    def osil_tag_sense(cls) -> str:
        return "maxOrMin"

    @classmethod
    def osil_sense_maximize(cls) -> str:
        return "max"

    @classmethod
    def osil_tag_variables(cls) -> str:
        return "variables"

    @classmethod
    def osil_tag_variable(cls) -> str:
        return "variable"

    @classmethod
    def osil_tag_number(cls) -> str:
        return "number"

    @classmethod
    def osil_tag_var(cls) -> str:
        return "var"

    @classmethod
    def osil_tag_constraints(cls) -> str:
        return "constraints"

    @classmethod
    def osil_tag_con(cls) -> str:
        return "con"

    @classmethod
    def osil_tag_objectives(cls) -> str:
        return "objectives"

    @classmethod
    def osil_tag_obj(cls) -> str:
        return "obj"

    @classmethod
    def osil_tag_el(cls) -> str:
        return "el"

    @classmethod
    def osil_tag_linear_constraint_coefficients(cls) -> str:
        return "linearConstraintCoefficients"

    @classmethod
    def osil_tag_start(cls) -> str:
        return "start"

    @classmethod
    def osil_tag_col_idx(cls) -> str:
        return "colIdx"

    @classmethod
    def osil_tag_value(cls) -> str:
        return "value"

    @classmethod
    def osil_tag_quadratic_coefficients(cls) -> str:
        return "quadraticCoefficients"

    @classmethod
    def osil_tag_qterm(cls) -> str:
        return "qTerm"

    @classmethod
    def osil_tag_nonlinear_expressions(cls) -> str:
        return "nonlinearExpressions"

    @classmethod
    def osil_tag_nl(cls) -> str:
        return "nl"

    @classmethod
    def osil_attr_lb(cls) -> str:
        return "lb"

    @classmethod
    def osil_attr_ub(cls) -> str:
        return "ub"

    @classmethod
    def osil_attr_type(cls) -> str:
        return "type"

    @classmethod
    def osil_attr_coef(cls) -> str:
        return "coef"

    @classmethod
    def osil_attr_idx(cls) -> str:
        return "idx"

    @classmethod
    def osil_attr_mult(cls) -> str:
        return "mult"

    @classmethod
    def osil_attr_incr(cls) -> str:
        return "incr"

    @classmethod
    def osil_attr_number_of_values(cls) -> str:
        return "numberOfValues"

    @classmethod
    def osil_attr_idx_one(cls) -> str:
        return "idxOne"

    @classmethod
    def osil_attr_idx_two(cls) -> str:
        return "idxTwo"
