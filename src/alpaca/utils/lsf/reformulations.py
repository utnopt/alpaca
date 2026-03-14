# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, too-many-public-methods
"""
@authors: kuen,
"""


class Reformulations:
    """McCormick, bilinear, indicator, and absolute value reformulations."""

    # --- McCormick Relaxations ---

    @classmethod
    def con_name_mc_cormick_binary_ub_ub(cls, name: str) -> str:
        return f"mc_cormick_bb_ub_ub_{name}"

    @classmethod
    def con_name_mc_cormick_binary_ub_lb(cls, name: str) -> str:
        return f"mc_cormick_bb_ub_lb_{name}"

    @classmethod
    def con_name_mc_cormick_binary_lb_ub(cls, name: str) -> str:
        return f"mc_cormick_bb_lb_ub_{name}"

    @classmethod
    def con_name_mc_cormick_continuous_ub_ub(cls, name: str) -> str:
        return f"mc_cormick_cb_ub_ub_{name}"

    @classmethod
    def con_name_mc_cormick_continuous_ub_lb(cls, name: str) -> str:
        return f"mc_cormick_cb_ub_lb_{name}"

    @classmethod
    def con_name_mc_cormick_continuous_lb_ub(cls, name: str) -> str:
        return f"mc_cormick_cb_lb_ub_{name}"

    @classmethod
    def con_name_mc_cormick_continuous_lb_lb(cls, name: str) -> str:
        return f"mc_cormick_cb_lb_lb_{name}"

    # --- Bilinear & Indicator Reformulations ---

    @classmethod
    def linear_expression_bilinear_to_sum_of_squares(cls, name: str) -> str:
        return f"bilinear_to_sum_of_squares_{name}"

    @classmethod
    def linear_expression_bilinear_to_sum_of_squares_helper(cls, name: str) -> str:
        return f"bilinear_to_sum_of_squares_helper_{name}"

    @classmethod
    def con_name_indicator_bilinear_mixed_binary_1(cls, name: str) -> str:
        return f"indicator_bilinear_mixed_binary_1_{name}"

    @classmethod
    def con_name_indicator_bilinear_mixed_binary_2(cls, name: str) -> str:
        return f"indicator_bilinear_mixed_binary_2_{name}"

    @classmethod
    def con_name_indicator_bilinear_mixed_binary_3(cls, name: str) -> str:
        return f"indicator_bilinear_mixed_binary_3_{name}"

    @classmethod
    def con_name_indicator_bilinear_mixed_binary_4(cls, name: str) -> str:
        return f"indicator_bilinear_mixed_binary_4_{name}"

    # --- Absolute Value Reformulations ---

    @classmethod
    def var_name_binary_abs_reformulation(cls, name: str) -> str:
        return f"abs_bin_{name}"

    @classmethod
    def con_name_abs_reformulation_negative(cls, name: str) -> str:
        return f"abs_reform_neg_{name}"

    @classmethod
    def con_name_abs_reformulation_positive(cls, name: str) -> str:
        return f"abs_reform_pos_{name}"

    @classmethod
    def con_name_abs_reformulation_negative_big_m(cls, name: str) -> str:
        return f"abs_reform_neg_big_m_{name}"

    @classmethod
    def con_name_abs_reformulation_positive_big_m(cls, name: str) -> str:
        return f"abs_reform_pos_big_m_{name}"

    # --- PWL Reformulations ---

    @classmethod
    def pwl_method_multiple_choice(cls) -> str:
        return "multiple_choice"

    @classmethod
    def pwl_method_delta(cls) -> str:
        return "delta"

    @classmethod
    def pwl_method_none(cls) -> str:
        return "none"

    @classmethod
    def var_name_pwl_multiple_choice_binary(
        cls, name: str, breakpoint_index: int
    ) -> str:
        return f"mcm_breakpoint_{name}_{breakpoint_index}"

    @classmethod
    def var_name_pwl_delta_binary(cls, name: str, breakpoint_index: int) -> str:
        return f"dm_breakpoint_{name}_{breakpoint_index}"

    @classmethod
    def var_name_pwl_multiple_choice_continuous(
        cls, name: str, breakpoint_index: int
    ) -> str:
        return f"mcm_continuous_{name}_{breakpoint_index}"

    @classmethod
    def var_name_pwl_delta_continuous(cls, name: str, breakpoint_index: int) -> str:
        return f"dm_continuous_{name}_{breakpoint_index}"

    @classmethod
    def con_name_pwl_multiple_choice_variable_link_continuous(cls, name: str) -> str:
        return f"mcm_varlink_cont_{name}"

    @classmethod
    def con_name_pwl_delta_variable_link_continuous(cls, name: str) -> str:
        return f"dm_varlink_cont_{name}"

    @classmethod
    def con_name_pwl_multiple_choice_sos(cls, name: str) -> str:
        return f"mcm_sos_{name}"

    @classmethod
    def con_name_pwl_multiple_choice_interval_lb(
        cls, name: str, breakpoint_index: int
    ) -> str:
        return f"mcm_interval_lb_{name}_{breakpoint_index}"

    @classmethod
    def con_name_pwl_multiple_choice_interval_ub(
        cls, name: str, breakpoint_index: int
    ) -> str:
        return f"mcm_interval_ub_{name}_{breakpoint_index}"

    @classmethod
    def con_name_pwl_delta_interval_lb(cls, name: str, breakpoint_index: int) -> str:
        return f"dm_interval_lb_{name}_{breakpoint_index}"

    @classmethod
    def con_name_pwl_delta_interval_ub(cls, name: str, breakpoint_index: int) -> str:
        return f"dm_interval_ub_{name}_{breakpoint_index}"

    @classmethod
    def con_name_pwl_multiple_choice_approximation(cls, name: str) -> str:
        return f"mcm_approx_{name}"

    @classmethod
    def con_name_pwl_delta_approximation(cls, name: str) -> str:
        return f"dm_approx_{name}"

    @classmethod
    def con_name_pwl_multiple_choice_underestimation(cls, name: str) -> str:
        return f"mcm_under_{name}"

    @classmethod
    def con_name_pwl_multiple_choice_overestimation(cls, name: str) -> str:
        return f"mcm_over_{name}"

    @classmethod
    def con_name_pwl_delta_underestimation(cls, name: str) -> str:
        return f"dm_under_{name}"

    @classmethod
    def con_name_pwl_delta_overestimation(cls, name: str) -> str:
        return f"dm_over_{name}"

    @classmethod
    def con_name_pwc_multiple_choice_multilinear(
        cls, name: str, implying_variable_indices: list[int]
    ) -> str:
        return f"mcm_pwc_{name}_{'_'.join(map(str, implying_variable_indices))}"
