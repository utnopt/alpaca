# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, no-member
"""
@authors: kuen,
"""


class LocalizedStringFactory:  # pylint: disable=too-many-public-methods
    """
    Localized string factory with english as default language.
    """

    @classmethod
    def project_name(cls) -> str:
        return "alpaca"

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
    def objective_var(cls) -> str:
        return "obj_var"

    @classmethod
    def error_duplicate_con_name(cls, name: str) -> str:
        return f"Duplicate constraint name {name}."

    @classmethod
    def error_duplicate_var_name(cls, name: str) -> str:
        return f"Duplicate variable name {name}."

    @classmethod
    def error_not_implemented_delta_method(cls) -> str:
        return "Delta method not implemented yet."

    @classmethod
    def var_type_binary(cls) -> str:
        return "B"

    @classmethod
    def var_type_integer(cls) -> str:
        return "I"

    @classmethod
    def var_type_continuous(cls) -> str:
        return "C"

    @classmethod
    def info_init_mpip_handler(cls) -> str:
        return "Initializing Multipartite Implication Polytope handler."

    @classmethod
    def info_init_mip_model_buildup(cls) -> str:
        return "Initializing MIP model build-up."

    @classmethod
    def info_init_solver(cls) -> str:
        return "Start Solving."

    @classmethod
    def info_optimization_finished(cls, runtime: int) -> str:
        return f"Optimization finished in {runtime:.2f} seconds."

    @classmethod
    def error_exception_occurred(cls, exception: Exception) -> str:
        return f"Exception occurred: {exception}"

    @classmethod
    def mpip_id(cls, counter: int) -> str:
        return f"mpip_{counter}"

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
    def mpip_interval_lp_var_name(cls, name: str) -> str:
        return f"x_{name}"

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
    def var_name_pwl_multiple_choice_binary(
        cls, name: str, breakpoint_index: int
    ) -> str:
        return f"mcm_breakpoint_{name}_{breakpoint_index}"

    @classmethod
    def var_name_pwl_multiple_choice_continuous(
        cls, name: str, breakpoint_index: int
    ) -> str:
        return f"mcm_continuous_{name}_{breakpoint_index}"

    @classmethod
    def con_name_pwl_multiple_choice_variable_link_continuous(cls, name: str) -> str:
        return f"mcm_varlink_cont_{name}"

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
    def con_name_pwl_multiple_choice_approximation(cls, name: str) -> str:
        return f"mcm_approx_{name}"

    @classmethod
    def con_name_pwl_multiple_choice_underestimation(cls, name: str) -> str:
        return f"mcm_under_{name}"

    @classmethod
    def con_name_pwl_multiple_choice_overestimation(cls, name: str) -> str:
        return f"mcm_over_{name}"

    @classmethod
    def con_name_pwc_multiple_choice_multilinear(
        cls, name: str, implying_variable_indices: list[int]
    ) -> str:
        return f"mcm_pwc_{name}_{'_'.join(map(str, implying_variable_indices))}"

    @classmethod
    def solver_name_scip(cls) -> str:
        return "scip"

    @classmethod
    def solver_name_gurobi(cls) -> str:
        return "gurobi"

    @classmethod
    def scip_separator_name_mpip(cls) -> str:
        return "mpip"

    @classmethod
    def scip_separator_description_mpip(cls) -> str:
        return "Multipartite Implication Polytope Separator"

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
    def warning_expression_type_not_supported(cls, expr_type: str) -> str:
        return f"Expression type '{expr_type}' is not supported."

    @classmethod
    def error_path_exists(cls, path: str) -> str:
        return f"Path {path} already exists and is not a directory."

    @classmethod
    def warning_could_not_remove_log_file_handler(
        cls, handler: str, ex: Exception
    ) -> str:
        return (
            f"Couldn't remove previous log file handler with"
            f" files {handler} due to error {ex}"
        )

    @classmethod
    def log_rotation_type_size(cls) -> str:
        return "size"

    @classmethod
    def log_rotation_type_time(cls) -> str:
        return "time"

    @classmethod
    def osil_file_suffix(cls) -> str:
        return ".osil"

    @classmethod
    def osil_input_format_xml(cls) -> str:
        return "xml"

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
    def osil_inf_neg(cls) -> str:
        return "-INF"

    @classmethod
    def osil_inf_pos(cls) -> str:
        return "INF"

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
    def osil_attr_lb(cls) -> str:
        return "lb"

    @classmethod
    def osil_attr_ub(cls) -> str:
        return "ub"

    @classmethod
    def osil_attr_type(cls) -> str:
        return "type"

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
    def osil_attr_coef(cls) -> str:
        return "coef"

    @classmethod
    def osil_attr_idx(cls) -> str:
        return "idx"

    @classmethod
    def osil_tag_el(cls) -> str:
        return "el"

    @classmethod
    def osil_attr_mult(cls) -> str:
        return "mult"

    @classmethod
    def osil_attr_incr(cls) -> str:
        return "incr"

    @classmethod
    def osil_tag_linear_constraint_coefficients(cls) -> str:
        return "linearConstraintCoefficients"

    @classmethod
    def osil_attr_number_of_values(cls) -> str:
        return "numberOfValues"

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
    def osil_attr_idx_one(cls) -> str:
        return "idxOne"

    @classmethod
    def osil_attr_idx_two(cls) -> str:
        return "idxTwo"

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
    def osil_tag_nonlinear_expressions(cls) -> str:
        return "nonlinearExpressions"

    @classmethod
    def osil_tag_nl(cls) -> str:
        return "nl"

    @classmethod
    def file_mode_read(cls) -> str:
        return "r"

    @classmethod
    def file_encoding_utf8(cls) -> str:
        return "utf-8"

    @classmethod
    def pwl_method_multiple_choice(cls) -> str:
        return "multiple_choice"

    @classmethod
    def pwl_method_delta(cls) -> str:
        return "delta"

    @classmethod
    def mip_solver_parameter_time_limit(cls, solver_name: str) -> str:
        if solver_name == cls.solver_name_scip():
            return "limits/time"
        if solver_name == cls.solver_name_gurobi():
            return "TimeLimit"
        return "time_limit"

    @classmethod
    def mip_solver_parameter_thread_limit(cls, solver_name: str) -> str:
        if solver_name == cls.solver_name_scip():
            return "parallel/maxnthreads"
        if solver_name == cls.solver_name_gurobi():
            return "Threads"
        return "thread_limit"

    @classmethod
    def mip_solver_parameter_seed(cls, solver_name: str) -> str:
        if solver_name == cls.solver_name_scip():
            return "randomization/randomseedshift"
        if solver_name == cls.solver_name_gurobi():
            return "Seed"
        return "seed"

    @classmethod
    def scip_result_tag(cls) -> str:
        return "result"

    @classmethod
    def scip_parameter_reoptimization(cls) -> str:
        return "reoptimization/enable"

    @classmethod
    def gurobi_parameter_output_flag(cls) -> str:
        return "OutputFlag"

    @classmethod
    def gurobi_model_attribute_sense(cls) -> str:
        return "ModelSense"

    @classmethod
    def error_input_choice_invalid(cls, choice: str, valid_choices: list[str]) -> str:
        return (
            f"Input choice '{choice}' is invalid. "
            f"Valid choices are: {', '.join(valid_choices)}."
        )

    @classmethod
    def empty_string(cls) -> str:
        return ""

    @classmethod
    def numpy_infinity(cls) -> str:
        return "inf"

    @classmethod
    def representative_variable_name(cls, name: str) -> str:
        return f"rep_{name}"

    @classmethod
    def helper_variable_name(cls, name: str) -> str:
        return f"helper_{name}"

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

    @classmethod
    def linear_expression_bilinear_to_sum_of_squares(cls, name: str) -> str:
        return f"bilinear_to_sum_of_squares_{name}"

    @classmethod
    def linear_expression_bilinear_to_sum_of_squares_helper(cls, name: str) -> str:
        return f"bilinear_to_sum_of_squares_helper_{name}"

    @classmethod
    def expression_hash_square(cls, name: str) -> str:
        return f"square_{name}"

    @classmethod
    def expression_hash_generic_nonlinear(cls, name: str, nonlinearity: str) -> str:
        return f"{nonlinearity}_{name}"

    @classmethod
    def error_one_variable_in_product(cls) -> str:
        return "A product expression must have at least two variables."

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

    @classmethod
    def error_subclasses_must_implement_method(cls) -> str:
        return "Subclass must implement method."

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
