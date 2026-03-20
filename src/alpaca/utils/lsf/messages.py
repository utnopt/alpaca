# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, too-many-public-methods
"""
@authors: kuen,
"""
import datetime


class Messages:
    """Errors, warnings, and informational messages."""

    @classmethod
    def _get_timestamp(cls) -> str:
        """Generates a formatted timestamp string."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Errors ---

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
    def error_exception_occurred(cls, exception: Exception) -> str:
        return f"Exception occurred: {exception}"

    @classmethod
    def error_path_exists(cls, path: str) -> str:
        return f"Path {path} already exists and is not a directory."

    @classmethod
    def error_input_choice_invalid(cls, choice: str, valid_choices: list[str]) -> str:
        return (
            f"Input choice '{choice}' is invalid. "
            f"Valid choices are: {', '.join(valid_choices)}."
        )

    @classmethod
    def error_one_variable_in_product(cls) -> str:
        return "A product expression must have at least two variables."

    @classmethod
    def error_subclasses_must_implement_method(cls) -> str:
        return "Subclass must implement method."

    @classmethod
    def error_nonlinearity_not_implemented(cls, nonlinearity_type: str) -> str:
        return f"Nonlinearity '{nonlinearity_type}' not implemented."

    @classmethod
    def error_filter_infinite_bounds_discretized_var(cls, variable_name: str) -> str:
        return f"Variable {variable_name} is discretized but has infinite bounds."

    @classmethod
    def error_filter_too_many_variables(
        cls, num_variables: int, max_variables: int
    ) -> str:
        return (f"Model has {num_variables} variables, "
                f"which exceeds the maximum of {max_variables} allowed.")

    @classmethod
    def error_filter_no_bilinear_expressions(cls) -> str:
        return "No bilinear expressions."

    @classmethod
    def error_filter_no_mpip_instances(cls) -> str:
        return "No MPIP instances."

    # --- Warnings ---

    @classmethod
    def warning_expression_type_not_supported(cls, expr_type: str) -> str:
        return f"Expression type '{expr_type}' is not supported."

    @classmethod
    def warning_could_not_remove_log_file_handler(
        cls, handler: str, ex: Exception
    ) -> str:
        return (
            f"Couldn't remove previous log file handler with"
            f" files {handler} due to error {ex}"
        )

    @classmethod
    def warning_mpip_features_disabled_for_pwl_method_none(cls) -> str:
        return (
            "MPIP features are disabled when pwl_method is set to 'none'. "
            "Set pwl_method to 'multiple_choice' or 'delta' to enable MPIP features."
        )

    @classmethod
    def warning_mpip_features_enabled_only_for_pwl_method_mc(cls) -> str:
        return (
            "Some MPIP features are only enabled when pwl_method is set to 'multiple_choice'. "
            "Set pwl_method to 'multiple_choice' to enable these MPIP features."
        )

    @classmethod
    def warning_not_implemented_mpip_feature_for_pwl_method(
        cls, pwl_method: str
    ) -> str:
        return (
            f"MPIP feature not implemented for pwl_method '{pwl_method}'. "
            "Set pwl_method to 'multiple_choice' to enable special MPIP features."
        )

    # --- Info ---

    @classmethod
    def info_init_mpip_handler(cls) -> str:
        return (
            f"[{cls._get_timestamp()}] Extract multipartite implication information..."
        )

    @classmethod
    def info_init_mip_model_buildup(cls) -> str:
        return f"[{cls._get_timestamp()}] Build MIP model..."

    @classmethod
    def info_init_solver(cls) -> str:
        return f"[{cls._get_timestamp()}] Solve MIP ..."

    @classmethod
    def info_init_stair_locatelli(cls) -> str:
        return f"[{cls._get_timestamp()}] Initialize stair Locatelli cuts..."

    @classmethod
    def info_total_stair_locatelli_cuts_added(cls, count: int) -> str:
        return f"[{cls._get_timestamp()}] Total stair Locatelli cuts added: {count}"

    @classmethod
    def info_optimization_finished(cls, runtime: float) -> str:
        return (
            f"[{cls._get_timestamp()}] Optimization finished in {runtime:.2f} seconds."
        )

    @classmethod
    def info_read_osil_data(cls) -> str:
        return f"[{cls._get_timestamp()}] Read osil data..."

    @classmethod
    def info_propagate_bounds(cls) -> str:
        return f"[{cls._get_timestamp()}] Propagate bounds..."

    @classmethod
    def info_create_piecewise_linear_relaxation(cls) -> str:
        return f"[{cls._get_timestamp()}] Create piecewise linear relaxation..."

    @classmethod
    def info_generate_breakpoints(cls) -> str:
        return f"[{cls._get_timestamp()}] Generate breakpoints..."

    @classmethod
    def info_grow_expression_graphs(cls) -> str:
        return f"[{cls._get_timestamp()}] Grow expression graphs..."

    @classmethod
    def info_reformulate_multilinear_expressions(cls) -> str:
        return f"[{cls._get_timestamp()}] Reformulate multilinear expressions..."

    @classmethod
    def info_add_mccormick_envelopes(cls) -> str:
        return f"[{cls._get_timestamp()}] Add McCormick envelopes..."

    @classmethod
    def info_save_settings_json(cls) -> str:
        return (
            f"[{cls._get_timestamp()}]"
            f" The settings are saved as JSON-format to the export folder"
        )

    @classmethod
    def info_apply_obbt(cls) -> str:
        return f"[{cls._get_timestamp()}] Apply optimization-based bound tightening (OBBT)..."
