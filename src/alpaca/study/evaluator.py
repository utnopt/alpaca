# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import csv
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger
import alpaca.settings as s


class InstanceFilter(Enum):
    """Filter types for instance selection in comparisons."""

    NONE = auto()
    """No filtering - use all complete instances."""

    ALL_REACHED_ROOT = auto()
    """Only instances where all configs reached the root node (not by time limit)."""

    NON_EMPTY_BILINEAR_DOMAIN = auto()
    """Only instances where all configs have non-empty bilinear domains."""

    ALL_TERMINATED = auto()
    """Only instances where all configs terminated (not by time limit)."""

    ALL_TERMINATED_CONSISTENT = auto()
    """All terminated + consistent solution values across configs."""

    BRANCH_AND_BOUND = auto()
    """Instances where all configs have more than one branch and bound node."""


@dataclass
class FilterResult:
    """Result of applying instance filters.

    Attributes:
        instances: List of instances that passed the filter.
        warnings: List of warning messages for discarded instances.
    """

    instances: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class StudyData:
    """Container for loaded and processed study data.

    Attributes:
        headers: List of column names.
        rows: List of dictionaries, each representing a data row.
        instances: Sorted list of unique instance names.
        configs: Sorted list of unique configuration names.
        data_matrix: Dictionary mapping (instance, config) to row data.
    """

    headers: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    instances: list[str] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)
    data_matrix: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    @property
    def df(self) -> pd.DataFrame:
        """Converts the data to a pandas DataFrame for analysis."""
        return pd.DataFrame(self.rows, columns=self.headers)


class StudyEvaluator:
    """Evaluates study results from CSV files.

    This class loads CSV data, filters for complete instances,
    computes statistics per configuration, and provides data extraction utilities.
    """

    NUMERIC_COLUMNS = {
        lsf.stats_solving_time(),
        lsf.stats_nr_nodes(),
        lsf.stats_solution_value(),
        lsf.stats_mip_gap(),
        lsf.stats_final_nr_vars(),
        lsf.stats_final_nr_constraints(),
        lsf.stats_presolved_nr_vars(),
        lsf.stats_presolved_nr_constraints(),
        lsf.stats_presolved_nr_nonzeros(),
        lsf.stats_root_solution_value(),
        lsf.stats_root_solving_time(),
        lsf.stats_build_time(),
        lsf.stats_original_nr_variables(),
        lsf.stats_original_nr_constraints(),
        lsf.stats_original_nr_bilinear_expressions(),
        lsf.stats_original_nr_bilinear_binary_expressions(),
        lsf.stats_original_nr_mixed_binary_expressions(),
        lsf.stats_original_nr_multilinear_expressions(),
        lsf.stats_original_nr_one_dim_expressions(),
        lsf.stats_pwl_nr_variables(),
        lsf.stats_pwl_nr_constraints(),
        lsf.stats_pwl_nr_bilinear_expressions(),
        lsf.stats_pwl_nr_bilinear_binary_expressions(),
        lsf.stats_pwl_nr_mixed_binary_expressions(),
        lsf.stats_pwl_nr_multilinear_expressions(),
        lsf.stats_pwl_nr_one_dim_expressions(),
        lsf.stats_locatelli_domain_volume_polygon(),
        lsf.stats_locatelli_domain_volume_polytope(),
        lsf.stats_locatelli_nr_cuts(),
        lsf.stats_mpip_nr_instances(),
        lsf.stats_mpip_ratio(),
        lsf.stats_mpip_nr_cuts(),
    }

    def __init__(
        self,
        csv_path: str,
        time_limit: int = 3600,
        mean_trim: float = 0.05,
        base_config: str = "b_a_s_e",
    ) -> None:
        """Initializes the evaluator with a CSV file path.

        Args:
            csv_path: Path to the CSV results file.
            time_limit: Time limit for runs (in seconds) to identify timeouts.
            mean_trim: Trimming of mean values for outlier filtering.
            base_config: Base config name to identify in the CSV.
        """
        self.csv_path = csv_path
        self.time_limit = time_limit
        self.mean_trim = mean_trim
        self.base_config = base_config
        self.data = StudyData()

        # Cached filter results
        self._filter_cache: dict[InstanceFilter, FilterResult] = {}

        self._load_csv()
        self._filter_complete_instances()
        self._filter_converged_inconsistent_instances()
        self._filter_nones_in_solving_time()
        self._compute_filter_results()

    def _load_csv(self) -> None:
        """Loads and parses the CSV file into the data structure."""
        csv_file = Path(self.csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with open(
            self.csv_path, lsf.file_mode_read(), encoding=lsf.file_encoding_utf8()
        ) as file:
            reader = csv.DictReader(file)
            self.data.headers = reader.fieldnames or []

            for row in reader:
                parsed_row = self._parse_row(row)
                self.data.rows.append(parsed_row)

                instance = parsed_row.get(lsf.stats_instance_name(), lsf.empty_string())
                config = parsed_row.get(lsf.stats_config_name(), lsf.empty_string())
                self._cap_solving_time(parsed_row)
                self.data.data_matrix[(instance, config)] = parsed_row

        all_instances = sorted(
            {
                row.get(lsf.stats_instance_name(), lsf.empty_string())
                for row in self.data.rows
            }
        )
        self.data.configs = sorted(
            {
                row.get(lsf.stats_config_name(), lsf.empty_string())
                for row in self.data.rows
            }
        )
        self.data.instances = all_instances

        logger.info(
            "Loaded %d rows: %d instances x %d configs (before filtering)",
            len(self.data.rows),
            len(self.data.instances),
            len(self.data.configs),
        )

    def _cap_solving_time(self, row: dict[str, Any]) -> None:
        """Caps the solving time in the row to the specified time limit."""
        solving_time = row.get(lsf.stats_solving_time())
        if isinstance(solving_time, (int, float)) and solving_time > self.time_limit:
            row[lsf.stats_solving_time()] = self.time_limit

    def _filter_complete_instances(self) -> None:
        """Filters to keep only instances where all configurations have results."""
        complete_instances = []

        for instance in self.data.instances:
            has_all_configs = all(
                (instance, config) in self.data.data_matrix
                for config in self.data.configs
            )
            if has_all_configs:
                complete_instances.append(instance)

        removed_count = len(self.data.instances) - len(complete_instances)
        self.data.instances = complete_instances

        filtered_rows = [
            row
            for row in self.data.rows
            if row.get(lsf.stats_instance_name(), lsf.empty_string())
            in complete_instances
        ]
        self.data.rows = filtered_rows

        filtered_matrix = {
            key: value
            for key, value in self.data.data_matrix.items()
            if key[0] in complete_instances
        }
        self.data.data_matrix = filtered_matrix

        logger.info(
            "Filtered to %d complete instances (removed %d incomplete)",
            len(complete_instances),
            removed_count,
        )

    def _filter_converged_inconsistent_instances(self) -> None:
        """Filters out instances that converged but have inconsistent solution values."""
        consistent_or_terminated_instances = []
        for instance in self.data.instances:
            is_terminated_by_time_limit = any(
                self._is_terminated_by_time_limit(instance, config)
                for config in self.data.configs
            )
            if is_terminated_by_time_limit:
                consistent_or_terminated_instances.append(instance)
            elif self._compute_consistent_solution_filter([instance]).instances:
                consistent_or_terminated_instances.append(instance)

        removed_count = len(self.data.instances) - len(
            consistent_or_terminated_instances
        )
        self.data.instances = consistent_or_terminated_instances

        filtered_rows = [
            row
            for row in self.data.rows
            if row.get(lsf.stats_instance_name(), lsf.empty_string())
            in consistent_or_terminated_instances
        ]
        self.data.rows = filtered_rows

        filtered_matrix = {
            key: value
            for key, value in self.data.data_matrix.items()
            if key[0] in consistent_or_terminated_instances
        }
        self.data.data_matrix = filtered_matrix

        logger.info(
            "Filtered to %d consistent instances (removed %d inconsistent)",
            len(consistent_or_terminated_instances),
            removed_count,
        )

    def _filter_nones_in_solving_time(self) -> None:
        """Filters out instances that have None in solving time for any config."""
        valid_instances = []
        for instance in self.data.instances:
            has_none_solving_time = any(
                self.data.data_matrix.get((instance, config), {}).get(
                    lsf.stats_solving_time()
                )
                is None
                for config in self.data.configs
            )
            if not has_none_solving_time:
                valid_instances.append(instance)

        removed_count = len(self.data.instances) - len(valid_instances)
        self.data.instances = valid_instances

        filtered_rows = [
            row
            for row in self.data.rows
            if row.get(lsf.stats_instance_name(), lsf.empty_string()) in valid_instances
        ]
        self.data.rows = filtered_rows

        filtered_matrix = {
            key: value
            for key, value in self.data.data_matrix.items()
            if key[0] in valid_instances
        }
        self.data.data_matrix = filtered_matrix

        logger.info(
            "Filtered to %d instances with valid solving time (removed %d with None)",
            len(valid_instances),
            removed_count,
        )

    def _compute_filter_results(self) -> None:
        """Computes and caches filter results for all filter types."""
        # NONE filter - all complete instances
        self._filter_cache[InstanceFilter.NONE] = FilterResult(
            instances=list(self.data.instances),
            warnings=[],
        )
        # EMPTY_BILINEAR_DOMAIN filter
        empty_bilinear_domain_result = self._compute_empty_bilinear_domain_filter()
        self._filter_cache[InstanceFilter.NON_EMPTY_BILINEAR_DOMAIN] = (
            empty_bilinear_domain_result
        )

        # ALL_REACHED_ROOT filter
        reached_root_result = self._compute_all_reached_root_filter()
        self._filter_cache[InstanceFilter.ALL_REACHED_ROOT] = reached_root_result

        # ALL_TERMINATED filter
        terminated_result = self._compute_all_terminated_filter()
        self._filter_cache[InstanceFilter.ALL_TERMINATED] = terminated_result

        # ALL_TERMINATED_CONSISTENT filter
        consistent_result = self._compute_consistent_solution_filter(
            terminated_result.instances
        )
        self._filter_cache[InstanceFilter.ALL_TERMINATED_CONSISTENT] = consistent_result

        # BRANCH_AND_BOUND filter
        branch_and_bound_result = self._compute_branch_and_bound_filter(
            consistent_result.instances
        )
        self._filter_cache[InstanceFilter.BRANCH_AND_BOUND] = branch_and_bound_result

        # Log filter statistics
        logger.info(
            "Filter results - NONE: %d, ALL_TERMINATED: %d, ALL_TERMINATED_CONSISTENT: %d",
            len(self._filter_cache[InstanceFilter.NONE].instances),
            len(self._filter_cache[InstanceFilter.ALL_TERMINATED].instances),
            len(self._filter_cache[InstanceFilter.ALL_TERMINATED_CONSISTENT].instances),
        )

        # Log warnings
        for warning in self._filter_cache[
            InstanceFilter.ALL_TERMINATED_CONSISTENT
        ].warnings:
            logger.warning(warning)

    def _is_terminated_by_time_limit(self, instance: str, config: str) -> bool:
        """Checks if a run was terminated by the time limit.

        Uses solving time to determine termination status.

        Args:
            instance: Instance name.
            config: Configuration name.

        Returns:
            True if terminated by time limit, False otherwise.
        """
        row_data = self.data.data_matrix.get((instance, config), {})
        solving_time = row_data.get(lsf.stats_solving_time())

        return solving_time == self.time_limit

    def _compute_empty_bilinear_domain_filter(self) -> FilterResult:
        """Computes instances where all configs have non-empty bilinear domains.

        Returns:
            FilterResult with instances that have non-empty bilinear domains for all configs.
        """
        non_empty_domain_instances = []
        warnings = []

        for instance in self.data.instances:
            no_empty_domain = True

            for config in self.data.configs:
                row_data = self.data.data_matrix.get((instance, config), {})
                polygon_domain = row_data.get(
                    lsf.stats_locatelli_domain_volume_polygon()
                )
                polytope_domain = row_data.get(
                    lsf.stats_locatelli_domain_volume_polytope()
                )
                if (
                    abs(polytope_domain) < s.StaticSettings.feasibility_tolerance
                    or abs(polygon_domain) < s.StaticSettings.feasibility_tolerance
                ):
                    no_empty_domain = False
                    break

            if no_empty_domain:
                non_empty_domain_instances.append(instance)

        return FilterResult(instances=non_empty_domain_instances, warnings=warnings)

    def _compute_all_reached_root_filter(self) -> FilterResult:
        """Computes instances where all configs reached the root node.

        Returns:
            FilterResult with instances that reached root for all configs.
        """
        reached_root_instances = []
        warnings = []

        for instance in self.data.instances:
            all_reached_root = True

            for config in self.data.configs:
                row_data = self.data.data_matrix.get((instance, config), {})
                root_solution = row_data.get(lsf.stats_root_solution_value())
                if root_solution == -s.StaticSettings.infinity:
                    all_reached_root = False
                    break

            if all_reached_root:
                reached_root_instances.append(instance)

        return FilterResult(instances=reached_root_instances, warnings=warnings)

    def _compute_all_terminated_filter(self) -> FilterResult:
        """Computes instances where all configs terminated successfully.

        Returns:
            FilterResult with instances that terminated for all configs.
        """
        terminated_instances = []
        warnings = []

        for instance in self.data.instances:
            all_terminated = True
            time_limit_configs = []

            for config in self.data.configs:
                if self._is_terminated_by_time_limit(instance, config):
                    all_terminated = False
                    time_limit_configs.append(config)

            if all_terminated:
                terminated_instances.append(instance)

        return FilterResult(instances=terminated_instances, warnings=warnings)

    def _compute_consistent_solution_filter(
        self,
        base_instances: list[str],
    ) -> FilterResult:
        """Filters for instances with consistent solution values across configs.

        Args:
            base_instances: Pre-filtered list of instances to check.

        Returns:
            FilterResult with consistent instances and warnings for inconsistent ones.
        """
        consistent_instances = []
        warnings = []

        for instance in base_instances:
            solution_values = []

            for config in self.data.configs:
                row_data = self.data.data_matrix.get((instance, config), {})
                sol_val = row_data.get(lsf.stats_solution_value())
                if sol_val is not None:
                    solution_values.append((config, sol_val))

            if not solution_values:
                # No solution values - cannot verify consistency
                warnings.append(
                    f"Instance '{instance}': No solution values found for any config, "
                    "discarding from comparison."
                )
                continue

            if len(solution_values) < len(self.data.configs):
                # Some configs have no solution value
                missing_configs = set(self.data.configs) - {
                    c for c, _ in solution_values
                }
                warnings.append(
                    f"Instance '{instance}': Missing solution values for configs "
                    f"{missing_configs}, discarding from comparison."
                )
                continue

            # Check consistency
            is_consistent = self._are_solution_values_consistent(
                [v for _, v in solution_values]
            )

            if is_consistent:
                consistent_instances.append(instance)
            else:
                values_str = ", ".join(
                    f"{config}={val:.6g}" for config, val in solution_values
                )
                warnings.append(
                    f"Instance '{instance}': Inconsistent solution values across configs "
                    f"({values_str}), discarding from comparison."
                )

        return FilterResult(instances=consistent_instances, warnings=warnings)

    def _compute_branch_and_bound_filter(
        self,
        base_instances: list[str],
    ) -> FilterResult:
        """Filters for instances where min one config has more than one branch and bound node.

        Args:
            base_instances: Pre-filtered list of instances to check.

        Returns:
            FilterResult with instances that have more than
             one branch and bound node for min one config.
        """
        valid_instances = []
        warnings = []

        for instance in base_instances:
            has_multiple_nodes = False
            for config in self.data.configs:
                row_data = self.data.data_matrix.get((instance, config), {})
                nr_nodes = row_data.get(lsf.stats_nr_nodes())
                if nr_nodes is not None and nr_nodes > 1:
                    has_multiple_nodes = True
                    break

            if has_multiple_nodes:
                valid_instances.append(instance)
            else:
                warnings.append(
                    f"Instance '{instance}': "
                    f"Not all configs have more than one branch and bound node, "
                    "discarding from comparison."
                )

        return FilterResult(instances=valid_instances, warnings=warnings)

    @staticmethod
    def _are_solution_values_consistent(values: list[float]) -> bool:
        """Checks if solution values are consistent within tolerance.

        Args:
            values: List of solution values to compare.

        Returns:
            True if all values are consistent, False otherwise.
        """
        if len(values) <= 1:
            return True

        reference = values[0]

        for val in values[1:]:
            if abs(reference - val) > s.StaticSettings.feasibility_tolerance:
                return False

        return True

    def get_filtered_instances(
        self,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> list[str]:
        """Gets instances that pass the specified filter.

        Args:
            filter_type: Type of filter to apply.

        Returns:
            List of instance names that pass the filter.
        """
        return self._filter_cache[filter_type].instances

    def get_filter_warnings(
        self,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> list[str]:
        """Gets warnings generated during filtering.

        Args:
            filter_type: Type of filter to get warnings for.

        Returns:
            List of warning messages.
        """
        return self._filter_cache[filter_type].warnings

    def _parse_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Parses a CSV row, converting numeric columns to floats.

        Args:
            row: Dictionary of string values from CSV.

        Returns:
            Dictionary with numeric columns converted to float or None.
        """
        parsed = {}
        for key, value in row.items():
            if key in self.NUMERIC_COLUMNS:
                parsed[key] = self._parse_numeric(value)
            else:
                parsed[key] = value
        return parsed

    @staticmethod
    def _parse_numeric(value: str) -> float | None:
        """Parses a string to float, returning None for empty strings.

        Args:
            value: String value to parse.

        Returns:
            Float value or None if empty/invalid.
        """
        if value is None or value.strip() == lsf.empty_string():
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def get_column_values_for_config(
        self,
        column: str,
        config: str,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> list[float]:
        """Extracts numeric values for a column filtered by configuration.

        Args:
            column: Column name to extract.
            config: Configuration name to filter by.
            filter_type: Instance filter to apply.

        Returns:
            List of non-None numeric values.
        """
        instances = self.get_filtered_instances(filter_type)
        values = []
        for instance in instances:
            val = self.data.data_matrix.get((instance, config), {}).get(column)
            if val is not None:
                values.append(val)
        return values

    def get_column_values_for_instance(
        self, column: str, instance: str
    ) -> dict[str, float | None]:
        """Extracts numeric values for a column filtered by instance.

        Args:
            column: Column name to extract.
            instance: Instance name to filter by.

        Returns:
            Dictionary mapping config names to values.
        """
        values = {}
        for config in self.data.configs:
            val = self.data.data_matrix.get((instance, config), {}).get(column)
            values[config] = val
        return values

    def compute_statistics_for_config(
        self,
        column: str,
        config: str,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> dict[str, float | None]:
        """Computes comprehensive statistics for a column and configuration.

        Args:
            column: Column name.
            config: Configuration name.
            filter_type: Instance filter to apply.

        Returns:
            Dictionary with mean, std, min, max, median, q25, q75 values.
        """
        values = self.get_column_values_for_config(column, config, filter_type)

        if not values:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "q25": None,
                "q75": None,
            }

        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    def get_instances_solved_over_time(
        self,
        time_column: str = None,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> dict[str, list[tuple[float, int]]]:
        """Computes cumulative instances solved over time for each config.

        Args:
            time_column: Column containing solving times.
            filter_type: Instance filter to apply.

        Returns:
            Dictionary mapping config to list of (time, cumulative_count) tuples.
        """
        if time_column is None:
            time_column = lsf.stats_solving_time()

        instances = self.get_filtered_instances(filter_type)
        result = {}

        for config in self.data.configs:
            times = []
            for instance in instances:
                val = self.data.data_matrix.get((instance, config), {}).get(time_column)
                if val is not None:
                    times.append(val)

            if not times:
                result[config] = []
                continue

            sorted_times = sorted(times)
            points = []
            for i, time in enumerate(sorted_times):
                points.append((time, i + 1))

            result[config] = points

        return result

    def get_pivot_data(
        self,
        columns: list[str],
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> dict[str, dict[str, dict[str, float | None]]]:
        """Creates pivot data for instances with multiple columns per config.

        Args:
            columns: List of column names to include.
            filter_type: Instance filter to apply.

        Returns:
            Nested dict: {instance: {config: {column: value}}}.
        """
        instances = self.get_filtered_instances(filter_type)
        pivot = {}
        for instance in instances:
            pivot[instance] = {}
            for config in self.data.configs:
                pivot[instance][config] = {}
                row_data = self.data.data_matrix.get((instance, config), {})
                for column in columns:
                    pivot[instance][config][column] = row_data.get(column)
        return pivot

    def get_boxplot_data_for_config(
        self,
        column: str,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> dict[str, dict[str, float | None]]:
        """Gets boxplot statistics for a column across all configurations.

        Args:
            column: Column name.
            filter_type: Instance filter to apply.

        Returns:
            Dictionary mapping config to boxplot statistics.
        """
        result = {}
        for config in self.data.configs:
            result[config] = self.compute_statistics_for_config(
                column, config, filter_type
            )
        return result

    def get_instance_data(
        self,
        columns: list[str],
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> dict[str, dict[str, float | None]]:
        """Gets instance-level data for specified columns (first config's values).

        Note: For instance-specific metrics that don't vary by config.

        Args:
            columns: List of column names.
            filter_type: Instance filter to apply.

        Returns:
            Dictionary mapping instance to column values.
        """
        instances = self.get_filtered_instances(filter_type)
        result = {}
        for instance in instances:
            result[instance] = {}
            first_config = self.data.configs[0] if self.data.configs else None
            if first_config:
                row_data = self.data.data_matrix.get((instance, first_config), {})
                for column in columns:
                    result[instance][column] = row_data.get(column)
        return result

    def get_filter_statistics(self) -> dict[str, dict[str, int]]:
        """Gets statistics about filtered instances.

        Returns:
            Dictionary with filter statistics.
        """
        return {
            "instance_counts": {
                "total_complete": len(self.data.instances),
                "non-empty_bilinear_domain": len(
                    self._filter_cache[
                        InstanceFilter.NON_EMPTY_BILINEAR_DOMAIN
                    ].instances
                ),
                "all_reached_root": len(
                    self._filter_cache[InstanceFilter.ALL_REACHED_ROOT].instances
                ),
                "all_terminated": len(
                    self._filter_cache[InstanceFilter.ALL_TERMINATED].instances
                ),
                "all_terminated_consistent": len(
                    self._filter_cache[
                        InstanceFilter.ALL_TERMINATED_CONSISTENT
                    ].instances
                ),
                "branch_and_bound": len(
                    self._filter_cache[InstanceFilter.BRANCH_AND_BOUND].instances
                ),
            },
            "warning_counts": {
                "all_terminated": len(
                    self._filter_cache[InstanceFilter.ALL_TERMINATED].warnings
                ),
                "all_terminated_consistent": len(
                    self._filter_cache[
                        InstanceFilter.ALL_TERMINATED_CONSISTENT
                    ].warnings
                ),
            },
        }
