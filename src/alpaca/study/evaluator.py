# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger


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
    }

    GEOMETRIC_MEAN_SHIFT = 10.0

    def __init__(self, csv_path: str) -> None:
        """Initializes the evaluator with a CSV file path.

        Args:
            csv_path: Path to the CSV results file.
        """
        self.csv_path = csv_path
        self.data = StudyData()
        self._load_csv()
        self._filter_complete_instances()

    def _load_csv(self) -> None:
        """Loads and parses the CSV file into the data structure."""
        csv_file = Path(self.csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.data.headers = reader.fieldnames or []

            for row in reader:
                parsed_row = self._parse_row(row)
                self.data.rows.append(parsed_row)

                instance = parsed_row.get(lsf.stats_instance_name(), "")
                config = parsed_row.get(lsf.stats_config_name(), "")
                self.data.data_matrix[(instance, config)] = parsed_row

        all_instances = sorted(
            {row.get(lsf.stats_instance_name(), "") for row in self.data.rows}
        )
        self.data.configs = sorted(
            {row.get(lsf.stats_config_name(), "") for row in self.data.rows}
        )
        self.data.instances = all_instances

        logger.info(
            "Loaded %d rows: %d instances x %d configs (before filtering)",
            len(self.data.rows),
            len(self.data.instances),
            len(self.data.configs),
        )

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
            if row.get(lsf.stats_instance_name(), "") in complete_instances
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
        if value is None or value.strip() == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def get_column_values_for_config(self, column: str, config: str) -> list[float]:
        """Extracts numeric values for a column filtered by configuration.

        Args:
            column: Column name to extract.
            config: Configuration name to filter by.

        Returns:
            List of non-None numeric values.
        """
        values = []
        for instance in self.data.instances:
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

    def compute_mean(self, column: str, config: str) -> float | None:
        """Computes arithmetic mean for a column and configuration.

        Args:
            column: Column name.
            config: Configuration name.

        Returns:
            Mean value or None if no data.
        """
        values = self.get_column_values_for_config(column, config)
        if not values:
            return None
        return float(np.mean(values))

    def compute_shifted_geometric_mean(
        self, column: str, config: str, shift: float | None = None
    ) -> float | None:
        """Computes shifted geometric mean for a column and configuration.

        Formula: exp(mean(log(values + shift))) - shift

        Args:
            column: Column name.
            config: Configuration name.
            shift: Shift value. Defaults to GEOMETRIC_MEAN_SHIFT.

        Returns:
            Shifted geometric mean or None if no data.
        """
        if shift is None:
            shift = self.GEOMETRIC_MEAN_SHIFT

        values = self.get_column_values_for_config(column, config)
        if not values:
            return None

        shifted_values = [v + shift for v in values]
        if any(v <= 0 for v in shifted_values):
            return None

        log_mean = np.mean(np.log(shifted_values))
        return float(np.exp(log_mean) - shift)

    def compute_statistics_for_config(
        self, column: str, config: str
    ) -> dict[str, float | None]:
        """Computes comprehensive statistics for a column and configuration.

        Args:
            column: Column name.
            config: Configuration name.

        Returns:
            Dictionary with mean, std, min, max, median, q25, q75 values.
        """
        values = self.get_column_values_for_config(column, config)

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
        self, time_column: str = None
    ) -> dict[str, list[tuple[float, int]]]:
        """Computes cumulative instances solved over time for each config.

        Args:
            time_column: Column containing solving times.

        Returns:
            Dictionary mapping config to list of (time, cumulative_count) tuples.
        """
        if time_column is None:
            time_column = lsf.stats_solving_time()

        result = {}

        for config in self.data.configs:
            times = []
            for instance in self.data.instances:
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
        self, columns: list[str]
    ) -> dict[str, dict[str, dict[str, float | None]]]:
        """Creates pivot data for instances with multiple columns per config.

        Args:
            columns: List of column names to include.

        Returns:
            Nested dict: {instance: {config: {column: value}}}.
        """
        pivot = {}
        for instance in self.data.instances:
            pivot[instance] = {}
            for config in self.data.configs:
                pivot[instance][config] = {}
                row_data = self.data.data_matrix.get((instance, config), {})
                for column in columns:
                    pivot[instance][config][column] = row_data.get(column)
        return pivot

    def get_config_aggregated_data(
        self,
        columns: list[str],
        use_shifted_geom_mean: list[str] | None = None,
    ) -> dict[str, dict[str, float | None]]:
        """Aggregates data by configuration with specified aggregation methods.

        Args:
            columns: List of column names to aggregate.
            use_shifted_geom_mean: Columns to use shifted geometric mean.

        Returns:
            Nested dict: {config: {column: aggregated_value}}.
        """
        if use_shifted_geom_mean is None:
            use_shifted_geom_mean = []

        result = {}
        for config in self.data.configs:
            result[config] = {}
            for column in columns:
                if column in use_shifted_geom_mean:
                    result[config][column] = self.compute_shifted_geometric_mean(
                        column, config
                    )
                else:
                    result[config][column] = self.compute_mean(column, config)
        return result

    def get_boxplot_data_for_config(
        self, column: str
    ) -> dict[str, dict[str, float | None]]:
        """Gets boxplot statistics for a column across all configurations.

        Args:
            column: Column name.

        Returns:
            Dictionary mapping config to boxplot statistics.
        """
        result = {}
        for config in self.data.configs:
            result[config] = self.compute_statistics_for_config(column, config)
        return result

    def get_instance_data(
        self, columns: list[str]
    ) -> dict[str, dict[str, float | None]]:
        """Gets instance-level data for specified columns (first config's values).

        Note: For instance-specific metrics that don't vary by config.

        Args:
            columns: List of column names.

        Returns:
            Dictionary mapping instance to column values.
        """
        result = {}
        for instance in self.data.instances:
            result[instance] = {}
            first_config = self.data.configs[0] if self.data.configs else None
            if first_config:
                row_data = self.data.data_matrix.get((instance, first_config), {})
                for column in columns:
                    result[instance][column] = row_data.get(column)
        return result
