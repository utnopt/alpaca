# -*- coding: utf-8 -*-

"""
Evaluator module for analyzing study results.

This module provides functionality to load CSV results, compute statistics,
and prepare data for LaTeX table and plot generation.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alpaca.utils.logger import logger


@dataclass
class ColumnStats:  # pylint: disable=too-many-instance-attributes
    """Statistical summary for a single numeric column.

    Attributes:
        name: Column name.
        count: Number of non-empty values.
        mean: Arithmetic mean.
        std: Standard deviation.
        min_val: Minimum value.
        max_val: Maximum value.
        median: Median value.
        q25: 25th percentile.
        q75: 75th percentile.
    """

    name: str
    count: int = 0
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None


@dataclass
class ConfigComparison:  # pylint: disable=too-many-instance-attributes
    """Comparison statistics between configurations.

    Attributes:
        config_a: Name of the first configuration.
        config_b: Name of the second configuration.
        column: Column being compared.
        wins_a: Number of instances where config_a is better.
        wins_b: Number of instances where config_b is better.
        ties: Number of instances with equal values.
        mean_ratio: Mean ratio of config_a / config_b values.
        geometric_mean_ratio: Geometric mean of ratios.
    """

    config_a: str
    config_b: str
    column: str
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    mean_ratio: float | None = None
    geometric_mean_ratio: float | None = None


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

    This class loads CSV data, computes statistics per configuration,
    and provides comparison utilities between configurations.
    """

    # Columns that should be parsed as numeric values
    NUMERIC_COLUMNS = {
        "solving_time",
        "nr_nodes",
        "solution_value",
        "mip_gap",
        "final_nr_vars",
        "final_nr_constraints",
        "presolved_nr_vars",
        "presolved_nr_constraints",
        "presolved_nr_nonzeros",
        "root_solution_value",
        "root_solving_time",
        "build_time",
        "original_nr_variables",
        "original_nr_constraints",
        "original_nr_bilinear_expressions",
        "original_nr_bilinear_binary_expressions",
        "original_nr_mixed_binary_expressions",
        "original_nr_multilinear_expressions",
        "original_nr_one_dim_expressions",
        "pwl_nr_variables",
        "pwl_nr_constraints",
        "pwl_nr_bilinear_expressions",
        "pwl_nr_bilinear_binary_expressions",
        "pwl_nr_mixed_binary_expressions",
        "pwl_nr_multilinear_expressions",
        "pwl_nr_one_dim_expressions",
        "locatelli_domain_volume_polygon",
        "locatelli_domain_volume_polytope",
        "stair_locatelli_domain_volume_polygon",
        "stair_locatelli_domain_volume_polytope",
        "mpip_nr_instances",
        "mpip_ratio",
    }

    # Columns where lower is better (for comparisons)
    LOWER_IS_BETTER = {
        "solving_time",
        "nr_nodes",
        "mip_gap",
        "build_time",
        "root_solving_time",
        "final_nr_vars",
        "final_nr_constraints",
        "presolved_nr_vars",
        "presolved_nr_constraints",
        "locatelli_domain_volume_polygon",
        "locatelli_domain_volume_polytope",
        "stair_locatelli_domain_volume_polygon",
        "stair_locatelli_domain_volume_polytope",
    }

    def __init__(self, csv_path: str) -> None:
        """Initializes the evaluator with a CSV file path.

        Args:
            csv_path: Path to the CSV results file.
        """
        self.csv_path = csv_path
        self.data = StudyData()
        self._load_csv()

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

                instance = parsed_row.get("instance_name", "")
                config = parsed_row.get("config_name", "")
                self.data.data_matrix[(instance, config)] = parsed_row

        # Extract unique instances and configs
        self.data.instances = sorted(
            set(row.get("instance_name", "") for row in self.data.rows)
        )
        self.data.configs = sorted(
            set(row.get("config_name", "") for row in self.data.rows)
        )

        logger.info(
            "Loaded %d rows: %d instances × %d configs",
            len(self.data.rows),
            len(self.data.instances),
            len(self.data.configs),
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

    def get_column_values(
        self, column: str, config: str | None = None, instance: str | None = None
    ) -> list[float]:
        """Extracts numeric values for a column, optionally filtered.

        Args:
            column: Column name to extract.
            config: If provided, filter by this configuration.
            instance: If provided, filter by this instance.

        Returns:
            List of non-None numeric values.
        """
        values = []
        for row in self.data.rows:
            if config is not None and row.get("config_name") != config:
                continue
            if instance is not None and row.get("instance_name") != instance:
                continue

            val = row.get(column)
            if val is not None:
                values.append(val)

        return values

    def compute_column_stats(
        self, column: str, config: str | None = None
    ) -> ColumnStats:
        """Computes statistical summary for a column.

        Args:
            column: Column name to analyze.
            config: If provided, compute stats only for this configuration.

        Returns:
            ColumnStats object with computed statistics.
        """
        values = self.get_column_values(column, config=config)
        stats = ColumnStats(name=column, count=len(values))

        if not values:
            return stats

        arr = np.array(values)
        stats.mean = float(np.mean(arr))
        stats.std = float(np.std(arr))
        stats.min_val = float(np.min(arr))
        stats.max_val = float(np.max(arr))
        stats.median = float(np.median(arr))
        stats.q25 = float(np.percentile(arr, 25))
        stats.q75 = float(np.percentile(arr, 75))

        return stats

    def compute_config_comparison(  # pylint: disable=too-many-branches
        self,
        config_a: str,
        config_b: str,
        column: str,
        lower_is_better: bool | None = None,
    ) -> ConfigComparison:
        """Compares two configurations on a specific column.

        Args:
            config_a: Name of the first configuration.
            config_b: Name of the second configuration.
            column: Column to compare.
            lower_is_better: If True, lower values win. If None, inferred from column name.

        Returns:
            ConfigComparison with win/loss counts and ratio statistics.
        """
        if lower_is_better is None:
            lower_is_better = column in self.LOWER_IS_BETTER

        comparison = ConfigComparison(
            config_a=config_a, config_b=config_b, column=column
        )

        ratios = []

        for instance in self.data.instances:
            val_a = self.data.data_matrix.get((instance, config_a), {}).get(column)
            val_b = self.data.data_matrix.get((instance, config_b), {}).get(column)

            if val_a is None or val_b is None:
                continue

            # Compute ratio (avoiding division by zero)
            if val_b != 0:
                ratios.append(val_a / val_b)

            # Determine winner
            if val_a < val_b:
                if lower_is_better:
                    comparison.wins_a += 1
                else:
                    comparison.wins_b += 1
            elif val_a > val_b:
                if lower_is_better:
                    comparison.wins_b += 1
                else:
                    comparison.wins_a += 1
            else:
                comparison.ties += 1

        if ratios:
            comparison.mean_ratio = float(np.mean(ratios))
            # Geometric mean (only for positive ratios)
            positive_ratios = [r for r in ratios if r > 0]
            if positive_ratios:
                comparison.geometric_mean_ratio = float(
                    np.exp(np.mean(np.log(positive_ratios)))
                )

        return comparison

    def get_pivot_table(self, value_column: str) -> dict[str, dict[str, float | None]]:
        """Creates a pivot table with instances as rows and configs as columns.

        Args:
            value_column: Column to aggregate.
            aggfunc: Aggregation function ("mean", "sum", "min", "max").

        Returns:
            Nested dictionary: {instance: {config: value}}.
        """
        pivot: dict[str, dict[str, float | None]] = {}

        for instance in self.data.instances:
            pivot[instance] = {}
            for config in self.data.configs:
                value = self.data.data_matrix.get((instance, config), {}).get(
                    value_column
                )
                pivot[instance][config] = value

        return pivot

    def get_aggregated_stats_by_config(
        self, columns: list[str] | None = None
    ) -> dict[str, dict[str, ColumnStats]]:
        """Computes statistics for each column grouped by configuration.

        Args:
            columns: List of columns to analyze. If None, uses all numeric columns.

        Returns:
            Nested dictionary: {config: {column: ColumnStats}}.
        """
        if columns is None:
            columns = [h for h in self.data.headers if h in self.NUMERIC_COLUMNS]

        result: dict[str, dict[str, ColumnStats]] = {}

        for config in self.data.configs:
            result[config] = {}
            for column in columns:
                result[config][column] = self.compute_column_stats(
                    column, config=config
                )

        return result

    def compute_performance_profile_data(  # pylint: disable=too-many-locals, too-many-branches
        self, column: str, lower_is_better: bool | None = None
    ) -> dict[str, list[tuple[float, float]]]:
        """Computes data for performance profile plots.

        For each configuration, computes the cumulative distribution of
        performance ratios relative to the best configuration per instance.

        Args:
            column: Column to analyze (typically solving_time).
            lower_is_better: If True, lower values are better.

        Returns:
            Dictionary mapping config names to lists of (ratio, fraction) tuples.
        """
        if lower_is_better is None:
            lower_is_better = column in self.LOWER_IS_BETTER

        # Compute best value per instance
        best_per_instance: dict[str, float] = {}
        for instance in self.data.instances:
            values = []
            for config in self.data.configs:
                val = self.data.data_matrix.get((instance, config), {}).get(column)
                if val is not None:
                    values.append(val)

            if values:
                if lower_is_better:
                    best_per_instance[instance] = min(values)
                else:
                    best_per_instance[instance] = max(values)

        # Compute ratios for each config
        ratios_by_config: dict[str, list[float]] = {c: [] for c in self.data.configs}

        for instance in self.data.instances:
            if instance not in best_per_instance:
                continue

            best = best_per_instance[instance]
            if best == 0:
                continue

            for config in self.data.configs:
                val = self.data.data_matrix.get((instance, config), {}).get(column)
                if val is not None:
                    if lower_is_better:
                        ratio = val / best
                    else:
                        ratio = best / val if val != 0 else float("inf")
                    ratios_by_config[config].append(ratio)

        # Convert to cumulative distribution
        profile_data: dict[str, list[tuple[float, float]]] = {}

        for config, ratios in ratios_by_config.items():
            if not ratios:
                profile_data[config] = []
                continue

            sorted_ratios = sorted(ratios)
            n = len(sorted_ratios)
            points = [(sorted_ratios[i], (i + 1) / n) for i in range(n)]
            profile_data[config] = points

        return profile_data
