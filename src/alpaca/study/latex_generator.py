# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals
"""
@authors: kuen,
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np

from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.study.evaluator import StudyEvaluator, InstanceFilter


@dataclass
class TableFormatOptions:
    """Formatting options for table values.

    Attributes:
        precision: Decimal precision for values.
        percentage: Bool if use percentage format.
        bold: Optional string to specify which value to bold (e.g., "max" or "min").
    """

    precision: int = 2
    percentage: bool = False
    bold: str | None = None


@dataclass
class TableMetadata:
    """Metadata for a LaTeX table.

    Attributes:
        filename: Output filename.
        caption: Table caption.
        label: Table label for referencing.
    """

    filename: str
    caption: str = lsf.empty_string()
    label: str = lsf.empty_string()


@dataclass
class TableDefinition:
    """Definition of a table to generate.

    Attributes:
        column: Column name(s) to compare.
        metadata: Table metadata (filename, caption, label).
        config: Config to get model size from or to compare.
        filter_type: Instance filter to apply.
        format_options: Formatting options for values.
    """

    column: str | list[str]
    metadata: TableMetadata
    config: str = lsf.empty_string()
    filter_type: InstanceFilter = InstanceFilter.NONE
    format_options: TableFormatOptions = field(default_factory=TableFormatOptions)


@dataclass
class GeneratorConfig:
    """Configuration for the LaTeX table generator.

    Attributes:
        evaluator: StudyEvaluator instance with loaded data.
        output_dir: Directory to save generated LaTeX files.
        config_shortener: Optional function to shorten config names.
    """

    evaluator: StudyEvaluator
    output_dir: str
    config_shortener: Callable[[str], str] | None = None


class LatexTableGenerator:
    """Generates LaTeX tables from study evaluation results."""

    _LSF_METHODS: list[Callable[[], str]] = [
        lsf.stats_solving_time,
        lsf.stats_nr_nodes,
        lsf.stats_mip_gap,
        lsf.stats_solution_value,
        lsf.stats_final_nr_vars,
        lsf.stats_final_nr_constraints,
        lsf.stats_presolved_nr_vars,
        lsf.stats_presolved_nr_constraints,
        lsf.stats_presolved_nr_nonzeros,
        lsf.stats_root_solution_value,
        lsf.stats_root_solving_time,
        lsf.stats_build_time,
        lsf.stats_original_nr_variables,
        lsf.stats_original_nr_constraints,
        lsf.stats_original_nr_bilinear_expressions,
        lsf.stats_original_nr_bilinear_binary_expressions,
        lsf.stats_original_nr_mixed_binary_expressions,
        lsf.stats_original_nr_multilinear_expressions,
        lsf.stats_original_nr_one_dim_expressions,
        lsf.stats_pwl_nr_variables,
        lsf.stats_pwl_nr_constraints,
        lsf.stats_pwl_nr_bilinear_expressions,
        lsf.stats_pwl_nr_bilinear_binary_expressions,
        lsf.stats_pwl_nr_mixed_binary_expressions,
        lsf.stats_pwl_nr_multilinear_expressions,
        lsf.stats_pwl_nr_one_dim_expressions,
        lsf.stats_locatelli_domain_volume_polygon,
        lsf.stats_locatelli_domain_volume_polytope,
        lsf.stats_locatelli_nr_cuts,
        lsf.stats_mpip_nr_instances,
        lsf.stats_mpip_ratio,
        lsf.stats_instance_name,
        lsf.stats_config_name,
    ]

    COLUMN_TO_LSF_METHOD: dict[str, Callable[[], str]] = {
        method(): method for method in _LSF_METHODS
    }

    GEOMETRIC_MEAN_SHIFT = 10.0

    def __init__(self, config: GeneratorConfig) -> None:
        """Initialize the LaTeX table generator.

        Args:
            config: Generator configuration containing evaluator, output_dir,
                    and optional config_shortener.
        """
        self._config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config_shortnames = self._build_config_shortnames()

    @property
    def evaluator(self) -> StudyEvaluator:
        """Get the study evaluator."""
        return self._config.evaluator

    @property
    def config_shortener(self) -> Callable[[str], str]:
        """Get the config shortener function."""
        if self._config.config_shortener is not None:
            return self._config.config_shortener
        return self._default_config_shortener

    def _build_config_shortnames(self) -> dict[str, str]:
        """Build mapping of config names to short names."""
        shortnames = {}
        used_shortnames: set[str] = set()

        for config in self.evaluator.data.configs:
            short = self._get_unique_shortname(config, used_shortnames)
            shortnames[config] = short
            used_shortnames.add(short)

        return shortnames

    def _get_unique_shortname(self, config: str, used: set[str]) -> str:
        """Get a unique short name for a config."""
        short = self.config_shortener(config)
        original_short = short
        counter = 1
        while short in used:
            short = f"{original_short}{counter}"
            counter += 1
        return short

    @staticmethod
    def _default_config_shortener(config_name: str) -> str:
        """Default config name shortener."""
        if lsf.underscore() in config_name:
            parts = config_name.split(lsf.underscore())
            abbrev = lsf.empty_string().join(p[0].upper() for p in parts if p)
            if len(abbrev) >= 2:
                return abbrev

        capitals = [c for c in config_name if c.isupper()]
        if len(capitals) >= 2:
            return lsf.empty_string().join(capitals)

        return config_name[:4].upper()

    def _format_value(
        self,
        value: float | None,
        fmt: TableFormatOptions,
        relative_to: float | None = None,
        bold: bool = False,
    ) -> str:
        """Format a numeric value for LaTeX."""
        if value is None:
            return lsf.placeholder()

        if relative_to is not None:
            value = (
                0.0 if relative_to == 0 else 100 * (relative_to - value) / relative_to
            )
            return lsf.math_mode(
                f"{value:.{fmt.precision}f}{lsf.percentage_suffix()}", bold=bold
            )

        if value == self._config.evaluator.time_limit:
            return lsf.timeout_placeholder()

        if fmt.percentage:
            value *= 100
            return lsf.math_mode(
                f"{value:.{fmt.precision}f}{lsf.percentage_suffix()}", bold=bold
            )

        if fmt.precision == 0:
            return lsf.math_mode(str(int(round(value))), bold=bold)

        return lsf.math_mode(f"{value:.{fmt.precision}f}", bold=bold)

    def _get_column_header(self, column: str) -> str:
        """Get LaTeX formatted column header."""
        latex_format = lsf.stats_format_latex()

        if column in self.COLUMN_TO_LSF_METHOD:
            return self.COLUMN_TO_LSF_METHOD[column](latex_format)

        return lsf.escape_underscore(column)

    @staticmethod
    def _build_latex_table(
        headers: list[str],
        rows: list[list[str]],
        metadata: TableMetadata,
        alignment: str | None = None,
    ) -> str:
        """Build a complete LaTeX table."""
        if alignment is None:
            alignment = lsf.default_alignment(len(headers))

        lines = [lsf.begin_table(), lsf.centering()]

        if metadata.caption:
            lines.append(lsf.caption(metadata.caption))
        if metadata.label:
            lines.append(lsf.label(metadata.label))

        lines.extend(
            [
                lsf.begin_tabular(alignment),
                lsf.toprule(),
                lsf.column_separator().join(headers) + lsf.row_end(),
                lsf.midrule(),
            ]
        )

        for row in rows[:-3]:
            lines.append(lsf.column_separator().join(row) + lsf.row_end())

        lines.extend(
            [
                lsf.bottomrule(),
                lsf.column_separator().join(rows[-3]) + lsf.row_end(),
                lsf.column_separator().join(rows[-2]) + lsf.row_end(),
                lsf.column_separator().join(rows[-1]) + lsf.row_end(),
                lsf.end_tabular(),
                lsf.end_table(),
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _compute_mean(values: list[float]) -> float | None:
        """Compute arithmetic mean of values."""
        return float(np.mean(values))

    @staticmethod
    def _compute_median(values: list[float]) -> float | None:
        """Compute median of values."""
        return float(np.median(values))

    def _compute_shifted_geometric_mean(self, values: list[float]) -> float | None:
        """Compute shifted geometric mean of values."""
        shifted_values = [v + self.GEOMETRIC_MEAN_SHIFT for v in values]
        if any(v <= 0 for v in shifted_values):
            return None

        log_mean = np.mean(np.log(shifted_values))
        return float(np.exp(log_mean) - self.GEOMETRIC_MEAN_SHIFT)

    def _build_model_size_headers(self, columns: list[str]) -> list[str]:
        """Build headers for model size table."""
        headers = [lsf.stats_instance_name(lsf.stats_format_latex())]
        for col in columns:
            header = self.COLUMN_TO_LSF_METHOD[col](lsf.stats_format_latex())
            headers.append(lsf.escape_underscore(header))
        return headers

    def _build_model_size_rows(
        self, definition: TableDefinition
    ) -> tuple[list[list[str]], dict[str, list[float]]]:
        """Build rows and collect values for model size table."""
        instances = self.evaluator.get_filtered_instances(InstanceFilter.NONE)
        columns = definition.column
        fmt = definition.format_options

        rows = []
        all_values = {col: [] for col in columns}

        for instance in instances:
            row = [lsf.escape_underscore(instance)]
            for col in columns:
                value = self.evaluator.get_column_values_for_instance(col, instance)[
                    definition.config
                ]
                all_values[col].append(value)
                row.append(self._format_value(value, fmt))
            rows.append(row)

        return rows, all_values

    def generate_instance_model_size_table(self, definition: TableDefinition) -> str:
        """Generate table with instances as rows and model sizes as columns.

        Args:
            definition: Table definition with column as list of column names.

        Returns:
            LaTeX table code.
        """
        columns = definition.column
        fmt = definition.format_options

        headers = self._build_model_size_headers(columns)
        rows, all_values = self._build_model_size_rows(definition)

        mean_row = [lsf.mean_label()]
        sgm_row = [lsf.shifted_geometric_mean_label()]
        median_row = [lsf.median_label()]
        for col in columns:
            mean_value = self._compute_mean(all_values[col])
            sgm_value = self._compute_shifted_geometric_mean(all_values[col])
            median_value = self._compute_median(all_values[col])
            mean_row.append(self._format_value(mean_value, fmt))
            sgm_row.append(self._format_value(sgm_value, fmt))
            median_row.append(self._format_value(median_value, fmt))
        rows.append(mean_row)
        rows.append(sgm_row)
        rows.append(median_row)

        return self._build_latex_table(headers, rows, definition.metadata)

    def _build_pivot_headers(self) -> list[str]:
        """Build headers for pivot table."""
        headers = [lsf.stats_instance_name(lsf.stats_format_latex())]
        for config in self.evaluator.data.configs:
            headers.append(lsf.escape_underscore(self._config_shortnames[config]))
        return headers

    def _build_pivot_rows(
        self, definition: TableDefinition
    ) -> tuple[list[list[str]], dict[str, list[float]]]:
        """Build rows and collect values for pivot table."""
        instances = self.evaluator.get_filtered_instances(definition.filter_type)
        fmt = definition.format_options

        rows = []
        all_values = {cfg: [] for cfg in self.evaluator.data.configs}

        for instance in instances:
            row = [lsf.escape_underscore(instance)]
            values = self.evaluator.get_column_values_for_instance(
                definition.column, instance
            )
            relative_to = (
                None
                if definition.config == lsf.empty_string()
                else values.get(definition.config)
            )
            bold_value = (
                None
                if (
                    definition.format_options.bold is None
                    or any(
                        values.get(config) is None
                        for config in self.evaluator.data.configs
                    )
                )
                else (
                    max(values.get(config) for config in self.evaluator.data.configs)
                    if definition.format_options.bold == "max"
                    else min(
                        values.get(config) for config in self.evaluator.data.configs
                    )
                )
            )
            for config in self.evaluator.data.configs:
                value = values.get(config)
                bold = bold_value is not None and value == bold_value
                value = 0.0 if value is None else value
                row.append(self._format_value(value, fmt, relative_to, bold=bold))
            if any(v is None or "inf" in v for v in row):
                continue
            for config in self.evaluator.data.configs:
                value = values.get(config)
                value = 0.0 if value is None else value
                all_values[config].append(value)
            rows.append(row)

        return rows, all_values

    def generate_instance_pivot_table(self, definition: TableDefinition) -> str:
        """Generate table with instances as rows and configs as columns.

        Args:
            definition: Table definition with column as single column name.

        Returns:
            LaTeX table code.
        """
        fmt = definition.format_options

        headers = self._build_pivot_headers()
        rows, all_values = self._build_pivot_rows(definition)

        mean_row = [lsf.mean_label()]
        sgm_row = [lsf.shifted_geometric_mean_label()]
        median_row = [lsf.median_label()]
        relative_to_sgm = (
            self._compute_shifted_geometric_mean(all_values[definition.config])
            if definition.config != lsf.empty_string()
            else None
        )
        relative_to_mean = (
            self._compute_mean(all_values[definition.config])
            if definition.config != lsf.empty_string()
            else None
        )
        relative_to_median = (
            self._compute_median(all_values[definition.config])
            if definition.config != lsf.empty_string()
            else None
        )
        for config in self.evaluator.data.configs:
            values = all_values[config]
            sgm_value = self._compute_shifted_geometric_mean(values)
            mean_value = self._compute_mean(values)
            median_value = self._compute_median(values)
            mean_row.append(self._format_value(mean_value, fmt, relative_to_mean))
            sgm_row.append(self._format_value(sgm_value, fmt, relative_to_sgm))
            median_row.append(self._format_value(median_value, fmt, relative_to_median))
        rows.append(mean_row)
        rows.append(sgm_row)
        rows.append(median_row)

        return self._build_latex_table(headers, rows, definition.metadata)

    def save_table(self, latex_code: str, filename: str) -> Path:
        """Save LaTeX code to a file."""
        filepath = self._output_dir / filename
        with open(
            filepath, lsf.file_mode_write(), encoding=lsf.file_encoding_utf8()
        ) as file:
            file.write(latex_code)
        return filepath

    def generate_table_from_definition(self, definition: TableDefinition) -> str:
        """Generate a table from a TableDefinition."""
        if isinstance(definition.column, list):
            return self.generate_instance_model_size_table(definition)
        return self.generate_instance_pivot_table(definition)

    def generate_and_save_table(self, definition: TableDefinition) -> Path:
        """Generate and save a table from a definition."""
        latex_code = self.generate_table_from_definition(definition)
        return self.save_table(latex_code, definition.metadata.filename)

    def generate_tables(self, definitions: list[TableDefinition]) -> list[Path]:
        """Generate and save multiple tables from definitions."""
        return [self.generate_and_save_table(defn) for defn in definitions]

    def get_predefined_table_definitions(self) -> list[TableDefinition]:
        """Get the predefined table definitions."""
        return [
            TableDefinition(
                column=lsf.stats_solving_time(),
                filter_type=InstanceFilter.NONE,
                metadata=TableMetadata(
                    filename="table_instance_solution_time.tex",
                    caption="Solving time per instance and configuration",
                    label="tab:instance_solution_time",
                ),
                format_options=TableFormatOptions(precision=2, bold="min"),
            ),
            TableDefinition(
                column=lsf.stats_nr_nodes(),
                filter_type=InstanceFilter.BRANCH_AND_BOUND,
                metadata=TableMetadata(
                    filename="table_instance_nr_nodes.tex",
                    caption="Number of nodes per instance and configuration",
                    label="tab:instance_nr_nodes",
                ),
                format_options=TableFormatOptions(precision=0, bold="min"),
            ),
            TableDefinition(
                column=lsf.stats_root_solution_value(),
                filter_type=InstanceFilter.ALL_REACHED_ROOT,
                metadata=TableMetadata(
                    filename="table_instance_root_solution.tex",
                    caption="Root relaxation solution per instance and configuration",
                    label="tab:instance_root_solution",
                ),
                format_options=TableFormatOptions(precision=0, bold="max"),
            ),
            TableDefinition(
                column=lsf.stats_mip_gap(),
                filter_type=InstanceFilter.NONE,
                metadata=TableMetadata(
                    filename="table_instance_mip_gap.tex",
                    caption="MIP gap per instance and configuration",
                    label="tab:instance_mip_gap",
                ),
                format_options=TableFormatOptions(percentage=True, bold="min"),
            ),
            TableDefinition(
                config=self.evaluator.data.configs[0],
                column=[
                    lsf.stats_pwl_nr_variables(),
                    lsf.stats_pwl_nr_constraints(),
                    lsf.stats_pwl_nr_bilinear_expressions(),
                ],
                metadata=TableMetadata(
                    filename="table_instance_model_size.tex",
                    caption="Model size per instance",
                    label="tab:instance_model_size",
                ),
                format_options=TableFormatOptions(precision=0),
            ),
            TableDefinition(
                config="nonlinear_gurobi",
                filter_type=InstanceFilter.NON_EMPTY_BILINEAR_DOMAIN,
                column=lsf.stats_locatelli_domain_volume_polygon(),
                metadata=TableMetadata(
                    filename="table_instance_domain_volume_polygon.tex",
                    caption="Domain volume reduction relative to McCormick"
                    " over polygon per instance and configuration",
                    label="tab:instance_domain_volume_polygon",
                ),
                format_options=TableFormatOptions(precision=2, bold="min"),
            ),
            TableDefinition(
                config="nonlinear_gurobi",
                filter_type=InstanceFilter.NON_EMPTY_BILINEAR_DOMAIN,
                column=lsf.stats_locatelli_domain_volume_polytope(),
                metadata=TableMetadata(
                    filename="table_instance_domain_volume_polytope.tex",
                    caption="Domain volume reduction relative to McCormick"
                    " over polytope per instance and configuration",
                    label="tab:instance_domain_volume_polytope",
                ),
                format_options=TableFormatOptions(precision=2, bold="min"),
            ),
        ]

    def generate_all_predefined_tables(self) -> list[Path]:
        """Generate all predefined tables."""
        definitions = self.get_predefined_table_definitions()
        return self.generate_tables(definitions)

    def get_config_legend(self) -> str:
        """Get LaTeX code for config name legend."""
        lines = [lsf.begin_itemize()]
        for config, short in self._config_shortnames.items():
            escaped_config = lsf.escape_underscore(config)
            escaped_short = lsf.escape_underscore(short)
            lines.append(lsf.item(escaped_short, escaped_config))
        lines.append(lsf.end_itemize())
        return "\n".join(lines)

    def save_config_legend(self, filename: str = "config_legend.tex") -> Path:
        """Save config name legend to file."""
        legend = self.get_config_legend()
        return self.save_table(legend, filename)
