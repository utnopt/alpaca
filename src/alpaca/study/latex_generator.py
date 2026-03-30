# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
        config: Config to get model size from.
        instance_filter: Filter to apply to instances.
        format_options: Formatting options for values.
    """

    column: str | list[str]
    metadata: TableMetadata
    config: str = lsf.empty_string()
    instance_filter: InstanceFilter = InstanceFilter.NONE
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

    def __init__(self, config: GeneratorConfig) -> None:
        """Initialize the LaTeX table generator.

        Args:
            config: Generator configuration containing evaluator, output_dir,
                    and optional config_shortener.
        """
        self._config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config_shortnames = self.evaluator.build_config_shortnames()

    @property
    def evaluator(self) -> StudyEvaluator:
        """Get the study evaluator."""
        return self._config.evaluator

    def _format_value(
        self,
        value: float | None,
        fmt: TableFormatOptions,
        bold: bool = False,
    ) -> str:
        """Format a numeric value for LaTeX."""
        if value is None:
            return lsf.placeholder()

        if value == self._config.evaluator.time_limit:
            return lsf.timeout_placeholder()

        if fmt.percentage:
            return lsf.math_mode(
                f"{100 * value:.{fmt.precision}f}{lsf.percentage_suffix()}", bold=bold
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

    def _build_model_size_headers(self, columns: list[str]) -> list[str]:
        """Build headers for model size table."""
        headers = [lsf.stats_instance_name(lsf.stats_format_latex())]
        for col in columns:
            header = self.COLUMN_TO_LSF_METHOD[col](lsf.stats_format_latex())
            headers.append(lsf.escape_underscore(header))
        return headers

    def _build_model_size_rows(self, definition: TableDefinition) -> list[list[str]]:
        """Build rows and collect values for model size table."""
        instances = self.evaluator.get_filtered_instances(definition.instance_filter)
        columns = definition.column
        format_options = definition.format_options

        rows = []

        for instance in instances + [
            lsf.mean_label(),
            lsf.median_label(),
            lsf.shifted_geometric_mean_label(),
        ]:
            row = [lsf.escape_underscore(instance)]
            for col in columns:
                value = self.evaluator.get_column_values_for_instance(col, instance)[
                    definition.config
                ]
                value_str = self._format_value(value, format_options)
                row.append(value_str)
            rows.append(row)

        return rows

    def generate_instance_model_size_table(self, definition: TableDefinition) -> str:
        """Generate table with instances as rows and model sizes as columns.

        Args:
            definition: Table definition with column as list of column names.

        Returns:
            LaTeX table code.
        """
        headers = self._build_model_size_headers(definition.column)
        rows = self._build_model_size_rows(definition)
        return self._build_latex_table(headers, rows, definition.metadata)

    def _build_pivot_headers(self, header_configs: list[str]) -> list[str]:
        """Build headers for pivot table."""
        headers = [lsf.stats_instance_name(lsf.stats_format_latex())]
        for config in header_configs:
            headers.append(lsf.escape_underscore(self._config_shortnames[config]))
        return headers

    def _compute_bold_value(
        self, values: dict[str, float | None], bold_type: str | None
    ) -> float | None:
        """Compute the value that should be bolded based on bold_type."""
        if bold_type is None:
            return None

        config_values = list(values.values())

        if any(val is None for val in config_values):
            return None

        return max(config_values) if bold_type == "max" else min(config_values)

    def _build_pivot_row_for_instance(
        self, instance: str, definition: TableDefinition
    ) -> list[str]:
        """Build a single pivot table row for an instance."""
        row = [lsf.escape_underscore(instance)]
        values = self.evaluator.get_column_values_for_instance(
            definition.column, instance
        )
        bold_value = self._compute_bold_value(values, definition.format_options.bold)

        for config in self.evaluator.data.configs:
            if not config in values:
                continue
            value = values.get(config)
            bold = bold_value is not None and value == bold_value
            formatted_value = 0.0 if value is None else value
            value_str = self._format_value(
                formatted_value, definition.format_options, bold=bold
            )
            row.append(value_str)

        return row

    def _build_pivot_rows(self, definition: TableDefinition) -> list[list[str]]:
        """Build rows and collect values for pivot table."""
        instances = self.evaluator.get_filtered_instances(definition.instance_filter)
        rows = []

        for instance in instances + [
            lsf.mean_label(),
            lsf.median_label(),
            lsf.shifted_geometric_mean_label(),
        ]:
            row = self._build_pivot_row_for_instance(instance, definition)
            rows.append(row)

        return rows

    def generate_instance_pivot_table(self, definition: TableDefinition) -> str:
        """Generate table with instances as rows and configs as columns.

        Args:
            definition: Table definition with column as single column name.

        Returns:
            LaTeX table code.
        """
        rows = self._build_pivot_rows(definition)
        header_configs = list(
            self.evaluator.get_column_values_for_instance(
                definition.column, lsf.mean_label()
            ).keys()
        )
        headers = self._build_pivot_headers(header_configs)

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
                metadata=TableMetadata(
                    filename="table_instance_solution_time.tex",
                    caption="Solving time per instance and configuration.",
                    label="tab:instance_solution_time",
                ),
                format_options=TableFormatOptions(precision=2, bold="min"),
            ),
            TableDefinition(
                column=lsf.stats_nr_nodes(),
                metadata=TableMetadata(
                    filename="table_instance_nr_nodes.tex",
                    caption="Number of nodes per instance and configuration. "
                    "Filtered to instances solved not "
                    "in root node and solved to optimality.",
                    label="tab:instance_nr_nodes",
                ),
                format_options=TableFormatOptions(precision=0, bold="min"),
            ),
            TableDefinition(
                column=lsf.stats_root_solution_value(),
                metadata=TableMetadata(
                    filename="table_instance_root_solution.tex",
                    caption="Root relaxation solution per instance and configuration. "
                    "Filtered to instances that reached "
                    "the root node.",
                    label="tab:instance_root_solution",
                ),
                format_options=TableFormatOptions(precision=2, bold="max"),
            ),
            TableDefinition(
                column=lsf.stats_mip_gap(),
                metadata=TableMetadata(
                    filename="table_instance_mip_gap.tex",
                    caption="MIP gap per instance and configuration.",
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
                    caption="Model size per instance.",
                    label="tab:instance_model_size",
                ),
                format_options=TableFormatOptions(precision=0),
            ),
            TableDefinition(
                column=lsf.stats_volume_reduction_polygon(),
                metadata=TableMetadata(
                    filename="table_instance_domain_volume_polygon.tex",
                    caption="Domain volume reduction relative to McCormick"
                    " over polygon per instance and configuration. "
                    "Filtered to instances with non-empty bilinear domain.",
                    label="tab:instance_domain_volume_polygon",
                ),
                format_options=TableFormatOptions(
                    precision=2, bold="max", percentage=True
                ),
            ),
            TableDefinition(
                column=lsf.stats_volume_reduction_polytope(),
                metadata=TableMetadata(
                    filename="table_instance_domain_volume_polytope.tex",
                    caption="Domain volume reduction relative to McCormick"
                    " over polytope per instance and configuration. "
                    "Filtered to instances with non-empty bilinear domain.",
                    label="tab:instance_domain_volume_polytope",
                ),
                format_options=TableFormatOptions(
                    precision=2, bold="max", percentage=True
                ),
            ),
        ]

    def generate_all_predefined_tables(self) -> list[Path]:
        """Generate all predefined tables."""
        definitions = self.get_predefined_table_definitions()
        return self.generate_tables(definitions)
