# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals, too-many-lines
"""
@authors: Claude Opus,
"""
import os
from dataclasses import dataclass

from alpaca.study.evaluator import StudyEvaluator, InstanceFilter
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger
import alpaca.utils.inout as uio


@dataclass
class LaTeXConfig:
    """Configuration for LaTeX output generation.

    Attributes:
        tables_dir: Directory for LaTeX table files.
        plots_dir: Directory for TikZ plot files.
        float_precision: Number of decimal places for floats.
        use_siunitx: Whether to use siunitx package for numbers.
    """

    tables_dir: str
    plots_dir: str
    float_precision: int = 3
    use_siunitx: bool = True


class LaTeXGenerator:
    """Generates LaTeX tables and TikZ plots from study results."""

    COLORS = [
        "MidnightNavy",
        "BrickRose",
        "OceanTeal",
        "AmberGold",
    ]

    # Define which filter to use for each metric category
    PERFORMANCE_FILTER = InstanceFilter.ALL_TERMINATED_CONSISTENT
    SOLUTION_FILTER = InstanceFilter.ALL_TERMINATED  # Allow different solutions
    MODEL_FILTER = InstanceFilter.NONE  # Model properties, no filter needed
    SURVIVAL_FILTER = InstanceFilter.NONE  # Survival plot shows all instances

    def __init__(self, evaluator: StudyEvaluator, config: LaTeXConfig) -> None:
        """Initializes the generator with an evaluator and configuration.

        Args:
            evaluator: StudyEvaluator instance with loaded data.
            config: LaTeXConfig with output settings.
        """
        self.evaluator = evaluator
        self.config = config

        os.makedirs(config.tables_dir, exist_ok=True)
        os.makedirs(config.plots_dir, exist_ok=True)

        # Log filter statistics
        stats = self.evaluator.get_filter_statistics()
        logger.info("Instance filter statistics:")
        logger.info("  Total complete: %d", stats["instance_counts"]["total_complete"])
        logger.info("  All terminated: %d", stats["instance_counts"]["all_terminated"])
        logger.info(
            "  All terminated + consistent: %d",
            stats["instance_counts"]["all_terminated_consistent"],
        )

    def _format_float(self, value: float | None) -> str:
        """Formats a float value for LaTeX output.

        Args:
            value: Float value to format, or None.

        Returns:
            Formatted string, or "--" if None.
        """
        if value is None:
            return "--"

        formatted = f"{value:.{self.config.float_precision}f}"

        if self.config.use_siunitx:
            return f"\\num{{{formatted}}}"
        return formatted

    def _format_percent(self, value: float | None) -> str:
        """Formats a percentage value for LaTeX output.

        Args:
            value: Percentage value to format (0-100), or None.

        Returns:
            Formatted string with percent sign, or "--" if None.
        """
        if value is None:
            return "--"

        formatted = f"{value:.{self.config.float_precision}f}"

        if self.config.use_siunitx:
            return f"\\SI{{{formatted}}}{{\\percent}}"
        return f"{formatted}\\%"

    def _format_int(self, value: int | float | None) -> str:
        """Formats an integer value for LaTeX output.

        Args:
            value: Integer value to format, or None.

        Returns:
            Formatted string, or "--" if None.
        """
        if value is None:
            return "--"

        int_val = int(value)
        if self.config.use_siunitx:
            return f"\\num{{{int_val}}}"
        return str(int_val)

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escapes special LaTeX characters in text.

        Args:
            text: Text to escape.

        Returns:
            Escaped text safe for LaTeX.
        """
        replacements = {
            "%": "\\%",
            "_": "\\_",
            "&": "\\&",
            "#": "\\#",
            "{": "\\{",
            "}": "\\}",
            "$": "\\$",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

    @staticmethod
    def _escape_latex_comment(text: str) -> str:
        """Escapes text for use in LaTeX comments.

        In comments, % starts a new comment, so we need to escape it.

        Args:
            text: Text to escape for comment.

        Returns:
            Escaped text safe for LaTeX comments.
        """
        # In comments, we only need to handle % specially
        # but since comments are already prefixed with %,
        # we just replace % in the content
        return text.replace("%", "pct")

    @staticmethod
    def _get_latex_column_name(column: str) -> str:
        """Gets the LaTeX display name for a column using LSF.

        Args:
            column: Internal column name.

        Returns:
            LaTeX-formatted display name.
        """
        latex_format = lsf.stats_format_latex()
        column_map = {
            lsf.stats_solving_time(): lsf.stats_solving_time(latex_format),
            lsf.stats_nr_nodes(): lsf.stats_nr_nodes(latex_format),
            lsf.stats_mip_gap(): lsf.stats_mip_gap(latex_format),
            lsf.stats_solution_value(): lsf.stats_solution_value(latex_format),
            lsf.stats_root_solution_value(): lsf.stats_root_solution_value(
                latex_format
            ),
            lsf.stats_final_nr_vars(): lsf.stats_final_nr_vars(latex_format),
            lsf.stats_final_nr_constraints(): lsf.stats_final_nr_constraints(
                latex_format
            ),
            lsf.stats_presolved_nr_vars(): lsf.stats_presolved_nr_vars(latex_format),
            lsf.stats_presolved_nr_constraints(): lsf.stats_presolved_nr_constraints(
                latex_format
            ),
            lsf.stats_locatelli_domain_volume_polygon(): (
                lsf.stats_locatelli_domain_volume_polygon(latex_format)
            ),
            lsf.stats_locatelli_domain_volume_polytope(): (
                lsf.stats_locatelli_domain_volume_polytope(latex_format)
            ),
            lsf.stats_locatelli_nr_cuts(): lsf.stats_locatelli_nr_cuts(latex_format),
            lsf.stats_pwl_nr_bilinear_expressions(): (
                lsf.stats_pwl_nr_bilinear_expressions(latex_format)
            ),
            lsf.stats_pwl_nr_multilinear_expressions(): (
                lsf.stats_pwl_nr_multilinear_expressions(latex_format)
            ),
            lsf.stats_pwl_nr_one_dim_expressions(): (
                lsf.stats_pwl_nr_one_dim_expressions(latex_format)
            ),
            lsf.stats_pwl_nr_bilinear_binary_expressions(): (
                lsf.stats_pwl_nr_bilinear_binary_expressions(latex_format)
            ),
            lsf.stats_pwl_nr_mixed_binary_expressions(): (
                lsf.stats_pwl_nr_mixed_binary_expressions(latex_format)
            ),
            lsf.stats_instance_name(): lsf.stats_instance_name(latex_format),
            lsf.stats_config_name(): lsf.stats_config_name(latex_format),
        }
        return column_map.get(column, column)

    def _get_color(self, index: int) -> str:
        """Gets color for a given index, cycling through available colors.

        Args:
            index: Color index.

        Returns:
            Color name string.
        """
        return self.COLORS[index % len(self.COLORS)]

    def _get_filter_note(self, filter_type: InstanceFilter) -> str:
        """Gets a note about the filter applied.

        Args:
            filter_type: The filter type used.

        Returns:
            LaTeX-formatted note about the filter.
        """
        instances = self.evaluator.get_filtered_instances(filter_type)
        total = len(self.evaluator.data.instances)

        if filter_type == InstanceFilter.NONE:
            return f"All {len(instances)} instances included."
        if filter_type == InstanceFilter.ALL_TERMINATED:
            return (
                f"{len(instances)}/{total} instances "
                "(only instances where all configs terminated)."
            )
        if filter_type == InstanceFilter.ALL_TERMINATED_CONSISTENT:
            return (
                f"{len(instances)}/{total} instances "
                "(only instances where all configs terminated with consistent solutions)."
            )
        return ""

    def _compute_relative_domain_volumes(
        self,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> dict[str, dict[str, float | None]]:
        """Computes relative domain volumes as percentage of maximum.

        For each domain volume column, finds the config with the highest mean
        and expresses all other values as percentage of that maximum.

        Args:
            filter_type: Instance filter to apply.

        Returns:
            Nested dict: {config: {column: relative_percentage}}.
        """
        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]

        # Get absolute values first
        absolute_data = self.evaluator.get_config_aggregated_data(
            columns, filter_type=filter_type
        )

        # Compute relative values for each column
        result = {config: {} for config in self.evaluator.data.configs}

        for col in columns:
            # Find maximum value for this column
            max_value = None
            for config in self.evaluator.data.configs:
                val = absolute_data[config][col]
                if val is not None:
                    if max_value is None or val > max_value:
                        max_value = val

            # Compute relative percentages
            for config in self.evaluator.data.configs:
                val = absolute_data[config][col]
                if val is not None and max_value is not None and max_value > 0:
                    result[config][col] = (max_value - val) / max_value * 100.0
                else:
                    result[config][col] = None

        return result

    def _compute_relative_domain_volumes_per_instance(
        self,
        filter_type: InstanceFilter = InstanceFilter.NONE,
    ) -> dict[str, dict[str, dict[str, float | None]]]:
        """Computes relative domain volumes per instance as percentage of maximum.

        For each instance and domain volume column, finds the config with the
        highest value and expresses all other values as percentage of that maximum.

        Args:
            filter_type: Instance filter to apply.

        Returns:
            Nested dict: {instance: {config: {column: relative_percentage}}}.
        """
        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]

        pivot_data = self.evaluator.get_pivot_data(columns, filter_type=filter_type)
        instances = self.evaluator.get_filtered_instances(filter_type)
        configs = self.evaluator.data.configs

        result = {}

        for instance in instances:
            result[instance] = {config: {} for config in configs}

            for col in columns:
                # Find maximum value for this instance and column
                max_value = None
                for config in configs:
                    val = pivot_data[instance][config][col]
                    if val is not None:
                        if max_value is None or val > max_value:
                            max_value = val

                # Compute relative percentages
                for config in configs:
                    val = pivot_data[instance][config][col]
                    if val is not None and max_value is not None and max_value > 0:
                        result[instance][config][col] = (max_value - val) / max_value * 100.0
                    else:
                        result[instance][config][col] = None

        return result

    def generate_config_performance_table(
        self, filename: str = "config_performance.tex"
    ) -> str:
        """Generates table: Configs x (SGM solving_time, Mean mip_gap, Mean nr_nodes).

        Filter: ALL_TERMINATED_CONSISTENT - only instances where all configs
        terminated and found consistent solutions.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.PERFORMANCE_FILTER

        columns = [
            lsf.stats_solving_time(),
            lsf.stats_mip_gap(),
            lsf.stats_nr_nodes(),
        ]
        data = self.evaluator.get_config_aggregated_data(
            columns,
            use_shifted_geom_mean=[lsf.stats_solving_time()],
            filter_type=filter_type,
        )

        col_spec = "l" + "r" * len(columns)
        header_parts = [self._get_latex_column_name(lsf.stats_config_name())] + [
            f"SGM {self._get_latex_column_name(columns[0])}"
        ]
        for col in columns[1:]:
            header_parts.append(f"Mean {self._get_latex_column_name(col)}")
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for config in self.evaluator.data.configs:
            row_parts = [self._escape_latex(config)]
            for col in columns:
                row_parts.append(self._format_float(data[config][col]))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        filter_note = self._get_filter_note(filter_type)

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=f"Configuration performance metrics. {filter_note}",
            label="tab:config_performance",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated config performance table: %s", output_path)
        return output_path

    def generate_config_performance_barplot(
        self, filename: str = "config_performance_barplot.tex"
    ) -> str:
        """Generates grouped bar plot for config performance metrics.

        Filter: ALL_TERMINATED_CONSISTENT

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.PERFORMANCE_FILTER

        columns = [
            lsf.stats_solving_time(),
            lsf.stats_mip_gap(),
            lsf.stats_nr_nodes(),
        ]
        data = self.evaluator.get_config_aggregated_data(
            columns,
            use_shifted_geom_mean=[lsf.stats_solving_time()],
            filter_type=filter_type,
        )

        configs = self.evaluator.data.configs
        symbolic_coords = ",".join(self._escape_latex(c) for c in configs)

        filter_note = self._escape_latex_comment(self._get_filter_note(filter_type))

        lines = [
            f"% Filter: {filter_note}",
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    ybar,",
            "    bar width=0.2cm,",
            "    width=0.9\\textwidth,",
            "    height=8cm,",
            "    ylabel={Value},",
            f"    symbolic x coords={{{symbolic_coords}}},",
            "    xtick=data,",
            "    x tick label style={rotate=45, anchor=east},",
            "    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=-1},",
            "    ymin=0,",
            "    enlarge x limits=0.15,",
            "    nodes near coords,",
            "    nodes near coords style={font=\\tiny, rotate=90, anchor=west},",
            "]",
        ]

        for i, col in enumerate(columns):
            color = self._get_color(i)
            col_label = self._get_latex_column_name(col)
            if col == lsf.stats_solving_time():
                col_label = f"SGM {col_label}"
            else:
                col_label = f"Mean {col_label}"

            lines.append(f"\\addplot[fill={color}] coordinates {{")
            for config in configs:
                val = data[config][col]
                if val is not None:
                    lines.append(f"    ({self._escape_latex(config)}, {val:.4f})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{self._escape_latex(col_label)}}}")

        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")

        content = "\n".join(lines)

        output_path = os.path.join(self.config.plots_dir, filename)
        uio.write_file(output_path, content)
        logger.info("Generated config performance bar plot: %s", output_path)
        return output_path

    def generate_config_solution_table(
        self, filename: str = "config_solution.tex"
    ) -> str:
        """Generates table: Configs x (Mean solution_value, Mean root_solution_value).

        Filter: ALL_TERMINATED - only instances where all configs terminated,
        but allows different solution values.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.SOLUTION_FILTER

        columns = [lsf.stats_solution_value(), lsf.stats_root_solution_value()]
        data = self.evaluator.get_config_aggregated_data(
            columns, filter_type=filter_type
        )

        col_spec = "l" + "r" * len(columns)
        header_parts = [self._get_latex_column_name(lsf.stats_config_name())]
        for col in columns:
            header_parts.append(f"Mean {self._get_latex_column_name(col)}")
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for config in self.evaluator.data.configs:
            row_parts = [self._escape_latex(config)]
            for col in columns:
                row_parts.append(self._format_float(data[config][col]))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        filter_note = self._get_filter_note(filter_type)

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=f"Configuration solution values. {filter_note}",
            label="tab:config_solution",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated config solution table: %s", output_path)
        return output_path

    def generate_config_solution_barplot(
        self, filename: str = "config_solution_barplot.tex"
    ) -> str:
        """Generates grouped bar plot for config solution values.

        Filter: ALL_TERMINATED

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.SOLUTION_FILTER

        columns = [lsf.stats_solution_value(), lsf.stats_root_solution_value()]
        data = self.evaluator.get_config_aggregated_data(
            columns, filter_type=filter_type
        )

        configs = self.evaluator.data.configs
        symbolic_coords = ",".join(self._escape_latex(c) for c in configs)

        filter_note = self._escape_latex_comment(self._get_filter_note(filter_type))

        lines = [
            f"% Filter: {filter_note}",
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    ybar,",
            "    bar width=0.3cm,",
            "    width=0.9\\textwidth,",
            "    height=8cm,",
            "    ylabel={Value},",
            f"    symbolic x coords={{{symbolic_coords}}},",
            "    xtick=data,",
            "    x tick label style={rotate=45, anchor=east},",
            "    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=-1},",
            "    enlarge x limits=0.15,",
            "]",
        ]

        for i, col in enumerate(columns):
            color = self._get_color(i)
            col_label = f"Mean {self._get_latex_column_name(col)}"

            lines.append(f"\\addplot[fill={color}] coordinates {{")
            for config in configs:
                val = data[config][col]
                if val is not None:
                    lines.append(f"    ({self._escape_latex(config)}, {val:.4f})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{self._escape_latex(col_label)}}}")

        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")

        content = "\n".join(lines)

        output_path = os.path.join(self.config.plots_dir, filename)
        uio.write_file(output_path, content)
        logger.info("Generated config solution bar plot: %s", output_path)
        return output_path

    def generate_config_domain_volume_table(
        self, filename: str = "config_domain_volume.tex"
    ) -> str:
        """Generates table: Configs x (Relative domain volumes in percent of max).

        Filter: NONE - domain volumes are model properties.
        Values are shown as percentage of the maximum value per column.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.MODEL_FILTER

        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]
        data = self._compute_relative_domain_volumes(filter_type=filter_type)

        col_spec = "l" + "r" * len(columns)
        header_parts = [self._get_latex_column_name(lsf.stats_config_name())]
        for col in columns:
            header_parts.append(f"Rel. {self._get_latex_column_name(col)}")
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for config in self.evaluator.data.configs:
            row_parts = [self._escape_latex(config)]
            for col in columns:
                row_parts.append(self._format_percent(data[config][col]))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=(
                "Configuration domain volume metrics "
                "(relative to maximum, 100\\% = largest volume)."
            ),
            label="tab:config_domain_volume",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated config domain volume table: %s", output_path)
        return output_path

    def generate_config_domain_volume_barplot(
        self, filename: str = "config_domain_volume_barplot.tex"
    ) -> str:
        """Generates grouped bar plot for config domain volumes (relative).

        Filter: NONE - domain volumes are model properties.
        Values are shown as percentage of the maximum value per column.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.MODEL_FILTER

        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]
        data = self._compute_relative_domain_volumes(filter_type=filter_type)

        configs = self.evaluator.data.configs
        symbolic_coords = ",".join(self._escape_latex(c) for c in configs)

        lines = [
            "% Domain volumes shown as percentage of maximum (100 = largest)",
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    ybar,",
            "    bar width=0.3cm,",
            "    width=0.9\\textwidth,",
            "    height=8cm,",
            "    ylabel={Relative Volume (\\%)},",
            f"    symbolic x coords={{{symbolic_coords}}},",
            "    xtick=data,",
            "    x tick label style={rotate=45, anchor=east},",
            "    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=-1},",
            "    ymin=0,",
            "    ymax=105,",
            "    enlarge x limits=0.15,",
            "]",
        ]
        for i, col in enumerate(columns):
            color = self._get_color(i)
            col_label = f"Rel. {self._get_latex_column_name(col)}"
            lines.append(f"\\addplot[fill={color}] coordinates {{")
            for config in configs:
                val = data[config][col]
                if val is not None:
                    lines.append(f"    ({self._escape_latex(config)}, {val:.2f})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{self._escape_latex(col_label)}}}")
        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")
        content = "\n".join(lines)
        output_path = os.path.join(self.config.plots_dir, filename)
        uio.write_file(output_path, content)
        logger.info("Generated config domain volume bar plot: %s", output_path)
        return output_path

    def generate_instance_performance_table(
        self, filename: str = "instance_performance.tex"
    ) -> str:
        """Generates table: Instances x (solving_time, mip_gap, nr_nodes) per config.

        Filter: ALL_TERMINATED_CONSISTENT

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.PERFORMANCE_FILTER

        columns = [
            lsf.stats_solving_time(),
            lsf.stats_mip_gap(),
            lsf.stats_nr_nodes(),
        ]
        pivot_data = self.evaluator.get_pivot_data(columns, filter_type=filter_type)
        instances = self.evaluator.get_filtered_instances(filter_type)
        configs = self.evaluator.data.configs

        num_cols = len(columns) * len(configs)
        col_spec = "l" + "r" * num_cols
        header_parts = [self._get_latex_column_name(lsf.stats_instance_name())]
        for config in configs:
            for col in columns:
                header_parts.append(
                    f"{self._escape_latex(config)}: {self._get_latex_column_name(col)}"
                )
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for instance in instances:
            row_parts = [self._escape_latex(instance)]
            for config in configs:
                for col in columns:
                    val = pivot_data[instance][config][col]
                    row_parts.append(self._format_float(val))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        filter_note = self._get_filter_note(filter_type)

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=f"Instance performance metrics. {filter_note}",
            label="tab:instance_performance",
        )
        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated instance performance table: %s", output_path)
        return output_path

    def generate_instance_performance_lineplot(
        self,
        column: str,
        filename: str,
        filter_type: InstanceFilter | None = None,
    ) -> str:
        """Generates line plot for instance performance metric.

        Args:
            column: Column to plot.
            filename: Output filename.
            filter_type: Filter to apply. Defaults to PERFORMANCE_FILTER for
                performance metrics, MODEL_FILTER for model properties.

        Returns:
            Path to the generated file.
        """
        # Determine appropriate filter based on column type
        if filter_type is None:
            performance_columns = {
                lsf.stats_solving_time(),
                lsf.stats_mip_gap(),
                lsf.stats_nr_nodes(),
            }
            solution_columns = {
                lsf.stats_solution_value(),
                lsf.stats_root_solution_value(),
            }

            if column in performance_columns:
                filter_type = self.PERFORMANCE_FILTER
            elif column in solution_columns:
                filter_type = self.SOLUTION_FILTER
            else:
                filter_type = self.MODEL_FILTER

        pivot_data = self.evaluator.get_pivot_data([column], filter_type=filter_type)
        instances = self.evaluator.get_filtered_instances(filter_type)
        configs = self.evaluator.data.configs

        filter_note = self._escape_latex_comment(self._get_filter_note(filter_type))

        lines = [
            f"% Filter: {filter_note}",
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    width=\\textwidth,",
            "    height=10cm,",
            f"    ylabel={{{self._get_latex_column_name(column)}}},",
            "    xlabel={Instance},",
            "    xtick=data,",
            "    xticklabels={},",
            "    legend style={at={(1.02,1)}, anchor=north west},",
            "    grid=major,",
            "    cycle list name=color list,",
            "]",
        ]
        for i, config in enumerate(configs):
            color = self._get_color(i)
            lines.append(
                f"\\addplot[{color}, thick, mark=*, mark size=1pt] coordinates {{"
            )
            for j, instance in enumerate(instances):
                val = pivot_data[instance][config][column]
                if val is not None:
                    lines.append(f"    ({j}, {val})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{self._escape_latex(config)}}}")
        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")
        content = "\n".join(lines)
        output_path = os.path.join(self.config.plots_dir, filename)
        uio.write_file(output_path, content)
        logger.info("Generated instance performance line plot: %s", output_path)
        return output_path

    def generate_instance_domain_volume_lineplot(
        self,
        column: str,
        filename: str,
    ) -> str:
        """Generates line plot for instance domain volumes (relative).

        Args:
            column: Domain volume column to plot.
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.MODEL_FILTER

        relative_data = self._compute_relative_domain_volumes_per_instance(
            filter_type=filter_type
        )
        instances = self.evaluator.get_filtered_instances(filter_type)
        configs = self.evaluator.data.configs

        lines = [
            "% Domain volumes shown as percentage of maximum per instance (100 = largest)",
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    width=\\textwidth,",
            "    height=10cm,",
            f"    ylabel={{Rel. {self._get_latex_column_name(column)} (\\%)}},",
            "    xlabel={Instance},",
            "    xtick=data,",
            "    xticklabels={},",
            "    legend style={at={(1.02,1)}, anchor=north west},",
            "    grid=major,",
            "    ymin=0,",
            "    ymax=105,",
            "    cycle list name=color list,",
            "]",
        ]
        for i, config in enumerate(configs):
            color = self._get_color(i)
            lines.append(
                f"\\addplot[{color}, thick, mark=*, mark size=1pt] coordinates {{"
            )
            for j, instance in enumerate(instances):
                val = relative_data[instance][config][column]
                if val is not None:
                    lines.append(f"    ({j}, {val:.2f})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{self._escape_latex(config)}}}")
        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")
        content = "\n".join(lines)
        output_path = os.path.join(self.config.plots_dir, filename)
        uio.write_file(output_path, content)
        logger.info("Generated instance domain volume line plot: %s", output_path)
        return output_path

    def generate_instance_solution_table(
        self, filename: str = "instance_solution.tex"
    ) -> str:
        """Generates table: Instances x (solution_value, root_solution_value) per config.

        Filter: ALL_TERMINATED

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.SOLUTION_FILTER

        columns = [lsf.stats_solution_value(), lsf.stats_root_solution_value()]
        pivot_data = self.evaluator.get_pivot_data(columns, filter_type=filter_type)
        instances = self.evaluator.get_filtered_instances(filter_type)
        configs = self.evaluator.data.configs

        num_cols = len(columns) * len(configs)
        col_spec = "l" + "r" * num_cols
        header_parts = [self._get_latex_column_name(lsf.stats_instance_name())]
        for config in configs:
            for col in columns:
                header_parts.append(
                    f"{self._escape_latex(config)}: {self._get_latex_column_name(col)}"
                )
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for instance in instances:
            row_parts = [self._escape_latex(instance)]
            for config in configs:
                for col in columns:
                    val = pivot_data[instance][config][col]
                    row_parts.append(self._format_float(val))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        filter_note = self._get_filter_note(filter_type)

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=f"Instance solution values. {filter_note}",
            label="tab:instance_solution",
        )
        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated instance solution table: %s", output_path)
        return output_path

    def generate_instance_domain_volume_table(
        self, filename: str = "instance_domain_volume.tex"
    ) -> str:
        """Generates table: Instances x (relative domain volumes) per config.

        Filter: NONE - domain volumes are model properties.
        Values are shown as percentage of the maximum value per instance.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.MODEL_FILTER

        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]
        relative_data = self._compute_relative_domain_volumes_per_instance(
            filter_type=filter_type
        )
        instances = self.evaluator.get_filtered_instances(filter_type)

        configs = self.evaluator.data.configs
        num_cols = len(columns) * len(configs)
        col_spec = "l" + "r" * num_cols

        header_parts = [self._get_latex_column_name(lsf.stats_instance_name())]
        for config in configs:
            for col in columns:
                header_parts.append(
                    f"{self._escape_latex(config)}: Rel. {self._get_latex_column_name(col)}"
                )
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for instance in instances:
            row_parts = [self._escape_latex(instance)]
            for config in configs:
                for col in columns:
                    val = relative_data[instance][config][col]
                    row_parts.append(self._format_percent(val))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=(
                "Instance domain volume metrics "
                "(relative to maximum per instance, 100\\% = largest volume)."
            ),
            label="tab:instance_domain_volume",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated instance domain volume table: %s", output_path)
        return output_path

    def generate_config_model_size_table(
        self, filename: str = "config_model_size.tex"
    ) -> str:
        """Generates table: Configs x (Mean model size metrics).

        Filter: NONE - model size is a model property.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.MODEL_FILTER

        columns = [
            lsf.stats_final_nr_vars(),
            lsf.stats_final_nr_constraints(),
            lsf.stats_presolved_nr_vars(),
            lsf.stats_presolved_nr_constraints(),
        ]
        data = self.evaluator.get_config_aggregated_data(
            columns, filter_type=filter_type
        )

        col_spec = "l" + "r" * len(columns)
        header_parts = [self._get_latex_column_name(lsf.stats_config_name())]
        for col in columns:
            header_parts.append(f"Mean {self._get_latex_column_name(col)}")
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for config in self.evaluator.data.configs:
            row_parts = [self._escape_latex(config)]
            for col in columns:
                row_parts.append(self._format_float(data[config][col]))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption="Configuration model size metrics",
            label="tab:config_model_size",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated config model size table: %s", output_path)
        return output_path

    def generate_instance_expressions_table(
        self, filename: str = "instance_expressions.tex"
    ) -> str:
        """Generates table: Instances x (PWL expression counts).

        Filter: NONE - expression counts are model properties.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.MODEL_FILTER

        columns = [
            lsf.stats_pwl_nr_bilinear_expressions(),
            lsf.stats_pwl_nr_multilinear_expressions(),
            lsf.stats_pwl_nr_one_dim_expressions(),
            lsf.stats_pwl_nr_bilinear_binary_expressions(),
            lsf.stats_pwl_nr_mixed_binary_expressions(),
        ]
        data = self.evaluator.get_instance_data(columns, filter_type=filter_type)
        instances = self.evaluator.get_filtered_instances(filter_type)

        col_spec = "l" + "r" * len(columns)
        header_parts = [self._get_latex_column_name(lsf.stats_instance_name())]
        for col in columns:
            header_parts.append(self._get_latex_column_name(col))
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for instance in instances:
            row_parts = [self._escape_latex(instance)]
            for col in columns:
                row_parts.append(self._format_int(data[instance][col]))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption="Instance expression counts",
            label="tab:instance_expressions",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated instance expressions table: %s", output_path)
        return output_path

    def generate_instances_solved_over_time_plot(
        self, filename: str = "instances_solved_over_time.tex"
    ) -> str:
        """Generates survival curve: time vs. number of instances solved.

        Filter: NONE - shows all instances to display survival behavior.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.SURVIVAL_FILTER

        data = self.evaluator.get_instances_solved_over_time(filter_type=filter_type)
        total_instances = len(self.evaluator.get_filtered_instances(filter_type))

        lines = [
            "% Filter: NONE - showing all instances for survival analysis",
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    width=0.9\\textwidth,",
            "    height=8cm,",
            f"    xlabel={{{self._get_latex_column_name(lsf.stats_solving_time())}}},",
            "    ylabel={Number of instances solved},",
            f"    ymax={total_instances + 1},",
            "    ymin=0,",
            "    xmin=0,",
            "    legend style={at={(0.02,0.98)}, anchor=north west},",
            "    grid=both,",
            "    minor grid style={gray!25},",
            "    major grid style={gray!50},",
            "]",
        ]

        for i, (config, points) in enumerate(data.items()):
            color = self._get_color(i)
            config_escaped = self._escape_latex(config)

            if not points:
                continue

            lines.append(
                f"\\addplot[{color}, thick, mark=none, const plot] coordinates {{"
            )
            lines.append("    (0, 0)")
            for time, count in points:
                lines.append(f"    ({time:.4f}, {count})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{config_escaped}}}")

        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")

        content = "\n".join(lines)
        output_path = os.path.join(self.config.plots_dir, filename)
        uio.write_file(output_path, content)
        logger.info("Generated instances solved over time plot: %s", output_path)
        return output_path

    def generate_config_performance_boxplot(
        self, filename: str = "config_performance_boxplot.tex"
    ) -> str:
        """Generates boxplot for config performance metrics.

        Filter: ALL_TERMINATED_CONSISTENT

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        filter_type = self.PERFORMANCE_FILTER

        columns = [
            lsf.stats_solving_time(),
            lsf.stats_mip_gap(),
            lsf.stats_nr_nodes(),
        ]
        configs = self.evaluator.data.configs

        if not columns:
            content = "% No columns specified for boxplot"
            output_path = os.path.join(self.config.plots_dir, filename)
            uio.write_file(output_path, content)
            return output_path

        filter_note = self._escape_latex_comment(self._get_filter_note(filter_type))

        lines = [
            f"% Filter: {filter_note}",
            "\\begin{tikzpicture}",
        ]

        for col_idx, column in enumerate(columns):
            x_offset = col_idx * 6
            col_label = self._get_latex_column_name(column)

            lines.append("\\begin{axis}[")
            lines.append(f"    at={{({x_offset}cm, 0)}},")
            lines.append("    boxplot/draw direction=y,")
            lines.append("    width=5cm,")
            lines.append("    height=8cm,")
            lines.append(f"    title={{{self._escape_latex(col_label)}}},")
            lines.append(
                f"    xtick={{{','.join(str(i + 1) for i in range(len(configs)))}}},"
            )
            lines.append(
                f"    xticklabels={{{','.join(self._escape_latex(c) for c in configs)}}},"
            )
            lines.append(
                "    x tick label style={rotate=45, anchor=east, font=\\small},"
            )
            lines.append("    ylabel={Value},")
            lines.append("]")

            for i, config in enumerate(configs):
                color = self._get_color(i)
                stats = self.evaluator.compute_statistics_for_config(
                    column, config, filter_type=filter_type
                )

                if stats["median"] is None:
                    continue

                lines.append("\\addplot+[")
                lines.append("    boxplot prepared={")
                lines.append(f"        lower whisker={stats['min']:.6f},")
                lines.append(f"        lower quartile={stats['q25']:.6f},")
                lines.append(f"        median={stats['median']:.6f},")
                lines.append(f"        upper quartile={stats['q75']:.6f},")
                lines.append(f"        upper whisker={stats['max']:.6f},")
                lines.append("    },")
                lines.append(f"    fill={color}!50,")
                lines.append(f"    draw={color},")
                lines.append("] coordinates {};")

            lines.append("\\end{axis}")

        lines.append("\\end{tikzpicture}")

        content = "\n".join(lines)

        output_path = os.path.join(self.config.plots_dir, filename)
        uio.write_file(output_path, content)
        logger.info("Generated config performance boxplot: %s", output_path)
        return output_path

    def generate_filter_summary_table(
        self, filename: str = "filter_summary.tex"
    ) -> str:
        """Generates a summary table of filter statistics.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        stats = self.evaluator.get_filter_statistics()

        col_spec = "lrr"
        header_row = "Filter & Instances & Removed \\\\"

        total = stats["instance_counts"]["total_complete"]
        terminated = stats["instance_counts"]["all_terminated"]
        consistent = stats["instance_counts"]["all_terminated_consistent"]

        data_rows = [
            f"All complete & {total} & -- \\\\",
            f"All terminated & {terminated} & {total - terminated} \\\\",
            f"All terminated + consistent & {consistent} & {total - consistent} \\\\",
        ]

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption="Summary of instance filtering",
            label="tab:filter_summary",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated filter summary table: %s", output_path)
        return output_path

    def generate_all(self) -> dict[str, list[str]]:
        """Generates all tables and plots.

        Returns:
            Dictionary with lists of generated file paths.
        """
        tables = []
        plots = []

        # Filter summary
        tables.append(self.generate_filter_summary_table())

        # Performance tables and plots (filtered: ALL_TERMINATED_CONSISTENT)
        tables.append(self.generate_config_performance_table())
        plots.append(self.generate_config_performance_barplot())
        plots.append(self.generate_config_performance_boxplot())
        tables.append(self.generate_instance_performance_table())
        plots.append(
            self.generate_instance_performance_lineplot(
                lsf.stats_solving_time(), "instance_solving_time_lineplot.tex"
            )
        )
        plots.append(
            self.generate_instance_performance_lineplot(
                lsf.stats_mip_gap(), "instance_mip_gap_lineplot.tex"
            )
        )
        plots.append(
            self.generate_instance_performance_lineplot(
                lsf.stats_nr_nodes(), "instance_nr_nodes_lineplot.tex"
            )
        )

        # Solution tables and plots (filtered: ALL_TERMINATED)
        tables.append(self.generate_config_solution_table())
        plots.append(self.generate_config_solution_barplot())
        tables.append(self.generate_instance_solution_table())
        plots.append(
            self.generate_instance_performance_lineplot(
                lsf.stats_solution_value(), "instance_solution_value_lineplot.tex"
            )
        )
        plots.append(
            self.generate_instance_performance_lineplot(
                lsf.stats_root_solution_value(),
                "instance_root_solution_value_lineplot.tex",
            )
        )

        # Domain volume tables and plots (no filter, relative values)
        tables.append(self.generate_config_domain_volume_table())
        plots.append(self.generate_config_domain_volume_barplot())
        tables.append(self.generate_instance_domain_volume_table())
        plots.append(
            self.generate_instance_domain_volume_lineplot(
                lsf.stats_locatelli_domain_volume_polygon(),
                "instance_domain_volume_polygon_lineplot.tex",
            )
        )
        plots.append(
            self.generate_instance_domain_volume_lineplot(
                lsf.stats_locatelli_domain_volume_polytope(),
                "instance_domain_volume_polytope_lineplot.tex",
            )
        )

        # Other model property tables (no filter)
        tables.append(self.generate_config_model_size_table())
        tables.append(self.generate_instance_expressions_table())

        # Survival plot (no filter - shows complete picture)
        plots.append(self.generate_instances_solved_over_time_plot())

        return {"tables": tables, "plots": plots}
