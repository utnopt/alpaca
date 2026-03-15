# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals
"""
@authors: kuen,
"""
import os
from dataclasses import dataclass

from alpaca.study.evaluator import StudyEvaluator
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
            "_": "\\_",
            "&": "\\&",
            "%": "\\%",
            "#": "\\#",
            "{": "\\{",
            "}": "\\}",
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

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

    def generate_config_performance_table(
        self, filename: str = "config_performance.tex"
    ) -> str:
        """Generates table: Configs x (SGM solving_time, Mean mip_gap, Mean nr_nodes).

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_solving_time(),
            lsf.stats_mip_gap(),
            lsf.stats_nr_nodes(),
        ]
        data = self.evaluator.get_config_aggregated_data(
            columns, use_shifted_geom_mean=[lsf.stats_solving_time()]
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

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption="Configuration performance metrics",
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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_solving_time(),
            lsf.stats_mip_gap(),
            lsf.stats_nr_nodes(),
        ]
        data = self.evaluator.get_config_aggregated_data(
            columns, use_shifted_geom_mean=[lsf.stats_solving_time()]
        )

        configs = self.evaluator.data.configs
        symbolic_coords = ",".join(self._escape_latex(c) for c in configs)

        lines = [
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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [lsf.stats_solution_value(), lsf.stats_root_solution_value()]
        data = self.evaluator.get_config_aggregated_data(columns)

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
            caption="Configuration solution values",
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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [lsf.stats_solution_value(), lsf.stats_root_solution_value()]
        data = self.evaluator.get_config_aggregated_data(columns)

        configs = self.evaluator.data.configs
        symbolic_coords = ",".join(self._escape_latex(c) for c in configs)

        lines = [
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
        """Generates table: Configs x (Mean domain volumes).

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]
        data = self.evaluator.get_config_aggregated_data(columns)

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
            caption="Configuration domain volume metrics",
            label="tab:config_domain_volume",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated config domain volume table: %s", output_path)
        return output_path

    def generate_config_domain_volume_barplot(
        self, filename: str = "config_domain_volume_barplot.tex"
    ) -> str:
        """Generates grouped bar plot for config domain volumes.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]
        data = self.evaluator.get_config_aggregated_data(columns)

        configs = self.evaluator.data.configs
        symbolic_coords = ",".join(self._escape_latex(c) for c in configs)

        lines = [
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    ybar,",
            "    bar width=0.3cm,",
            "    width=0.9\\textwidth,",
            "    height=8cm,",
            "    ylabel={Volume},",
            f"    symbolic x coords={{{symbolic_coords}}},",
            "    xtick=data,",
            "    x tick label style={rotate=45, anchor=east},",
            "    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=-1},",
            "    ymin=0,",
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
                    lines.append(f"    ({self._escape_latex(config)}, {val:.6f})")
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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_solving_time(),
            lsf.stats_mip_gap(),
            lsf.stats_nr_nodes(),
        ]
        pivot_data = self.evaluator.get_pivot_data(columns)
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
        for instance in self.evaluator.data.instances:
            row_parts = [self._escape_latex(instance)]
            for config in configs:
                for col in columns:
                    val = pivot_data[instance][config][col]
                    row_parts.append(self._format_float(val))
            data_rows.append(" & ".join(row_parts) + " \\\\")
        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption="Instance performance metrics",
            label="tab:instance_performance",
        )
        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated instance performance table: %s", output_path)
        return output_path

    def generate_instance_performance_lineplot(self, column: str, filename: str) -> str:
        """Generates line plot for instance performance metric.

        Args:
            column: Column to plot.
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        pivot_data = self.evaluator.get_pivot_data([column])
        instances = self.evaluator.data.instances
        configs = self.evaluator.data.configs

        lines = [
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

    def generate_instance_solution_table(
        self, filename: str = "instance_solution.tex"
    ) -> str:
        """Generates table: Instances x (solution_value, root_solution_value) per config.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [lsf.stats_solution_value(), lsf.stats_root_solution_value()]
        pivot_data = self.evaluator.get_pivot_data(columns)
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
        for instance in self.evaluator.data.instances:
            row_parts = [self._escape_latex(instance)]
            for config in configs:
                for col in columns:
                    val = pivot_data[instance][config][col]
                    row_parts.append(self._format_float(val))
            data_rows.append(" & ".join(row_parts) + " \\\\")
        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption="Instance solution values",
            label="tab:instance_solution",
        )
        output_path = os.path.join(self.config.tables_dir, filename)
        uio.write_file(output_path, table_content)
        logger.info("Generated instance solution table: %s", output_path)
        return output_path

    def generate_instance_domain_volume_table(
        self, filename: str = "instance_domain_volume.tex"
    ) -> str:
        """Generates table: Instances x (domain volumes) per config.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_locatelli_domain_volume_polygon(),
            lsf.stats_locatelli_domain_volume_polytope(),
        ]
        pivot_data = self.evaluator.get_pivot_data(columns)

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
        for instance in self.evaluator.data.instances:
            row_parts = [self._escape_latex(instance)]
            for config in configs:
                for col in columns:
                    val = pivot_data[instance][config][col]
                    row_parts.append(self._format_float(val))
            data_rows.append(" & ".join(row_parts) + " \\\\")

        table_content = uio.build_latex_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption="Instance domain volume metrics",
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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_final_nr_vars(),
            lsf.stats_final_nr_constraints(),
            lsf.stats_presolved_nr_vars(),
            lsf.stats_presolved_nr_constraints(),
        ]
        data = self.evaluator.get_config_aggregated_data(columns)

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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        columns = [
            lsf.stats_pwl_nr_bilinear_expressions(),
            lsf.stats_pwl_nr_multilinear_expressions(),
            lsf.stats_pwl_nr_one_dim_expressions(),
            lsf.stats_pwl_nr_bilinear_binary_expressions(),
            lsf.stats_pwl_nr_mixed_binary_expressions(),
        ]
        data = self.evaluator.get_instance_data(columns)

        col_spec = "l" + "r" * len(columns)
        header_parts = [self._get_latex_column_name(lsf.stats_instance_name())]
        for col in columns:
            header_parts.append(self._get_latex_column_name(col))
        header_row = " & ".join(header_parts) + " \\\\"

        data_rows = []
        for instance in self.evaluator.data.instances:
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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        data = self.evaluator.get_instances_solved_over_time()
        total_instances = len(self.evaluator.data.instances)

        lines = [
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

        Args:
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
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

        lines = ["\\begin{tikzpicture}"]

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
                stats = self.evaluator.compute_statistics_for_config(column, config)

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

    def generate_all(self) -> dict[str, list[str]]:
        """Generates all tables and plots.

        Returns:
            Dictionary with lists of generated file paths.
        """
        tables = []
        plots = []
        tables.append(self.generate_config_performance_table())
        plots.append(self.generate_config_performance_barplot())
        tables.append(self.generate_config_solution_table())
        plots.append(self.generate_config_solution_barplot())
        tables.append(self.generate_config_domain_volume_table())
        plots.append(self.generate_config_domain_volume_barplot())
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
        tables.append(self.generate_instance_domain_volume_table())
        plots.append(
            self.generate_instance_performance_lineplot(
                lsf.stats_locatelli_domain_volume_polygon(),
                "instance_domain_volume_polygon_lineplot.tex",
            )
        )
        plots.append(
            self.generate_instance_performance_lineplot(
                lsf.stats_locatelli_domain_volume_polytope(),
                "instance_domain_volume_polytope_lineplot.tex",
            )
        )
        tables.append(self.generate_config_model_size_table())
        tables.append(self.generate_instance_expressions_table())
        plots.append(self.generate_instances_solved_over_time_plot())
        plots.append(self.generate_config_performance_boxplot())
        return {"tables": tables, "plots": plots}
