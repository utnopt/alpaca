# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
"""
LaTeX generator module for creating tables and TikZ plots.

This module provides functionality to generate LaTeX tables and TikZ plots
from study evaluation results.
"""
import os
from dataclasses import dataclass

from alpaca.study.evaluator import StudyEvaluator
from alpaca.utils.logger import logger


@dataclass
class LaTeXConfig:
    """Configuration for LaTeX output generation.

    Attributes:
        tables_dir: Directory for LaTeX table files.
        plots_dir: Directory for TikZ plot files.
        float_precision: Number of decimal places for floats.
        use_siunitx: Whether to use siunitx package for numbers.
        standalone: Whether to generate standalone documents.
    """

    tables_dir: str
    plots_dir: str
    float_precision: int = 3
    use_siunitx: bool = True
    standalone: bool = False


class LaTeXGenerator:
    """Generates LaTeX tables and TikZ plots from study results."""

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

    def _escape_latex(self, text: str) -> str:
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

    def generate_summary_table(
        self,
        columns: list[str] | None = None,
        filename: str = "summary_table.tex",
    ) -> str:
        """Generates a summary table with statistics per configuration.

        Args:
            columns: Columns to include. Defaults to common metrics.
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        if columns is None:
            columns = ["solving_time", "mip_gap", "nr_nodes"]

        stats = self.evaluator.get_aggregated_stats_by_config(columns)
        configs = self.evaluator.data.configs

        col_spec = "l" + "r" * (len(columns) * 3)
        header_row = "Config"
        for col in columns:
            col_name = self._escape_latex(col)
            header_row += f" & \\multicolumn{{3}}{{c}}{{{col_name}}}"
        header_row += " \\\\"

        subheader_row = ""
        for _ in columns:
            subheader_row += " & Mean & Std & Median"
        subheader_row += " \\\\"

        data_rows = []
        for config in configs:
            row = self._escape_latex(config)
            for col in columns:
                col_stats = stats[config][col]
                row += f" & {self._format_float(col_stats.mean)}"
                row += f" & {self._format_float(col_stats.std)}"
                row += f" & {self._format_float(col_stats.median)}"
            row += " \\\\"
            data_rows.append(row)

        table_content = self._build_table(
            col_spec=col_spec,
            header=header_row + "\n" + subheader_row,
            rows=data_rows,
            caption="Summary statistics by configuration",
            label="tab:summary",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        self._write_file(output_path, table_content)
        logger.info("Generated summary table: %s", output_path)

        return output_path

    def generate_results_matrix(
        self,
        value_column: str = "solving_time",
        filename: str = "results_matrix.tex",
    ) -> str:
        """Generates a matrix table with instances as rows and configs as columns.

        Args:
            value_column: Column to display in the matrix.
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        pivot = self.evaluator.get_pivot_table(value_column)
        instances = self.evaluator.data.instances
        configs = self.evaluator.data.configs

        col_spec = "l" + "r" * len(configs)
        header_row = "Instance"
        for config in configs:
            header_row += f" & {self._escape_latex(config)}"
        header_row += " \\\\"

        data_rows = []
        for instance in instances:
            row = self._escape_latex(instance)
            for config in configs:
                val = pivot[instance].get(config)
                row += f" & {self._format_float(val)}"
            row += " \\\\"
            data_rows.append(row)

        table_content = self._build_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=f"Results matrix: {self._escape_latex(value_column)}",
            label="tab:results_matrix",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        self._write_file(output_path, table_content)
        logger.info("Generated results matrix: %s", output_path)

        return output_path

    def generate_comparison_table(
        self,
        column: str = "solving_time",
        filename: str = "comparison_table.tex",
    ) -> str:
        """Generates a pairwise comparison table between configurations.

        Args:
            column: Column to compare.
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        configs = self.evaluator.data.configs

        col_spec = "l" + "c" * len(configs)
        header_row = ""
        for config in configs:
            header_row += f" & {self._escape_latex(config)}"
        header_row += " \\\\"

        data_rows = []
        for config_a in configs:
            row = self._escape_latex(config_a)
            for config_b in configs:
                if config_a == config_b:
                    row += " & --"
                else:
                    comparison = self.evaluator.compute_config_comparison(
                        config_a, config_b, column
                    )
                    ratio_str = ""
                    if comparison.geometric_mean_ratio is not None:
                        ratio_str = f" ({comparison.geometric_mean_ratio:.2f})"
                    row += f" & {comparison.wins_a}-{comparison.wins_b}{ratio_str}"
            row += " \\\\"
            data_rows.append(row)

        table_content = self._build_table(
            col_spec=col_spec,
            header=header_row,
            rows=data_rows,
            caption=f"Pairwise comparison: {self._escape_latex(column)} "
            f"(wins-losses, geom. mean ratio)",
            label="tab:comparison",
        )

        output_path = os.path.join(self.config.tables_dir, filename)
        self._write_file(output_path, table_content)
        logger.info("Generated comparison table: %s", output_path)

        return output_path

    def generate_performance_profile_plot(
        self,
        column: str = "solving_time",
        filename: str = "performance_profile.tex",
    ) -> str:
        """Generates a TikZ performance profile plot.

        Args:
            column: Column for the performance profile.
            filename: Output filename.

        Returns:
            Path to the generated file.
        """
        profile_data = self.evaluator.compute_performance_profile_data(column)

        lines = [
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    xlabel={Performance ratio $\\tau$},",
            "    ylabel={Fraction of instances},",
            "    xmin=1,",
            "    xmax=10,",
            "    ymin=0,",
            "    ymax=1,",
            "    legend pos=south east,",
            "    grid=both,",
            "    minor grid style={gray!25},",
            "    major grid style={gray!50},",
            "]",
        ]

        colors = ["blue", "red", "green!60!black", "orange", "purple", "cyan", "brown"]

        for i, (config, points) in enumerate(profile_data.items()):
            color = colors[i % len(colors)]
            config_escaped = self._escape_latex(config)

            if not points:
                continue

            lines.append(f"\\addplot[{color}, thick, mark=none] coordinates {{")
            if points[0][0] > 1:
                lines.append("    (1, 0)")

            for ratio, fraction in points:
                if ratio <= 10:
                    lines.append(f"    ({ratio:.4f}, {fraction:.4f})")

            lines.append("};")
            lines.append(f"\\addlegendentry{{{config_escaped}}}")

        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")

        content = "\n".join(lines)

        if self.config.standalone:
            content = self._wrap_standalone(content, use_pgfplots=True)

        output_path = os.path.join(self.config.plots_dir, filename)
        self._write_file(output_path, content)
        logger.info("Generated performance profile plot: %s", output_path)

        return output_path

    def generate_runtime_bar_plot(
        self,
        column: str = "solving_time",
        filename: str = "runtime_barplot.tex",
        max_instances: int = 20,
    ) -> str:
        """Generates a grouped bar plot comparing runtimes across configurations.

        Args:
            column: Column to plot.
            filename: Output filename.
            max_instances: Maximum number of instances to include.

        Returns:
            Path to the generated file.
        """
        pivot = self.evaluator.get_pivot_table(column)
        instances = self.evaluator.data.instances[:max_instances]
        configs = self.evaluator.data.configs

        colors = ["blue!70", "red!70", "green!60", "orange!70", "purple!70"]

        symbolic_coords = ",".join(self._escape_latex(i) for i in instances)

        lines = [
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "    ybar,",
            "    bar width=0.15cm,",
            "    width=\\textwidth,",
            "    height=8cm,",
            f"    ylabel={{{self._escape_latex(column)}}},",
            f"    symbolic x coords={{{symbolic_coords}}},",
            "    xtick=data,",
            "    x tick label style={rotate=45, anchor=east, font=\\tiny},",
            "    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=-1},",
            "    ymin=0,",
            "    enlarge x limits=0.05,",
            "]",
        ]

        for i, config in enumerate(configs):
            color = colors[i % len(colors)]
            config_escaped = self._escape_latex(config)

            lines.append(f"\\addplot[fill={color}] coordinates {{")
            for instance in instances:
                val = pivot.get(instance, {}).get(config)
                if val is not None:
                    lines.append(f"    ({self._escape_latex(instance)}, {val:.4f})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{config_escaped}}}")

        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")

        content = "\n".join(lines)

        if self.config.standalone:
            content = self._wrap_standalone(content, use_pgfplots=True)

        output_path = os.path.join(self.config.plots_dir, filename)
        self._write_file(output_path, content)
        logger.info("Generated runtime bar plot: %s", output_path)

        return output_path

    def _build_table(
        self,
        col_spec: str,
        header: str,
        rows: list[str],
        caption: str,
        label: str,
    ) -> str:
        """Builds a complete LaTeX table environment.

        Args:
            col_spec: Column specification (e.g., "lrrr").
            header: Header row(s) content.
            rows: List of data row strings.
            caption: Table caption.
            label: Table label for referencing.

        Returns:
            Complete LaTeX table as a string.
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            header,
            "\\midrule",
        ]
        lines.extend(rows)
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )

        return "\n".join(lines)

    def _wrap_standalone(self, content: str, use_pgfplots: bool = False) -> str:
        """Wraps content in a standalone LaTeX document.

        Args:
            content: LaTeX content to wrap.
            use_pgfplots: Whether to include pgfplots package.

        Returns:
            Complete standalone document.
        """
        packages = ["\\usepackage{booktabs}"]
        if self.config.use_siunitx:
            packages.append("\\usepackage{siunitx}")
        if use_pgfplots:
            packages.append("\\usepackage{pgfplots}")
            packages.append("\\pgfplotsset{compat=1.18}")

        header = "\n".join(
            [
                "\\documentclass[border=5pt]{standalone}",
                *packages,
                "\\begin{document}",
            ]
        )

        footer = "\\end{document}"

        return f"{header}\n{content}\n{footer}"

    def _write_file(self, path: str, content: str) -> None:
        """Writes content to a file.

        Args:
            path: Output file path.
            content: Content to write.
        """
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

    def generate_all(self) -> dict[str, list[str]]:
        """Generates all standard tables and plots.

        Returns:
            Dictionary with lists of generated file paths.
        """
        tables = [
            self.generate_summary_table(),
            self.generate_results_matrix(),
            self.generate_comparison_table(),
        ]

        plots = [
            self.generate_performance_profile_plot(),
            self.generate_runtime_bar_plot(),
        ]

        return {"tables": tables, "plots": plots}
