# -*- coding: utf-8 -*-

"""
TikZ plot generator for study evaluation results.

Generates scatter plots and grouped bar plots as TikZ/pgfplots code.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.study.evaluator import StudyEvaluator, InstanceFilter


@dataclass
class PlotFormatOptions:
    """Formatting options for plot values.

    Attributes:
        precision: Decimal precision for axis labels and values.
        title: Optional plot title.
    """

    precision: int = 2
    title: str = ""


@dataclass
class PlotMetadata:
    """Metadata for a TikZ plot.

    Attributes:
        filename: Output filename (without directory).
        caption: Plot caption for LaTeX figure environment.
        label: Plot label for referencing.
    """

    filename: str
    caption: str = ""
    label: str = ""


@dataclass
class PlotDefinition:
    """Definition of a plot to generate.

    Attributes:
        column: Column name to plot.
        metadata: Plot metadata (filename, caption, label).
        instance_filter: Filter to apply to instances.
        log_scale: Whether to use logarithmic scale for scatter plots.
        format_options: Formatting options for the plot.
    """

    column: str
    metadata: PlotMetadata
    instance_filter: InstanceFilter = InstanceFilter.NONE
    log_scale: bool = False
    format_options: PlotFormatOptions = field(default_factory=PlotFormatOptions)


@dataclass
class PlotGeneratorConfig:
    """Configuration for the TikZ plot generator.

    Attributes:
        evaluator: StudyEvaluator instance with loaded data.
        output_dir: Directory to save generated TikZ files.
        config_shortener: Optional function to shorten config names.
    """

    evaluator: StudyEvaluator
    output_dir: str
    config_shortener: Callable[[str], str] | None = None


class TikzPlotGenerator:
    """Generates TikZ plots from study evaluation results.

    Creates scatter plots comparing configurations against a base configuration,
    and grouped bar plots for summary statistics (mean, median, shifted geometric mean).
    """

    COLORS: dict[str, tuple[int, int, int]] = {
        "MidnightNavy": (25, 42, 86),
        "BrickRose": (156, 56, 72),
        "OceanTeal": (0, 119, 139),
        "AmberGold": (205, 145, 50),
    }

    COLOR_LIST: list[str] = ["MidnightNavy", "BrickRose", "OceanTeal", "AmberGold"]

    MARKERS: list[str] = [
        "*",
        "square*",
        "triangle*",
        "diamond*",
        "pentagon*",
        "x",
        "o",
        "+",
    ]

    def __init__(self, config: PlotGeneratorConfig) -> None:
        """Initialize the TikZ plot generator.

        Args:
            config: Generator configuration.
        """
        self._config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config_shortnames = self.evaluator.build_config_shortnames()

    def _get_color_definitions(self) -> list[str]:
        """Generate LaTeX color definition lines."""
        lines = []
        for name, (r, g, b) in self.COLORS.items():
            lines.append(f"\\definecolor{{{name}}}{{RGB}}{{{r}, {g}, {b}}}")
        return lines

    @property
    def evaluator(self) -> StudyEvaluator:
        """Get the study evaluator."""
        return self._config.evaluator

    def _get_config_style(self, config_index: int) -> tuple[str, str]:
        """Get color and marker for a config by index.

        Args:
            config_index: Index of the configuration.

        Returns:
            Tuple of (color_name, marker_style).
        """
        color = self.COLOR_LIST[config_index % len(self.COLOR_LIST)]
        marker = self.MARKERS[config_index % len(self.MARKERS)]
        return color, marker

    def _collect_scatter_data(
        self, definition: PlotDefinition
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Collect data points for scatter plot.

        Args:
            definition: Plot definition.

        Returns:
            Tuple of (base_values, config_values) dictionaries.
        """
        instances = self.evaluator.get_filtered_instances(definition.instance_filter)
        column = definition.column

        base_values: dict[str, float] = {}
        config_values: dict[str, dict[str, float]] = {
            config: {} for config in self.evaluator.non_base_configs
        }

        for instance in instances:
            values = self.evaluator.get_column_values_for_instance(column, instance)
            base_val = values.get(self.evaluator.base_config)

            if base_val is None or base_val <= 0:
                continue

            base_values[instance] = base_val

            for config in self.evaluator.non_base_configs:
                val = values.get(config)
                if val is not None and val > 0:
                    config_values[config][instance] = val

        return base_values, config_values

    def _compute_axis_limits(
        self,
        base_values: dict[str, float],
        config_values: dict[str, dict[str, float]],
    ) -> tuple[float, float]:
        """Compute axis limits for scatter plot.

        Args:
            base_values: Base configuration values.
            config_values: Other configuration values.

        Returns:
            Tuple of (min_val, max_val) for axis limits.
        """
        all_values = list(base_values.values())
        for cfg_vals in config_values.values():
            all_values.extend(cfg_vals.values())

        if not all_values:
            return 0.1, 100.0

        min_val = min(all_values)
        max_val = max(all_values)

        margin = 0.1
        if min_val > 0:
            min_val = min_val * (1 - margin)
        max_val = max_val * (1 + margin)

        return min_val, max_val

    def _build_scatter_axis_options(
        self,
        definition: PlotDefinition,
        min_val: float,
        max_val: float,
    ) -> list[str]:
        """Build axis options for scatter plot.

        Args:
            definition: Plot definition.
            min_val: Minimum axis value.
            max_val: Maximum axis value.

        Returns:
            List of axis option strings.
        """
        base_short = self._config_shortnames.get(
            self.evaluator.base_config, self.evaluator.base_config
        )

        options = [
            f"xlabel={{{base_short}}}",
            "ylabel={Configurations}",
            "legend pos=north west",
            "legend cell align=left",
            "grid=both",
            "grid style={line width=.1pt, draw=gray!20}",
            "major grid style={line width=.2pt, draw=gray!50}",
            f"xmin={min_val}",
            f"xmax={max_val}",
            f"ymin={min_val}",
            f"ymax={max_val}",
            "width=10cm",
            "height=10cm",
        ]

        if definition.log_scale:
            options.extend(["xmode=log", "ymode=log"])

        if definition.format_options.title:
            options.append(f"title={{{definition.format_options.title}}}")

        return options

    def _build_scatter_coordinates(
        self,
        base_values: dict[str, float],
        config_vals: dict[str, float],
    ) -> list[str]:
        """Build coordinate lines for a scatter plot series.

        Args:
            base_values: Base configuration values by instance.
            config_vals: Configuration values by instance.

        Returns:
            List of coordinate strings.
        """
        coords = []
        for instance, base_val in base_values.items():
            if instance in config_vals:
                y_val = config_vals[instance]
                coords.append(f"        ({base_val}, {y_val})")
        return coords

    def generate_scatter_plot(self, definition: PlotDefinition) -> str:
        """Generate a scatter plot comparing configs against base.

        Args:
            definition: Plot definition.

        Returns:
            TikZ code as string.
        """
        base_values, config_values = self._collect_scatter_data(definition)
        min_val, max_val = self._compute_axis_limits(base_values, config_values)

        lines = [
            "% Scatter plot: configurations vs base",
            f"% Column: {definition.column}",
            f"% Instances: {len(base_values)}",
            ""
            ]

        lines.extend(self._get_color_definitions())
        lines.append("")

        lines.append("\\begin{tikzpicture}")
        lines.append("\\begin{axis}[")

        axis_options = self._build_scatter_axis_options(definition, min_val, max_val)
        for i, opt in enumerate(axis_options):
            comma = "," if i < len(axis_options) - 1 else ""
            lines.append(f"    {opt}{comma}")

        lines.append("]")
        lines.append("")

        lines.append("% Diagonal reference line (y = x)")
        lines.append(
            f"\\addplot[black, dashed, thick, domain={min_val}:{max_val}] {{x}};"
        )
        lines.append("\\addlegendentry{$y = x$}")
        lines.append("")

        for i, config in enumerate(self.evaluator.non_base_configs):
            color, marker = self._get_config_style(i)
            short_name = self._config_shortnames.get(config, config)

            lines.append(f"% Configuration: {config}")
            lines.append(
                f"\\addplot[only marks, mark={marker}, mark size=2pt, "
                f"color={color}, fill={color}, fill opacity=0.7] coordinates {{"
            )

            lines.extend(self._build_scatter_coordinates(base_values, config_values[config]))

            lines.append("};")
            lines.append(f"\\addlegendentry{{{short_name}}}")
            lines.append("")

        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")

        return "\n".join(lines)

    def _collect_summary_data(
        self, definition: PlotDefinition
    ) -> dict[str, dict[str, float | None]]:
        """Collect summary statistics for bar plot.

        Args:
            definition: Plot definition.

        Returns:
            Dictionary mapping config to {mean, median, sgm} values.
        """
        column = definition.column
        configs = self.evaluator.data.configs

        summary_data: dict[str, dict[str, float | None]] = {}

        for config in configs:
            mean_val = self.evaluator.data.data_matrix.get(
                (lsf.mean_label(), config), {}
            ).get(column)

            median_val = self.evaluator.data.data_matrix.get(
                (lsf.median_label(), config), {}
            ).get(column)

            sgm_val = self.evaluator.data.data_matrix.get(
                (lsf.shifted_geometric_mean_label(), config), {}
            ).get(column)

            summary_data[config] = {
                "mean": mean_val,
                "median": median_val,
                "sgm": sgm_val,
            }

        return summary_data

    def _build_bar_axis_options(self, definition: PlotDefinition) -> list[str]:
        """Build axis options for grouped bar plot.

        Args:
            definition: Plot definition.

        Returns:
            List of axis option strings.
        """
        num_configs = len(self.evaluator.data.configs)
        bar_width = max(0.08, 0.6 / num_configs)

        options = [
            "ybar",
            f"bar width={bar_width:.2f}cm",
            "enlarge x limits=0.3",
            "legend style={at={(0.5,-0.2)}, anchor=north, legend columns=-1}",
            "legend cell align=left",
            "ylabel={Value}",
            "symbolic x coords={Mean, Median, SGM}",
            "xtick=data",
            "grid=major",
            "grid style={line width=.1pt, draw=gray!30}",
            "ymin=0",
            "width=12cm",
            "height=8cm",
            r"nodes near coords style={font=\tiny, rotate=90, anchor=west}",
        ]

        if definition.format_options.title:
            options.append(f"title={{{definition.format_options.title}}}")

        return options

    def generate_grouped_bar_plot(self, definition: PlotDefinition) -> str:
        """Generate a grouped bar plot for mean/median/sgm values.

        Args:
            definition: Plot definition.

        Returns:
            TikZ code as string.
        """
        summary_data = self._collect_summary_data(definition)

        lines: list[str] = []

        lines.append("% Grouped bar plot: summary statistics")
        lines.append(f"% Column: {definition.column}")
        lines.append("")

        lines.extend(self._get_color_definitions())
        lines.append("")

        lines.append("\\begin{tikzpicture}")
        lines.append("\\begin{axis}[")

        axis_options = self._build_bar_axis_options(definition)
        for i, opt in enumerate(axis_options):
            comma = "," if i < len(axis_options) - 1 else ""
            lines.append(f"    {opt}{comma}")

        lines.append("]")
        lines.append("")

        precision = definition.format_options.precision

        for i, config in enumerate(self.evaluator.data.configs):
            color, _ = self._get_config_style(i)

            data = summary_data[config]
            mean_val = data["mean"] if data["mean"] is not None else 0
            median_val = data["median"] if data["median"] is not None else 0
            sgm_val = data["sgm"] if data["sgm"] is not None else 0

            lines.append(f"% Configuration: {config}")
            lines.append(
                f"\\addplot[fill={color}, draw={color}!80!black] coordinates {{"
            )
            lines.append(f"    (Mean, {mean_val:.{precision}f})")
            lines.append(f"    (Median, {median_val:.{precision}f})")
            lines.append(f"    (SGM, {sgm_val:.{precision}f})")
            lines.append("};")
            lines.append(f"\\addlegendentry{{{self._config_shortnames.get(config, config)}}}")
            lines.append("")

        lines.append("\\end{axis}")
        lines.append("\\end{tikzpicture}")

        return "\n".join(lines)

    def save_plot(self, tikz_code: str, filename: str) -> Path:
        """Save TikZ code to a file.

        Args:
            tikz_code: TikZ code to save.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        filepath = self._output_dir / filename
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(tikz_code)
        return filepath

    def generate_and_save_scatter_plot(self, definition: PlotDefinition) -> Path:
        """Generate and save a scatter plot.

        Args:
            definition: Plot definition.

        Returns:
            Path to the saved file.
        """
        tikz_code = self.generate_scatter_plot(definition)
        return self.save_plot(tikz_code, definition.metadata.filename)

    def generate_and_save_bar_plot(self, definition: PlotDefinition) -> Path:
        """Generate and save a grouped bar plot.

        Args:
            definition: Plot definition.

        Returns:
            Path to the saved file.
        """
        tikz_code = self.generate_grouped_bar_plot(definition)
        bar_filename = definition.metadata.filename.replace(".tex", "_bar.tex")
        return self.save_plot(tikz_code, bar_filename)

    def generate_plots_from_definition(self, definition: PlotDefinition) -> list[Path]:
        """Generate both scatter and bar plots from a definition.

        Args:
            definition: Plot definition.

        Returns:
            List of paths to saved files.
        """
        scatter_path = self.generate_and_save_scatter_plot(definition)
        bar_path = self.generate_and_save_bar_plot(definition)
        return [scatter_path, bar_path]

    def generate_plots(self, definitions: list[PlotDefinition]) -> list[Path]:
        """Generate and save multiple plots from definitions.

        Args:
            definitions: List of plot definitions.

        Returns:
            List of paths to all saved files.
        """
        paths: list[Path] = []
        for defn in definitions:
            paths.extend(self.generate_plots_from_definition(defn))
        return paths

    def get_predefined_plot_definitions(self) -> list[PlotDefinition]:
        """Get predefined plot definitions.

        Returns:
            List of predefined PlotDefinition objects.
        """
        return [
            PlotDefinition(
                column=lsf.stats_solving_time(),
                metadata=PlotMetadata(
                    filename="plot_solving_time.tex",
                    caption="Solving time comparison across configurations.",
                    label="plot:solving_time",
                ),
                instance_filter=InstanceFilter.NONE,
                log_scale=True,
                format_options=PlotFormatOptions(precision=2, title="Solving Time"),
            ),
            PlotDefinition(
                column=lsf.stats_nr_nodes(),
                metadata=PlotMetadata(
                    filename="plot_nr_nodes.tex",
                    caption="Number of branch-and-bound nodes comparison.",
                    label="plot:nr_nodes",
                ),
                instance_filter=InstanceFilter.BRANCH_AND_BOUND,
                log_scale=True,
                format_options=PlotFormatOptions(
                    precision=0, title="Branch-and-Bound Nodes"
                ),
            ),
            PlotDefinition(
                column=lsf.stats_root_solving_time(),
                metadata=PlotMetadata(
                    filename="plot_root_solving_time.tex",
                    caption="Root node solving time comparison.",
                    label="plot:root_solving_time",
                ),
                instance_filter=InstanceFilter.ALL_REACHED_ROOT,
                log_scale=True,
                format_options=PlotFormatOptions(
                    precision=2, title="Root Node Solving Time"
                ),
            ),
            PlotDefinition(
                column=lsf.stats_mip_gap(),
                metadata=PlotMetadata(
                    filename="plot_mip_gap.tex",
                    caption="MIP gap comparison across configurations.",
                    label="plot:mip_gap",
                ),
                instance_filter=InstanceFilter.ALL_FOUND_SOLUTION,
                log_scale=False,
                format_options=PlotFormatOptions(precision=4, title="MIP Gap"),
            ),
        ]

    def generate_all_predefined_plots(self) -> list[Path]:
        """Generate all predefined plots.

        Returns:
            List of paths to all generated plot files.
        """
        definitions = self.get_predefined_plot_definitions()
        return self.generate_plots(definitions)
