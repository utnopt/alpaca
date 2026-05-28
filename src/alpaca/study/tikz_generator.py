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
class PlotFormatOptions:
    """Formatting options for plot values.

    Attributes:
        precision: Decimal precision for axis labels and values.
        title: Optional plot title.
    """

    precision: int = 2
    title: str = lsf.empty_string()


@dataclass
class PlotMetadata:
    """Metadata for a TikZ plot.

    Attributes:
        filename: Output filename (without directory).
        caption: Plot caption for LaTeX figure environment.
        label: Plot label for referencing.
    """

    filename: str
    caption: str = lsf.empty_string()
    label: str = lsf.empty_string()


@dataclass
class PlotDefinition:
    """Definition of a plot to generate.

    Attributes:
        column: Column name to plot.
        metadata: Plot metadata (filename, caption, label).
        x_axis: Column name for x-axis (optional, defaults to base config values).
        instance_filter: Filter to apply to instances.
        log_scale: Whether to use logarithmic scale for scatter plots.
        format_options: Formatting options for the plot.
    """

    column: str
    metadata: PlotMetadata
    x_axis: str = lsf.empty_string()
    instance_filter: InstanceFilter | list[InstanceFilter] = InstanceFilter.NONE
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

    COLOR_LIST: list[str] = [
        lsf.color_gold(),
        lsf.color_blue(),
        lsf.color_red(),
        lsf.color_teal(),
    ]

    MARKERS: list[str] = [
        lsf.marker_star(),
        lsf.marker_square(),
        lsf.marker_triangle(),
        lsf.marker_diamond(),
        lsf.marker_pentagon(),
        lsf.marker_x(),
        lsf.marker_circle(),
        lsf.marker_plus(),
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
        for color_index, color in enumerate(self.COLOR_LIST):
            lines.append(lsf.definecolor_rgb(lsf.color_name(color_index), color))
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
        color = lsf.color_name(config_index % len(self.COLOR_LIST))
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
        x_axis_config = (
            self.evaluator.base_config
            if definition.x_axis == lsf.empty_string()
            else definition.x_axis
        )

        base_values: dict[str, float] = {}
        config_values: dict[str, dict[str, float]] = {
            config: {} for config in self.evaluator.non_base_configs(x_axis_config)
        }

        for instance in instances:
            values = self.evaluator.get_column_values_for_instance(column, instance)
            base_val = values.get(x_axis_config)

            if base_val is None:
                continue

            base_values[instance] = base_val

            for config in self.evaluator.non_base_configs(base_config=x_axis_config):
                val = values.get(config)
                if val is not None:
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
        x_axis_config = (
            self.evaluator.base_config
            if definition.x_axis == lsf.empty_string()
            else definition.x_axis
        )
        base_short = self._config_shortnames.get(x_axis_config, x_axis_config)

        options = [
            lsf.tikz_xlabel(base_short),
            lsf.tikz_legend_pos_north_west(),
            lsf.tikz_legend_cell_align_left(),
            lsf.tikz_grid_both(),
            lsf.tikz_grid_style_minor(),
            lsf.tikz_grid_style_major(),
            lsf.tikz_xmin(min_val),
            lsf.tikz_xmax(max_val),
            lsf.tikz_ymin(min_val),
            lsf.tikz_ymax(max_val),
            lsf.tikz_width_10cm(),
            lsf.tikz_height_10cm(),
        ]

        if definition.log_scale:
            options.extend([lsf.tikz_xmode_log(), lsf.tikz_ymode_log()])

        if definition.format_options.title:
            options.append(lsf.tikz_title(definition.format_options.title))

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
                coords.append(lsf.tikz_coordinate_indented(base_val, y_val))
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
        x_axis_config = (
            self.evaluator.base_config
            if definition.x_axis == lsf.empty_string()
            else definition.x_axis
        )

        lines = [
            lsf.tikz_comment(lsf.tikz_scatter_plot_comment()),
            lsf.tikz_comment(lsf.tikz_column_comment(definition.column)),
            lsf.tikz_comment(lsf.tikz_instances_comment(len(base_values))),
            lsf.empty_string(),
        ]

        lines.extend(self._get_color_definitions())
        lines.append(lsf.empty_string())

        lines.append(lsf.begin_tikzpicture())
        lines.append(lsf.begin_axis())

        axis_options = self._build_scatter_axis_options(definition, min_val, max_val)
        for i, opt in enumerate(axis_options):
            comma = lsf.comma() if i < len(axis_options) - 1 else lsf.empty_string()
            lines.append(lsf.tikz_axis_option_line(opt, comma))

        lines.append(lsf.tikz_close_bracket())
        lines.append(lsf.empty_string())

        lines.append(lsf.tikz_comment(lsf.tikz_diagonal_line_comment()))
        lines.append(lsf.tikz_diagonal_reference_line(min_val, max_val))
        lines.append(lsf.empty_string())

        for i, config in enumerate(
            self.evaluator.non_base_configs(base_config=x_axis_config)
        ):
            if not config_values[config]:
                continue
            color, marker = self._get_config_style(i)

            lines.append(lsf.tikz_comment(lsf.tikz_configuration_comment(config)))
            lines.append(lsf.tikz_scatter_addplot(marker, color))

            lines.extend(
                self._build_scatter_coordinates(base_values, config_values[config])
            )

            lines.append(lsf.tikz_coordinates_end())
            lines.append(
                lsf.tikz_addlegendentry(self._config_shortnames.get(config, config))
            )
            lines.append(lsf.empty_string())

        lines.append(lsf.end_axis())
        lines.append(lsf.end_tikzpicture())
        lines.append(
            f"% Nr of instances: {len(
            config_values[self.evaluator.non_base_configs(
                base_config=x_axis_config)[-1]])}"
        )

        return lsf.newline().join(lines)

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
                lsf.summary_key_mean(): mean_val,
                lsf.summary_key_median(): median_val,
                lsf.summary_key_sgm(): sgm_val,
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
            lsf.tikz_ybar(),
            lsf.tikz_bar_width(bar_width),
            lsf.tikz_enlarge_x_limits(),
            lsf.tikz_legend_style_bottom(),
            lsf.tikz_legend_cell_align_left(),
            lsf.tikz_ylabel(lsf.tikz_value_label()),
            lsf.tikz_symbolic_x_coords_summary(),
            lsf.tikz_xtick_data(),
            lsf.tikz_grid_major(),
            lsf.tikz_grid_style_bar(),
            lsf.tikz_ymin_zero(),
            lsf.tikz_width_12cm(),
            lsf.tikz_height_8cm(),
            lsf.tikz_nodes_near_coords_style(),
        ]

        if definition.format_options.title:
            options.append(lsf.tikz_title(definition.format_options.title))

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

        lines.append(lsf.tikz_comment(lsf.tikz_bar_plot_comment()))
        lines.append(lsf.tikz_comment(lsf.tikz_column_comment(definition.column)))
        lines.append(lsf.empty_string())

        lines.extend(self._get_color_definitions())
        lines.append(lsf.empty_string())

        lines.append(lsf.begin_tikzpicture())
        lines.append(lsf.begin_axis())

        axis_options = self._build_bar_axis_options(definition)
        for i, opt in enumerate(axis_options):
            comma = lsf.comma() if i < len(axis_options) - 1 else lsf.empty_string()
            lines.append(lsf.tikz_axis_option_line(opt, comma))

        lines.append(lsf.tikz_close_bracket())
        lines.append(lsf.empty_string())

        precision = definition.format_options.precision

        for i, config in enumerate(self.evaluator.data.configs):
            color, _ = self._get_config_style(i)

            data = summary_data[config]
            mean_val = (
                data[lsf.summary_key_mean()]
                if data[lsf.summary_key_mean()] is not None
                else 0
            )
            median_val = (
                data[lsf.summary_key_median()]
                if data[lsf.summary_key_median()] is not None
                else 0
            )
            sgm_val = (
                data[lsf.summary_key_sgm()]
                if data[lsf.summary_key_sgm()] is not None
                else 0
            )

            lines.append(lsf.tikz_comment(lsf.tikz_configuration_comment(config)))
            lines.append(lsf.tikz_bar_addplot(color))
            lines.append(lsf.tikz_bar_coordinate(lsf.mean_label(), mean_val, precision))
            lines.append(
                lsf.tikz_bar_coordinate(lsf.median_label(), median_val, precision)
            )
            lines.append(
                lsf.tikz_bar_coordinate(lsf.sgm_short_label(), sgm_val, precision)
            )
            lines.append(lsf.tikz_coordinates_end())
            lines.append(
                lsf.tikz_addlegendentry(self._config_shortnames.get(config, config))
            )
            lines.append(lsf.empty_string())

        lines.append(lsf.end_axis())
        lines.append(lsf.end_tikzpicture())

        return lsf.newline().join(lines)

    def save_plot(self, tikz_code: str, filename: str) -> Path:
        """Save TikZ code to a file.

        Args:
            tikz_code: TikZ code to save.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        filepath = self._output_dir / filename
        with open(
            filepath, lsf.file_mode_write(), encoding=lsf.file_encoding_utf8()
        ) as file:
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
        bar_filename = definition.metadata.filename.replace(
            lsf.tex_extension(), lsf.bar_suffix_tex()
        )
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
                column=lsf.stats_root_gap_reduction(),
                x_axis="locatelli",
                metadata=PlotMetadata(
                    filename="plot_root_gap_reduction.tex",
                    caption="Root gap reduction comparison.",
                    label="plot:root_gap_reduction",
                ),
                instance_filter=[
                    InstanceFilter.ROOT_SUBOPTIMAL,
                    InstanceFilter.NON_EMPTY_BILINEAR_DOMAIN,
                ],
                log_scale=False,
                format_options=PlotFormatOptions(
                    precision=2, title="Root Gap Reduction"
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
            PlotDefinition(
                column=lsf.stats_volume_reduction_polygon(),
                x_axis="locatelli",
                metadata=PlotMetadata(
                    filename="plot_volume_reduction_polygon.tex",
                    caption="Volume reduction over polygon.",
                    label="plot:volume_reduction_polygon",
                ),
                instance_filter=InstanceFilter.NON_EMPTY_BILINEAR_DOMAIN,
                log_scale=False,
                format_options=PlotFormatOptions(
                    precision=2, title="Volume Reduction over Polygon"
                ),
            ),
            PlotDefinition(
                column=lsf.stats_volume_reduction_polytope(),
                x_axis="locatelli",
                metadata=PlotMetadata(
                    filename="plot_volume_reduction_polytope.tex",
                    caption="Volume reduction over polytope.",
                    label="plot:volume_reduction_polytope",
                ),
                instance_filter=InstanceFilter.NON_EMPTY_BILINEAR_DOMAIN,
                log_scale=False,
                format_options=PlotFormatOptions(
                    precision=2, title="Volume Reduction over Polytope"
                ),
            ),
        ]

    def generate_all_predefined_plots(self) -> list[Path]:
        """Generate all predefined plots.

        Returns:
            List of paths to all generated plot files.
        """
        definitions = self.get_predefined_plot_definitions()
        return self.generate_plots(definitions)
