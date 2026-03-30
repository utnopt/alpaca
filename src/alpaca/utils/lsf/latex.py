# -*- coding: utf-8 -*-
# pylint: disable=too-many-public-methods, missing-function-docstring

"""
@authors: kuen,
"""


class Latex:
    """LaTeX-related strings for table generation."""

    @classmethod
    def begin_table(cls) -> str:
        return r"\begin{table}[htbp]"

    @classmethod
    def end_table(cls) -> str:
        return r"\end{table}"

    @classmethod
    def centering(cls) -> str:
        return r"\centering"

    @classmethod
    def begin_tabular(cls, alignment: str) -> str:
        return f"\\begin{{tabular}}{{{alignment.strip()}}}"

    @classmethod
    def end_tabular(cls) -> str:
        return r"\end{tabular}"

    @classmethod
    def toprule(cls) -> str:
        return r"\toprule"

    @classmethod
    def midrule(cls) -> str:
        return r"\midrule"

    @classmethod
    def bottomrule(cls) -> str:
        return r"\bottomrule"

    @classmethod
    def caption(cls, text: str) -> str:
        return f"\\caption{{{text}}}"

    @classmethod
    def label(cls, text: str) -> str:
        return f"\\label{{{text}}}"

    @classmethod
    def row_end(cls) -> str:
        return r" \\"

    @classmethod
    def column_separator(cls) -> str:
        return " & "

    @classmethod
    def placeholder(cls) -> str:
        return "--"

    @classmethod
    def begin_itemize(cls) -> str:
        return r"\begin{itemize}"

    @classmethod
    def end_itemize(cls) -> str:
        return r"\end{itemize}"

    @classmethod
    def item(cls, short: str, full: str) -> str:
        return f"  \\item {short}: {full}"

    @classmethod
    def underscore(cls) -> str:
        return "_"

    @classmethod
    def escape_underscore(cls, text: str) -> str:
        return text.replace("_", r"\_").replace(r"\\_", r"\_")

    @classmethod
    def math_mode(cls, value: str, bold=False) -> str:
        if bold:
            value = r"\mathbf{" + value + "}"
        return f"${value}$"

    @classmethod
    def percentage_suffix(cls) -> str:
        return r"\%"

    @classmethod
    def mean_label(cls) -> str:
        return "Mean"

    @classmethod
    def median_label(cls) -> str:
        return "Median"

    @classmethod
    def shifted_geometric_mean_label(cls) -> str:
        return "Shifted Geometric Mean"

    @classmethod
    def default_alignment(cls, num_columns: int) -> str:
        return "l " + "r " * (num_columns - 1)

    @classmethod
    def timeout_placeholder(cls) -> str:
        return r"\textit{TL}"

    # TikZ-specific methods to add to LocalizedStringFactory

    @classmethod
    def begin_tikzpicture(cls) -> str:
        return r"\begin{tikzpicture}"

    @classmethod
    def end_tikzpicture(cls) -> str:
        return r"\end{tikzpicture}"

    @classmethod
    def begin_axis(cls) -> str:
        return r"\begin{axis}["

    @classmethod
    def end_axis(cls) -> str:
        return r"\end{axis}"

    @classmethod
    def tikz_close_bracket(cls) -> str:
        return "]"

    @classmethod
    def tikz_comment(cls, text: str) -> str:
        return f"% {text}"

    @classmethod
    def tikz_scatter_plot_comment(cls) -> str:
        return "Scatter plot: configurations vs base"

    @classmethod
    def tikz_bar_plot_comment(cls) -> str:
        return "Grouped bar plot: summary statistics"

    @classmethod
    def tikz_column_comment(cls, column: str) -> str:
        return f"Column: {column}"

    @classmethod
    def tikz_instances_comment(cls, count: int) -> str:
        return f"Instances: {count}"

    @classmethod
    def tikz_diagonal_line_comment(cls) -> str:
        return "Diagonal reference line (y = x)"

    @classmethod
    def tikz_configuration_comment(cls, config: str) -> str:
        return f"Configuration: {config}"

    @classmethod
    def tikz_xlabel(cls, label: str) -> str:
        return f"xlabel={{{label}}}"

    @classmethod
    def tikz_ylabel(cls, label: str) -> str:
        return f"ylabel={{{label}}}"

    @classmethod
    def tikz_title(cls, title: str) -> str:
        return f"title={{{title}}}"

    @classmethod
    def tikz_configurations_label(cls) -> str:
        return "Configurations"

    @classmethod
    def tikz_value_label(cls) -> str:
        return "Value"

    @classmethod
    def tikz_legend_pos_north_west(cls) -> str:
        return "legend pos=north west"

    @classmethod
    def tikz_legend_cell_align_left(cls) -> str:
        return "legend cell align=left"

    @classmethod
    def tikz_legend_style_bottom(cls) -> str:
        return "legend style={at={(0.5,-0.2)}, anchor=north, legend columns=-1}"

    @classmethod
    def tikz_grid_both(cls) -> str:
        return "grid=both"

    @classmethod
    def tikz_grid_major(cls) -> str:
        return "grid=major"

    @classmethod
    def tikz_grid_style_minor(cls) -> str:
        return "grid style={line width=.1pt, draw=gray!20}"

    @classmethod
    def tikz_grid_style_major(cls) -> str:
        return "major grid style={line width=.2pt, draw=gray!50}"

    @classmethod
    def tikz_grid_style_bar(cls) -> str:
        return "grid style={line width=.1pt, draw=gray!30}"

    @classmethod
    def tikz_xmin(cls, val: float) -> str:
        return f"xmin={val}"

    @classmethod
    def tikz_xmax(cls, val: float) -> str:
        return f"xmax={val}"

    @classmethod
    def tikz_ymin(cls, val: float) -> str:
        return f"ymin={val}"

    @classmethod
    def tikz_ymax(cls, val: float) -> str:
        return f"ymax={val}"

    @classmethod
    def tikz_ymin_zero(cls) -> str:
        return "ymin=0"

    @classmethod
    def tikz_width_10cm(cls) -> str:
        return "width=10cm"

    @classmethod
    def tikz_height_10cm(cls) -> str:
        return "height=10cm"

    @classmethod
    def tikz_width_12cm(cls) -> str:
        return "width=12cm"

    @classmethod
    def tikz_height_8cm(cls) -> str:
        return "height=8cm"

    @classmethod
    def tikz_xmode_log(cls) -> str:
        return "xmode=log"

    @classmethod
    def tikz_ymode_log(cls) -> str:
        return "ymode=log"

    @classmethod
    def tikz_ybar(cls) -> str:
        return "ybar"

    @classmethod
    def tikz_bar_width(cls, width: float) -> str:
        return f"bar width={width:.2f}cm"

    @classmethod
    def tikz_enlarge_x_limits(cls) -> str:
        return "enlarge x limits=0.3"

    @classmethod
    def tikz_symbolic_x_coords_summary(cls) -> str:
        return "symbolic x coords={Mean, Median, SGM}"

    @classmethod
    def tikz_xtick_data(cls) -> str:
        return "xtick=data"

    @classmethod
    def tikz_nodes_near_coords_style(cls) -> str:
        return r"nodes near coords style={font=\tiny, rotate=90, anchor=west}"

    @classmethod
    def tikz_axis_option_line(cls, option: str, comma: str) -> str:
        return f"    {option}{comma}"

    @classmethod
    def tikz_diagonal_reference_line(cls, min_val: float, max_val: float) -> str:
        return f"\\addplot[black, dashed, thick, domain={min_val}:{max_val}] {{x}};"

    @classmethod
    def tikz_y_equals_x_label(cls) -> str:
        return "$y = x$"

    @classmethod
    def tikz_addlegendentry(cls, text: str) -> str:
        return f"\\addlegendentry{{{text}}}"

    @classmethod
    def tikz_scatter_addplot(cls, marker: str, color: str) -> str:
        return (
            f"\\addplot[only marks, mark={marker}, mark size=2pt, "
            f"color={color}, fill={color}, fill opacity=0.7] coordinates {{"
        )

    @classmethod
    def tikz_bar_addplot(cls, color: str) -> str:
        return f"\\addplot[fill={color}, draw={color}!80!black] coordinates {{"

    @classmethod
    def tikz_coordinates_end(cls) -> str:
        return "};"

    @classmethod
    def tikz_coordinate_indented(cls, x: float, y: float) -> str:
        return f"        ({x}, {y})"

    @classmethod
    def tikz_bar_coordinate(cls, label: str, value: float, precision: int) -> str:
        return f"    ({label}, {value:.{precision}f})"

    @classmethod
    def definecolor_rgb(cls, name: str, color: str) -> str:
        return f"\\definecolor{{{name}}}{{RGB}}{{{color}}}"

    @classmethod
    def summary_key_mean(cls) -> str:
        return "mean"

    @classmethod
    def summary_key_median(cls) -> str:
        return "median"

    @classmethod
    def summary_key_sgm(cls) -> str:
        return "sgm"

    @classmethod
    def sgm_short_label(cls) -> str:
        return "SGM"

    @classmethod
    def comma(cls) -> str:
        return ","

    @classmethod
    def newline(cls) -> str:
        return "\n"

    @classmethod
    def tex_extension(cls) -> str:
        return ".tex"

    @classmethod
    def bar_suffix_tex(cls) -> str:
        return "_bar.tex"

    @classmethod
    def color_blue(cls) -> str:
        return "25, 42, 86"

    @classmethod
    def color_red(cls) -> str:
        return "156, 56, 72"

    @classmethod
    def color_teal(cls) -> str:
        return "0, 119, 139"

    @classmethod
    def color_gold(cls) -> str:
        return "205, 145, 50"

    @classmethod
    def color_name(cls, color_index: int) -> str:
        return f"color{color_index}"

    @classmethod
    def marker_star(cls) -> str:
        return "*"

    @classmethod
    def marker_square(cls) -> str:
        return "square*"

    @classmethod
    def marker_triangle(cls) -> str:
        return "triangle*"

    @classmethod
    def marker_diamond(cls) -> str:
        return "diamond*"

    @classmethod
    def marker_pentagon(cls) -> str:
        return "pentagon*"

    @classmethod
    def marker_x(cls) -> str:
        return "x"

    @classmethod
    def marker_circle(cls) -> str:
        return "o"

    @classmethod
    def marker_plus(cls) -> str:
        return "+"
