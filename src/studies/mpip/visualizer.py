# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals
"""
@authors: kuen,
"""
import os
import datetime
import numpy as np
import pandas as pd

import alpaca.settings as s
from alpaca.mpip import mpip_handler as mph, mpip as mpi
from alpaca.mpip.separation import mpip_separator as mse
import alpaca.external_solvers.solver_wrapper as swr
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class Visualizer:
    """Visualizer for mpip studies."""

    def __init__(self, user_settings: s.UserSettings):
        self.user_settings = user_settings
        self.results_df: pd.DataFrame | None = None

    def plot_blocks(self, mpip_handler: mph.MPIPHandler):
        """Plots block structures for MPIP structures with 2 implying variables using TikZ."""
        logger.info(lsf.study_mpip_info_plot_blocks())
        out_path = (
            self.user_settings.export_path + lsf.study_mpip_export_folder_blocks()
        )
        os.makedirs(out_path, exist_ok=True)
        for mpip in mpip_handler.mpip_dict.values():
            if len(mpip.implying_variables) != 2:
                continue
            for cut in self._find_good_cuts(mpip.separator):
                blocks = self._get_block_structure(mpip, cut)
                self._plot_blocks_tikz(mpip, out_path, cut.name, blocks)

    @staticmethod
    def _find_good_cuts(separator: mse.MPIPSeparator):
        return [
            cut
            for cut in separator.separated_cuts
            if cut.rhs - cut.lhs.getValue() < s.StaticSettings.feasibility_tolerance
        ]

    @staticmethod
    def _get_block_structure(mpip: mpi.MPIP, cut: swr.GurobiCut):
        lhs_expr = cut.lhs
        coeff_dict = {
            variable.VarName: coeff for coeff, variable in lhs_expr.linTerms()
        }
        implying_vars = list(mpip.implying_variables.values())
        blocks = [[] for _ in range(s.StaticSettings.nr_of_blocks_plotted)]
        for block_index in range(s.StaticSettings.nr_of_blocks_plotted):
            for x_var in implying_vars[0].pwl.pwl_variables_binary:
                x_coeff = coeff_dict[x_var.name]
                for y_var in implying_vars[1].pwl.pwl_variables_binary:
                    y_coeff = coeff_dict[y_var.name]
                    if x_coeff + s.StaticSettings.feasibility_tolerance >= (
                        block_index + 1
                    ) * (
                        1 / s.StaticSettings.nr_of_blocks_plotted
                    ) and y_coeff + s.StaticSettings.feasibility_tolerance >= (
                        s.StaticSettings.nr_of_blocks_plotted - block_index
                    ) * (
                        1 / s.StaticSettings.nr_of_blocks_plotted
                    ):
                        blocks[block_index].append((x_var.name, y_var.name))
        return blocks

    @staticmethod
    def _plot_blocks_tikz(
        mpip: mpi.MPIP,
        out_path: str,
        cut_name: str,
        blocks: list[list[tuple[str, str]]],
    ):
        """
        Generates a LaTeX TikZ file for the block structure.
        """
        file_path = os.path.join(out_path, f"{cut_name}.tex")

        implying_vars = list(mpip.implying_variables.values())
        x_vars_list = implying_vars[0].pwl.pwl_variables_binary
        y_vars_list = implying_vars[1].pwl.pwl_variables_binary

        n_rows = len(x_vars_list)
        n_cols = len(y_vars_list)

        latex_content = [
            r"\documentclass[tikz, border=10pt]{standalone}",
            r"\begin{document}",
            r"\begin{tabular}{" + "c" * len(blocks) + "}",
        ]

        row_content = []

        for block_idx, block in enumerate(blocks):
            tikz_code = [
                r"\begin{tikzpicture}[scale=0.5, font=\tiny]",
                # Draw grid
                f"\\draw[step=1.0,gray,thin] (0,0) grid ({n_cols},{n_rows});",
            ]
            for i in range(n_rows):
                for j in range(n_cols):
                    x_var = x_vars_list[i]
                    y_var = y_vars_list[j]
                    x_coord = j
                    y_coord = (n_rows - 1) - i

                    label = mpip.relation[(i, j)]

                    if (x_var.name, y_var.name) in block:
                        # Blue cell with white text
                        tikz_code.append(
                            f"\\fill[blue] ({x_coord},{y_coord}) rectangle ++(1,1);"
                        )
                        text_color = "white"
                    else:
                        # White cell (default) with black text
                        text_color = "black"

                    # Escape underscores in label for LaTeX
                    safe_label = str(label).replace("_", r"\_")
                    tikz_code.append(
                        f"\\node[text={text_color}, anchor=center] at"
                        f" ({x_coord + 0.5},{y_coord + 0.5}) {{{safe_label}}};"
                    )

            # Axes ticks (simplified)
            for j in range(n_cols):
                tikz_code.append(f"\\draw ({j + 0.5}, -0.1) -- ({j + 0.5}, 0);")
            for i in range(n_rows):
                y_coord = (n_rows - 1) - i
                tikz_code.append(
                    f"\\draw (-0.1, {y_coord + 0.5}) -- (0, {y_coord + 0.5});"
                )

            tikz_code.append(
                f"\\node[anchor=south] at ({n_cols / 2}, {n_rows}) {{Block {block_idx + 1}}};"
            )
            tikz_code.append(r"\end{tikzpicture}")

            row_content.append("\n".join(tikz_code))

        latex_content.append(" & ".join(row_content) + r" \\")
        latex_content.append(r"\end{tabular}")
        latex_content.append(r"\end{document}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_content))

    def create_performance_plots(self, results_csv: str):
        """
        Creates performance plots (TikZ) from the CSV results.
        """
        logger.info(lsf.study_mpip_info_plot_performance())
        input_csv = s.StaticSettings.import_path + results_csv
        out_path = (
            self.user_settings.export_path + lsf.study_mpip_export_folder_performance()
        )
        self._load_and_prep_data(input_csv)

        if self.results_df is not None:

            def to_tex(filename):
                base = os.path.splitext(filename)[0]
                return base + ".tex"

            self._plot_performance_profile_tikz(
                out_path + to_tex(lsf.study_mpip_performance_plot_name())
            )
            self._plot_runtime_by_breakpoints_tikz(
                out_path + to_tex(lsf.study_mpip_runtime_scaling_plot_name())
            )
            self._plot_instances_solved_over_time_tikz(
                out_path + to_tex(lsf.study_mpip_solved_instances_plot_name())
            )

    def create_latex_tables(self, results_csv: str):
        """
        Creates LaTeX tables from the CSV results."""
        logger.info(lsf.study_mpip_info_plot_performance())
        input_csv = s.StaticSettings.import_path + results_csv
        out_path = (
            self.user_settings.export_path + lsf.study_mpip_export_folder_latex_tables()
        )
        self._load_and_prep_data(input_csv)
        if self.results_df is not None:
            self._generate_latex_table_runtime(
                out_path + lsf.study_mpip_runtime_table_name()
            )

    def _load_and_prep_data(self, filepath):
        """
        Loads the CSV and creates a unique 'Method' identifier for the 4 configurations.
        """
        if not os.path.exists(filepath):
            return

        df = pd.read_csv(filepath)
        df["Method"] = df["test_case"] + " " + df["pwl_method"]
        df["Method"] = df["Method"].apply(lambda m: m.replace("_", r"\_"))
        df["Instance"] = (
            df["osil_file_name"]
            + "_"
            + df["nr_of_breakpoints"].astype(str)
            + "_"
            + df["seed"].astype(str)
        )
        df["runtime"] = df["runtime"].apply(lambda r: r if r <= 18000 else 18000)
        self.results_df = df

    def _plot_performance_profile_tikz(self, output_file):
        """
        Generates a Dolan-More performance profile as a standalone TikZ file.
        """
        pivot_df = self.results_df.pivot(
            index="Instance", columns="Method", values="runtime"
        )
        pivot_df = pivot_df.fillna(np.inf)
        min_times = pivot_df.min(axis=1)
        ratios = pivot_df.div(min_times, axis=0)
        methods = pivot_df.columns

        latex = [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"    width=\textwidth,",
            r"    height=8cm,",
            r"    xmode=log,",
            r"    xlabel={Performance Ratio ($\tau$)},",
            r"    ylabel={Fraction of problems solved within factor $\tau$},",
            r"    title={Performance Profile (Runtime)},",
            r"    grid=major,",
            r"    xmin=1, xmax=100,",
            r"    legend pos=south east,",
            r"]",
        ]

        for method in methods:
            method_ratios = ratios[method]
            sorted_ratios = np.sort(method_ratios)
            yvals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

            # Construct coordinate string for tikz
            coords = []
            for r, y in zip(sorted_ratios, yvals):
                if r <= 100:  # Optimization: Don't plot extremely large ratios
                    coords.append(f"({r:.4f}, {y:.4f})")

            # Ensure the line extends to the end if max ratio < xmax
            if len(sorted_ratios) > 0 and sorted_ratios[-1] < 100:
                coords.append(f"(100, {yvals[-1]:.4f})")

            coord_str = " ".join(coords)
            latex.append(f"\\addplot[const plot, thick] coordinates {{ {coord_str} }};")
            latex.append(f"\\addlegendentry{{{method}}}")

        latex.append(r"\end{axis}")
        latex.append(r"\end{tikzpicture}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(latex))

    def _plot_runtime_by_breakpoints_tikz(self, output_file):
        """
        Generates a Boxplot comparing runtimes using PGFPlots.
        Since pgfplots statistics can be heavy, we pre-calculate quantiles in Python.
        """
        latex = [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"    ymode=log,",
            r"    width=\textwidth,",
            r"    height=8cm,",
            r"    xlabel={Number of Breakpoints},",
            r"    ylabel={Runtime (s) [Log Scale]},",
            r"    title={Runtime Distribution by Breakpoint Count},",
            r"    grid=major,",
            r"    xtick=data,",
            r"    legend style={at={(1.05,1)}, anchor=north west},",
            r"    width=12cm, height=8cm,",
            r"]",
        ]

        # Group data
        methods = self.results_df["Method"].unique()
        breakpoints = sorted(self.results_df["nr_of_breakpoints"].unique())

        # We need to manually dodge the boxplots.
        # Base x-locations are the breakpoints. We add offsets.
        num_methods = len(methods)
        # width of a group of boxplots
        width_per_group = (
            0.8 * (breakpoints[1] - breakpoints[0]) if len(breakpoints) > 1 else 1.0
        )
        # width of a single boxplot
        box_width = width_per_group / (num_methods + 1)

        # Define a color cycle manually or use cycle list
        colors = ["blue", "red", "green", "orange", "purple"]

        for idx, method in enumerate(methods):
            color = colors[idx % len(colors)]
            latex.append(f"% Method: {method}")

            # For each breakpoint, calculate stats
            for bp in breakpoints:
                subset = self.results_df[
                    (self.results_df["Method"] == method)
                    & (self.results_df["nr_of_breakpoints"] == bp)
                ]["runtime"]

                if subset.empty:
                    continue

                # Calculate boxplot statistics
                q1 = subset.quantile(0.25)
                q3 = subset.quantile(0.75)
                median = subset.median()
                # Whiskers (1.5 IQR)
                iqr = q3 - q1
                lower_whisker = subset[subset >= q1 - 1.5 * iqr].min()
                upper_whisker = subset[subset <= q3 + 1.5 * iqr].max()

                # If whiskers are empty (single point), clamp to median
                if pd.isna(lower_whisker):
                    lower_whisker = median
                if pd.isna(upper_whisker):
                    upper_whisker = median

                offset = (idx - (num_methods - 1) / 2) * (
                    box_width if len(breakpoints) > 1 else 0.2
                )
                draw_at = bp + offset

                latex.append(
                    f"\\addplot+ ["
                    f"boxplot prepared={{"
                    f"lower whisker={lower_whisker}, upper whisker={upper_whisker}, "
                    f"lower quartile={q1}, upper quartile={q3}, "
                    f"median={median}"
                    f"}}, "
                    f"draw={color}, fill={color}!20, solid, mark=*, "
                    f"boxplot/draw position={draw_at}, "
                    f"boxplot/box extend={box_width * 0.8}"
                    f"] coordinates {{}};"
                )

            # Dummy legend entry (pgfplots boxplots are tricky with legends,
            # we add a dummy plot for the legend)
            latex.append(f"\\addlegendimage{{fill={color}!20, draw={color}}}")
            latex.append(f"\\addlegendentry{{{method}}}")

        latex.append(r"\end{axis}")
        latex.append(r"\end{tikzpicture}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(latex))

    def _plot_instances_solved_over_time_tikz(self, output_file):
        """
        Generates a Cactus plot (Instances Solved vs Time) using TikZ.
        """
        latex = [
            r"\begin{figure}",
            r"\centering",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"    width=\textwidth,",
            r"    height=8cm,",
            r"    xlabel={Time (s)},",
            r"    ylabel={Number of Instances Solved},",
            r"    title={Instances Solved over Time},",
            r"    grid=major,",
            r"    legend pos=south east,",
            r"]",
        ]

        methods = self.results_df["Method"].unique()
        runtimes = []
        colors = ["blue", "red", "green", "orange", "purple"]
        for j, method in enumerate(methods):
            # Get runtimes
            method_data = self.results_df[self.results_df["Method"] == method]
            runtimes = method_data["runtime"].values
            runtimes.sort()

            # Filter valid runtimes
            solved_runtimes = runtimes[np.isfinite(runtimes)]

            # Create coordinates: (0,0), (t1, 1), (t2, 2)...
            coords = ["(0,0)"]
            for i, time in enumerate(solved_runtimes):
                coords.append(f"({time:.4f}, {i + 1})")

            coord_str = " ".join(coords)
            latex.append(
                f"\\addplot[const plot, thick, color={colors[j]}] coordinates {{ {coord_str} }};"
            )
            latex.append(f"\\addlegendentry{{{method}}}")

        figure_caption = (
            f"Number of instances solved over time for different methods."
            f"Total number of instances: {len(runtimes)}"
        )

        latex.append(r"\end{axis}")
        latex.append(r"\end{tikzpicture}")
        latex.append(f"\\caption{{{figure_caption}}}")
        latex.append(r"\end{figure}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(latex))

    def _generate_latex_table_runtime(self, output_file):
        # Filter for relevant test cases
        df = self.results_df[
            self.results_df["test_case"].isin(["Standard", "MPIP"])
        ].copy()

        # Create unique row names
        df["row_name"] = (
            df["osil_file_name"] + " | " + df["nr_of_breakpoints"].astype(str)
        )

        def gmean(x):
            # Calculate geometric mean, ignoring non-positive values and NaNs
            data = x[x > 0].dropna()
            if len(data) == 0:
                return np.nan
            return np.exp(np.log(data).mean())

        # Pivot the table to get instances as rows and test cases as columns
        pivot_df = df.pivot_table(
            index="row_name", columns="test_case", values="runtime", aggfunc=gmean
        )

        # Calculate Speedup
        pivot_df["Speedup"] = pivot_df["Standard"] / pivot_df["MPIP"]
        pivot_df = pivot_df[["Standard", "MPIP", "Speedup"]]
        pivot_df.sort_values(by="Speedup", ascending=False, inplace=True)

        # Escape underscores for LaTeX compatibility
        pivot_df.index = pivot_df.index.str.replace("_", r"\_", regex=False)

        # --- Calculate Overall Geometric Means ---
        overall_std = (
            gmean(pivot_df["Standard"]) if "Standard" in pivot_df.columns else np.nan
        )
        overall_mpip = gmean(pivot_df["MPIP"]) if "MPIP" in pivot_df.columns else np.nan
        overall_speedup = (
            gmean(pivot_df["Speedup"]) if "Speedup" in pivot_df.columns else np.nan
        )

        # --- Generate LaTeX Content ---
        latex_content = "\\begin{table}[htbp]\n"
        latex_content += "  \\centering\n"
        latex_content += (
            "  \\caption{Geometric mean runtime (s) comparison "
            "between Standard and MPIP methods over all seeds.}\n"
        )
        latex_content += "  \\label{tab:runtime_comparison}\n"
        latex_content += "  \\begin{tabular}{lrrr}\n"
        latex_content += "    \\toprule\n"
        latex_content += (
            "    \\textbf{Instance} & \\textbf{Standard (s)} "
            "& \\textbf{MPIP (s)} & \\textbf{Speedup} \\\\\n"
        )
        latex_content += "    \\midrule\n"

        # Add data rows
        for instance, row in pivot_df.iterrows():
            std_val = (
                f"{row['Standard']:.2f}" if not pd.isna(row["Standard"]) else "N/A"
            )
            mpip_val = f"{row['MPIP']:.2f}" if not pd.isna(row["MPIP"]) else "N/A"

            if "Speedup" in pivot_df.columns:
                speedup_val = (
                    f"{row['Speedup']:.2f}" if not pd.isna(row["Speedup"]) else "-"
                )
                latex_content += (
                    f"    {instance} & {std_val} & {mpip_val} & {speedup_val} \\\\\n"
                )
            else:
                latex_content += f"    {instance} & {std_val} & {mpip_val} \\\\\n"

        # Add Overall Summary Row
        latex_content += "    \\midrule\n"

        ov_std_str = f"{overall_std:.2f}" if not pd.isna(overall_std) else "N/A"
        ov_mpip_str = f"{overall_mpip:.2f}" if not pd.isna(overall_mpip) else "N/A"

        if "Speedup" in pivot_df.columns:
            ov_speedup_str = (
                f"{overall_speedup:.2f}" if not pd.isna(overall_speedup) else "-"
            )
            latex_content += (
                f"    \\textbf{{Overall}} & \\textbf{{{ov_std_str}}} "
                f"& \\textbf{{{ov_mpip_str}}} & \\textbf{{{ov_speedup_str}}} \\\\\n"
            )
        else:
            latex_content += (
                f"    \\textbf{{Overall}} & \\textbf{{{ov_std_str}}} "
                f"& \\textbf{{{ov_mpip_str}}} \\\\\n"
            )

        latex_content += "    \\bottomrule\n"
        latex_content += "  \\end{tabular}\n"
        latex_content += "\\end{table}"

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_content)


if __name__ == "__main__":
    from alpaca.utils import inout as ut_io

    ut_io.config_console_logger()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    u_settings = s.UserSettings({})
    visualizer = Visualizer(u_settings)
    os.makedirs(
        u_settings.export_path + lsf.study_mpip_export_folder_performance(),
        exist_ok=True,
    )
    os.makedirs(
        u_settings.export_path + lsf.study_mpip_export_folder_latex_tables(),
        exist_ok=True,
    )
    RESULTS_FILE = "mpip_results_2025-12-20_13-16-54.csv"
    visualizer.create_latex_tables(RESULTS_FILE)
    visualizer.create_performance_plots(RESULTS_FILE)
