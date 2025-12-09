# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

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
        self.results_df: pd.DataFrame() | None = None

    def plot_blocks(self, mpip_handler: mph.MPIPHandler):
        """Plots block structures for MPIP structures with 2 implying variables."""
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
                self._plot_blocks(mpip, out_path, cut.name, blocks)

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
    def _plot_blocks(
        mpip: mpi.MPIP,
        out_path: str,
        cut_name: str,
        blocks: list[list[tuple[str, str]]],
    ):  # pylint: disable=too-many-locals
        num_blocks = s.StaticSettings.nr_of_blocks_plotted
        _, axes = plt.subplots(1, num_blocks, figsize=(5 * num_blocks, 5))
        if num_blocks == 1:
            axes = [axes]  # Ensure axes is iterable even for 1 plot

        implying_vars = list(mpip.implying_variables.values())
        x_vars_list = implying_vars[0].pwl.pwl_variables_binary
        y_vars_list = implying_vars[1].pwl.pwl_variables_binary

        n_rows = len(x_vars_list)
        n_cols = len(y_vars_list)

        # Map colors to numerical values for imshow
        cmap = mcolors.ListedColormap(["white", "blue"])
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        for block_idx, block in enumerate(blocks):
            ax = axes[block_idx]

            # Initialize matrix for colors (0 for white, 1 for blue)
            color_matrix = np.zeros((n_rows, n_cols))

            # Iterate through all cells to determine color and draw labels
            for i in range(n_rows):
                for j in range(n_cols):
                    x_var = x_vars_list[i]
                    y_var = y_vars_list[j]

                    # Core logic from the user's snippet
                    if (x_var.name, y_var.name) in block:
                        color_matrix[i, j] = 1  # Blue
                        color = "white"  # Text color
                    else:
                        color_matrix[i, j] = 0  # White
                        color = "black"  # Text color

                    label = mpip.relation[(i, j)]

                    # Draw the label text
                    ax.text(
                        j,
                        i,
                        str(label),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                        fontweight="bold",
                    )
            ax.imshow(
                color_matrix,
                cmap=cmap,
                norm=norm,
                origin="upper",
                extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5],
                aspect="auto",
            )
            ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
            ax.tick_params(which="minor", size=0)
            ax.set_title(f"Block {block_idx + 1}")
        plt.suptitle(f"Block Decomposition for Cut: {cut_name}", fontsize=16)
        plt.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.95)
        )  # Adjust layout to make space for subtitle
        plt.savefig(out_path + f"{cut_name}.png")
        plt.close()

    def create_performance_plots(self, results_csv: str):
        """
        Creates performance plots from the CSV results.
        """
        logger.info(lsf.study_mpip_info_plot_performance())
        input_csv = s.StaticSettings.import_path + results_csv
        out_path = (
            self.user_settings.export_path + lsf.study_mpip_export_folder_performance()
        )
        self._load_and_prep_data(input_csv)
        if self.results_df is not None:
            self._plot_performance_profile(
                out_path + lsf.study_mpip_performance_plot_name()
            )
            self._plot_runtime_by_breakpoints(
                out_path + lsf.study_mpip_runtime_scaling_plot_name()
            )

    def _load_and_prep_data(self, filepath):
        """
        Loads the CSV and creates a unique 'Method' identifier for the 4 configurations.
        """
        if not os.path.exists(filepath):
            return

        df = pd.read_csv(filepath)

        # Create a combined column for the legend (e.g., "Standard delta")
        df["Method"] = df["test_case"] + " " + df["pwl_method"]

        # Create a unique identifier for each problem instance
        # An instance is defined by the file name, number of breakpoints, and the random seed
        df["Instance"] = (
            df["osil_file_name"]
            + "_"
            + df["nr_of_breakpoints"].astype(str)
            + "_"
            + df["seed"].astype(str)
        )
        self.results_df = df

    def _plot_performance_profile(self, output_file):
        """
        Generates a Dolan-More performance profile.
        """

        # 1. Pivot the data: Rows = Instances, Columns = Methods, Values = Runtime
        pivot_df = self.results_df.pivot(
            index="Instance", columns="Method", values="runtime"
        )

        # Handle missing values (if a method failed to run on an instance, treat as inf)
        pivot_df = pivot_df.fillna(np.inf)

        # 2. Calculate the best (minimum) time for each instance
        min_times = pivot_df.min(axis=1)

        # 3. Calculate ratios: (Time for Method M) / (Best Time for Instance)
        ratios = pivot_df.div(min_times, axis=0)

        # 4. Plotting
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot a step line for each method
        methods = pivot_df.columns
        # custom_palette = sns.color_palette("husl", len(methods))

        for method in methods:
            method_ratios = ratios[method]

            # Sort ratios
            sorted_ratios = np.sort(method_ratios)

            # Y-axis: Fraction of problems solved within that ratio
            yvals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

            # Add a point at the end to extend the line to the edge of the plot if needed
            plt.step(sorted_ratios, yvals, where="post", label=method, linewidth=2)

        plt.xscale("log")
        plt.xlabel(r"Performance Ratio ($\tau$)", fontsize=12)
        plt.ylabel(r"Fraction of problems solved within factor $\tau$", fontsize=12)
        plt.title("Performance Profile (Runtime)", fontsize=14)
        plt.legend(title="Configuration")
        plt.grid(True, which="both", ls="-", alpha=0.5)

        # Limit x-axis to reasonable bounds (e.g., up to 10x slower) to keep plot readable
        # You can remove this or adjust the limit
        plt.xlim(1, 100)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _plot_runtime_by_breakpoints(self, output_file):
        """
        Generates a boxplot comparing runtimes across different breakpoint counts.
        """

        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")

        # Create boxplot
        sns.boxplot(
            data=self.results_df,
            x="nr_of_breakpoints",
            y="runtime",
            hue="Method",
            palette="Set2",
            showfliers=False,  # Hide extreme outliers to keep the scale readable
        )

        plt.yscale("log")  # Log scale is usually best for solver runtimes
        plt.xlabel("Number of Breakpoints", fontsize=12)
        plt.ylabel("Runtime (s) [Log Scale]", fontsize=12)
        plt.title("Runtime Distribution by Breakpoint Count", fontsize=14)
        plt.legend(title="Configuration", loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()


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
    visualizer.create_performance_plots("mpip_results_2025-12-08_16-08-27.csv")
