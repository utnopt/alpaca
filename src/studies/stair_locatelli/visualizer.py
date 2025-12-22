# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import alpaca.utils.geometry as geo
import alpaca.expressions.bilinear_expression as ble
import alpaca.settings as s
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class Visualizer:
    """Visualizer for Stair Locatelli projected domains and cuts."""

    def __init__(self, settings: s.UserSettings):
        self.settings = settings
        self.plot_dir = settings.export_path + lsf.study_stair_locatelli_export_folder()

    def plot_polygons(
        self,
        bilinear_projected_domains: list[
            tuple[ble.BilinearExpression, list[tuple[float, float]]]
        ],
    ):
        """Plots 2D projection polygons."""
        logger.info(lsf.study_stair_locatelli_info_plot_polygons())
        os.makedirs(self.plot_dir, exist_ok=True)

        for bilinear_expr, vertices in bilinear_projected_domains:
            if not vertices:
                continue

            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()

            # Close polygon
            x_coords, y_coords = zip(*vertices)
            x_coords += (x_coords[0],)
            y_coords += (y_coords[0],)

            ax.plot(
                x_coords,
                y_coords,
                marker="o",
                linestyle="-",
                label=f"Polygon {bilinear_expr.name}",
            )
            ax.set_xlabel(bilinear_expr.variables[0].name)
            ax.set_ylabel(bilinear_expr.variables[1].name)
            ax.set_title("Projected Feasible Domains")
            ax.legend()
            ax.grid(True)

            plt.savefig(
                os.path.join(
                    self.plot_dir,
                    lsf.study_stair_locatelli_polygon_plot_name(bilinear_expr.name),
                )
            )
            plt.close(fig)

    def plot_cuts(
        self,
        bilinear_projected_domains: list[
            tuple[ble.BilinearExpression, list[tuple[float, float]]]
        ],
    ):  # pylint: disable=too-many-locals
        """Plots 3D surfaces and cuts."""
        logger.info(lsf.study_stair_locatelli_info_plot_cuts())
        os.makedirs(self.plot_dir, exist_ok=True)
        view_angles = [(30, -60), (30, 30), (60, -120)]

        for bilinear_expr, vertices in bilinear_projected_domains:
            if len(vertices) < 3:
                continue

            x_var, y_var = bilinear_expr.variables
            x_range = np.linspace(x_var.lb, x_var.ub, 30)
            y_range = np.linspace(y_var.lb, y_var.ub, 30)
            x_mesh, y_mesh = np.meshgrid(x_range, y_range)
            z_mesh = x_mesh * y_mesh

            for i, (elev, azim) in enumerate(view_angles):
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(
                    x_mesh, y_mesh, z_mesh, alpha=0.5, color="r", label="z=xy"
                )
                for v1, v2, v3 in itertools.combinations(vertices, 3):
                    if geo.check_if_vertices_are_collinear(v1, v2, v3):
                        continue

                    valid, _ = geo.calculate_hyperplane_for_vertex_triple(v1, v2, v3)
                    if not valid:
                        continue

                ax.set_xlabel(x_var.name)
                ax.set_ylabel(y_var.name)
                ax.set_zlabel(bilinear_expr.representative_variable.name)
                ax.view_init(elev=elev, azim=azim)
                plt.savefig(
                    os.path.join(
                        self.plot_dir,
                        lsf.study_stair_locatelli_cut_plot_name(bilinear_expr.name, i),
                    )
                )
                plt.close(fig)
