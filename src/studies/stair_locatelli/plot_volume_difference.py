# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import numpy as np
import matplotlib.pyplot as plt

from alpaca import Alpaca
import alpaca.locatelli.stair_locatelli as stl
import alpaca.locatelli.locatelli_cut_generator as lcg
import alpaca.expressions.bilinear_expression as ble
import alpaca.model_data.variable as var


def analyze_volume_difference(
    resolution=10, x_ub=1.0, y_ub=1.0, direction="ne"
):  # pylint: disable=too-many-locals
    """
    Analyzes the volume difference for x, y in [0, 1].
    """
    # Create grid
    x_vals = np.linspace(0, x_ub, resolution)
    y_vals = np.linspace(0, y_ub, resolution)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    z_diff_mesh = np.zeros_like(x_mesh)

    print(f"Calculating volume differences over a {resolution}x{resolution} grid...")

    alp = Alpaca()
    x = var.Variable("x", lb=0.0, ub=x_ub)
    y = var.Variable("y", lb=0.0, ub=y_ub)
    z = var.Variable("z", lb=0.0, ub=x_ub * y_ub)
    stair_locatelli = stl.StairLocatelli(alp.model_data)
    cut_generator = lcg.LocatelliCutGenerator(alp.model_data)
    for i in range(resolution):
        for j in range(resolution):
            bilinear_expression_polygon = ble.BilinearExpression(
                "z_xy", alp.model_data, [x, y], level=0, representative_variable=z
            )
            bilinear_expression_polytope = ble.BilinearExpression(
                "z_xy_p", alp.model_data, [x, y], level=0, representative_variable=z
            )
            vertices_polygon = generate_vertices_polygon(
                x_mesh[i, j], y_mesh[i, j], x_ub=x_ub, y_ub=y_ub, direction=direction
            )
            vertices_polytope = generate_vertices_polytope(
                x_mesh[i, j], y_mesh[i, j], x_ub=x_ub, y_ub=y_ub, direction=direction
            )
            cut_generator.generate_cuts(bilinear_expression_polygon, vertices_polygon)
            cut_generator.generate_cuts(bilinear_expression_polytope, vertices_polytope)
            z_diff = stair_locatelli.calculate_3d_volume_polytope_over_2d_polygon(
                bilinear_expression_polytope, vertices_polygon
            ) - stair_locatelli.calculate_3d_volume_polygon_over_domain(
                bilinear_expression_polygon, vertices_polygon
            )
            z_diff_mesh[i, j] = z_diff
            alp.model_data.constraints = {}
    return x_mesh, y_mesh, z_diff_mesh


def generate_vertices_polygon(
    x_val, y_val, x_lb=0.0, x_ub=1.0, y_lb=0.0, y_ub=1.0, direction="ne"
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Generates the vertices of the polygon for given x and y values.
    """
    if direction == "ne":
        return [
            (x_lb, y_lb),
            (x_lb, y_ub),
            (x_val, y_ub),
            (x_val, y_val),
            (x_ub, y_val),
            (x_ub, y_lb),
        ]
    if direction == "nw":
        return [
            (x_lb, y_lb),
            (x_lb, y_val),
            (x_val, y_val),
            (x_val, y_ub),
            (x_ub, y_ub),
            (x_ub, y_lb),
        ]
    if direction == "se":
        return [
            (x_lb, y_lb),
            (x_lb, y_ub),
            (x_ub, y_ub),
            (x_ub, y_val),
            (x_val, y_val),
            (x_val, y_lb),
        ]
    return [
        (x_lb, y_val),
        (x_lb, y_ub),
        (x_ub, y_ub),
        (x_ub, y_lb),
        (x_val, y_lb),
        (x_val, y_val),
    ]


def generate_vertices_polytope(
    x_val, y_val, x_lb=0.0, x_ub=1.0, y_lb=0.0, y_ub=1.0, direction="ne"
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Generates the vertices of the polytope for given x and y values.
    """
    if direction == "ne":
        return [
            (x_lb, y_lb),
            (x_lb, y_ub),
            (x_val, y_ub),
            (x_ub, y_val),
            (x_ub, y_lb),
        ]
    if direction == "nw":
        return [
            (x_lb, y_lb),
            (x_lb, y_val),
            (x_val, y_ub),
            (x_ub, y_ub),
            (x_ub, y_lb),
        ]
    if direction == "se":
        return [
            (x_lb, y_lb),
            (x_lb, y_ub),
            (x_ub, y_ub),
            (x_ub, y_val),
            (x_val, y_lb),
        ]
    return [
        (x_lb, y_val),
        (x_lb, y_ub),
        (x_ub, y_ub),
        (x_ub, y_lb),
        (x_val, y_lb),
    ]


def plot_results(x_mesh, y_mesh, z_mesh):
    """
    Plots the heatmap of volume differences.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the contour plot
    contour = ax.contourf(x_mesh, y_mesh, z_mesh, levels=50, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Volume Difference (Vol P - Vol Q)", rotation=270, labelpad=20)

    # Labels and Title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run analysis
    X, Y, Z = analyze_volume_difference(
        resolution=10, x_ub=2.0, y_ub=2.0, direction="nw"
    )
    plot_results(X, Y, Z)
