# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import pytest
import numpy as np

from alpaca.model_data import variable as var, model_data as mde
from alpaca.expressions import one_dim_expression as ode
from alpaca.breakpoints import breakpoint_adaptive as bad
import alpaca.settings as s


@pytest.mark.parametrize(
    "one_dim_expression",
    [
        ode.SineExpression,
        ode.SquareExpression,
        ode.TangensHExpression,
    ],
)
def test_adaptive_breakpoint_generation(
    one_dim_expression,
):  # pylint: disable=too-many-locals
    """
    Test that the adaptive breakpoint generator refines intervals
    until the linear approximation deviation is within the specified tolerance.
    """
    # Test configuration
    lb = 0.0
    ub = 6.28  # Approximately 2*pi covering interesting regions for Sine/Tanh
    relaxation_tolerance = 0.05
    initial_breakpoints = 2  # Start with endpoints only to force refinement

    # Setup Variable
    test_variable = var.Variable(name="x", lb=lb, ub=ub, var_type="C")
    test_variable.is_discretized = True

    # Setup Settings
    # We specify relaxation_tolerance which drives the adaptive logic
    settings_dict = {
        "number_of_breakpoints": initial_breakpoints,
        "relaxation_tolerance": relaxation_tolerance,
        "breakpoint_generation": 2,
    }
    settings = s.UserSettings(settings_dict)

    # Setup ModelData container
    model_data = mde.ModelData(settings=settings)
    model_data.variables = {"x": test_variable}

    # Add the expression to the model.
    # This registers the nonlinearity with the variable's occurring_in dict.
    expression = model_data.add_one_dim_expression(
        one_dim_expression, "test_expression", test_variable, 0
    )

    # Execute Adaptive Breakpoint Generation
    adaptive_gen = bad.BreakpointAdaptive(test_variable, settings)
    breakpoints = adaptive_gen.breakpoints

    # Assertions

    # 1. Verify Strict Monotonicity
    # Breakpoints must be sorted and unique
    assert np.all(np.diff(breakpoints) > 0), "Breakpoints must be strictly increasing."

    # 2. Verify Bounds Preservation
    assert np.isclose(breakpoints[0], lb), "Lower bound must be preserved."
    assert np.isclose(breakpoints[-1], ub), "Upper bound must be preserved."

    # 3. Verify Refinement Occurred
    # Since we chose nonlinear functions and minimal initial points,
    # the generator must add points to meet the tolerance.
    assert len(breakpoints) > initial_breakpoints, (
        f"Breakpoints should have been refined for nonlinear expression "
        f"{one_dim_expression.__name__}."
    )

    # 4. Verify Deviation Compliance
    # Check each segment to ensure the deviation does not exceed tolerance.
    for i in range(len(breakpoints) - 1):
        bp_start = breakpoints[i]
        bp_end = breakpoints[i + 1]

        # Calculate the linear approximation (secant line) for this segment
        slope, intercept = (
            expression.get_linear_approximation_function_parameters_for_segment(
                bp_start, bp_end
            )
        )

        # Calculate the actual min/max deviation of the function from the line
        min_dev, max_dev = expression.get_min_max_deviation(
            bp_start, bp_end, slope, intercept
        )

        # Define a small numerical buffer for floating point comparisons
        buffer = 1e-7

        # Check deviations
        assert abs(min_dev) <= relaxation_tolerance + buffer, (
            f"Min deviation {min_dev} exceeds tolerance {relaxation_tolerance} "
            f"in interval [{bp_start}, {bp_end}] for {one_dim_expression.__name__}"
        )
        assert abs(max_dev) <= relaxation_tolerance + buffer, (
            f"Max deviation {max_dev} exceeds tolerance {relaxation_tolerance} "
            f"in interval [{bp_start}, {bp_end}] for {one_dim_expression.__name__}"
        )
