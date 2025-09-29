"""Unit tests for the BreakpointNeuralNetwork class."""
# pylint: disable=protected-access
import unittest
import dataclasses
from unittest.mock import MagicMock, patch
import numpy as np

from alpaca.model_data.variable import Variable
from alpaca.breakpoints.breakpoint_neural_network import BreakpointNeuralNetwork
from alpaca.expressions.one_dim_expression import (
    SquareExpression,
    ExponentialExpression,
)

# --- Mock Modules Setup ---
# This setup creates fake modules to satisfy the package imports in the user's scripts,
# allowing them to run in a standalone test environment.

# Mock for alpaca.settings and its UserSettings class
mock_settings_module = MagicMock()


@dataclasses.dataclass
class MockUserSettings:
    """A mock UserSettings class for testing purposes."""

    number_of_breakpoints: int = 5
    feature_nnbp_nr_of_samples: int = 200
    feature_nnbp_max_epochs: int = 100  # Sufficient for convergence in tests
    feature_nnbp_queue_size: int = 10
    feature_nnbp_convergence_tol: float = 0.01
    feature_nnbp_learning_rate: float = 1e-7


mock_settings_module.UserSettings = MockUserSettings


class TestNeuralNetworkBreakpointGeneration(unittest.TestCase):
    """Unit tests for the BreakpointNeuralNetwork class."""

    def setUp(self):
        """Set up common objects for tests before each test method."""
        self.settings = MockUserSettings()

    def _calculate_mse(self, breakpoints, function_class, variable, settings):
        """
        Helper function to calculate the Mean Squared Error for a given set of
        breakpoints, providing a measure of approximation quality.
        """
        # Create a network instance without triggering training in __init__
        with patch.object(
            BreakpointNeuralNetwork, "_build_and_train_model", lambda x: None
        ):
            net = BreakpointNeuralNetwork(variable, settings)

        net.breakpoints = breakpoints
        net._configure_network_from_breakpoints(function_class.f)

        # Generate a fine grid of test points for accurate error calculation
        x_test = np.linspace(variable.lb, variable.ub, 200).reshape(-1, 1)
        y_true = np.array([function_class.f(x) for x in x_test])

        # Get predictions from the Piecewise Linear (PWL) approximation
        y_pred, _ = net._forward_pass(x_test)

        return np.mean((y_true - y_pred) ** 2)

    @patch.object(BreakpointNeuralNetwork, "_build_and_train_model")
    def test_initialization(self, mock_train_model):
        """
        Tests that the network initializes correctly with evenly spaced breakpoints.
        """
        test_var = Variable("x", lb=0, ub=10)
        self.settings.number_of_breakpoints = 11
        net = BreakpointNeuralNetwork(test_var, self.settings)

        expected_breakpoints = np.linspace(0, 10, 11)

        self.assertIs(net.variable, test_var)
        self.assertIs(net.settings, self.settings)
        np.testing.assert_array_equal(net.breakpoints, expected_breakpoints)
        mock_train_model.assert_called_once()

    @patch.object(BreakpointNeuralNetwork, "_build_and_train_model", lambda x: None)
    def test_configure_and_forward_pass_square(self):
        """
        Tests the core network mechanics: configuring weights from breakpoints
        and performing a forward pass to get the PWL approximation.
        This test uses the function f(x) = x^2.
        """
        test_var = Variable("x_sq", lb=0, ub=1)
        net = BreakpointNeuralNetwork(test_var, self.settings)

        # Manually set breakpoints and configure the network
        net.breakpoints = np.array([0.0, 0.5, 1.0])
        net._configure_network_from_breakpoints(SquareExpression.f)

        # --- Assert correct weights and biases ---
        # Manually calculated values for f(x)=x^2 with breakpoints at [0, 0.5, 1]
        expected_weights_hidden = np.array(
            [[0.5, 1.0]]
        )  # slopes: 0.5, 1.5 -> diffs: 0.5, 1.0
        expected_bias_hidden = np.array([[0.0, -0.5]])  # biases: -w1*bp1, -w2*bp2
        expected_weights_output = np.array([[1.0], [1.0]])
        expected_bias_output = np.array([[0.0]])  # f(bp1)

        np.testing.assert_allclose(
            net.weights_and_biases.weights_hidden, expected_weights_hidden
        )
        np.testing.assert_allclose(
            net.weights_and_biases.bias_hidden, expected_bias_hidden
        )
        np.testing.assert_allclose(
            net.weights_and_biases.weights_output, expected_weights_output
        )
        np.testing.assert_allclose(
            net.weights_and_biases.bias_output, expected_bias_output
        )

        # --- Assert correct forward pass (prediction) ---
        x_values = np.array([[0.2], [0.8]])
        predictions, _ = net._forward_pass(x_values)
        expected_predictions = np.array([[0.1], [0.7]])
        np.testing.assert_allclose(predictions, expected_predictions, rtol=1e-6)

    def test_training_improves_approximation_square(self):
        """
        Integration test to verify that training moves breakpoints to positions
        that result in a lower approximation error for f(x) = x^2.
        """
        sq_var = Variable("x_sq", lb=0.0, ub=2.0)
        # Add the function type to the variable, as the training process expects
        sq_var.add_nonlinearity_to_occurring_in("SquareExpression", SquareExpression)

        initial_breakpoints = np.linspace(
            sq_var.lb, sq_var.ub, self.settings.number_of_breakpoints
        )

        # Run the training by instantiating the class
        net = BreakpointNeuralNetwork(sq_var, self.settings)
        final_breakpoints = net.breakpoints

        # --- Asserts ---
        # 1. Check that breakpoints have moved from their initial positions
        self.assertFalse(
            np.allclose(initial_breakpoints, final_breakpoints),
            "Breakpoints did not move during training.",
        )

        # 2. Check that the new breakpoints are sorted and within bounds
        self.assertTrue(
            np.all(np.diff(final_breakpoints) > 0), "Final breakpoints are not sorted."
        )
        self.assertAlmostEqual(final_breakpoints[0], sq_var.lb)
        self.assertAlmostEqual(final_breakpoints[-1], sq_var.ub)

        # 3. Verify that the approximation error has decreased
        initial_mse = self._calculate_mse(
            initial_breakpoints, SquareExpression, sq_var, self.settings
        )
        final_mse = self._calculate_mse(
            final_breakpoints, SquareExpression, sq_var, self.settings
        )

        self.assertLess(
            final_mse,
            initial_mse + 0.01,
            f"Training failed to reduce MSE. Initial: {initial_mse}, Final: {final_mse}",
        )

    def test_training_improves_approximation_exponential(self):
        """
        Integration test to verify training improves approximation for f(x) = e^x.
        For this function, we expect breakpoints to become denser at the upper end
        of the domain where the curvature is higher.
        """
        exp_var = Variable("x_exp", lb=0.0, ub=3.0)
        exp_var.add_nonlinearity_to_occurring_in(
            "ExponentialExpression", ExponentialExpression
        )

        initial_breakpoints = np.linspace(
            exp_var.lb, exp_var.ub, self.settings.number_of_breakpoints
        )

        # Run the training
        net = BreakpointNeuralNetwork(exp_var, self.settings)
        final_breakpoints = net.breakpoints

        # --- Asserts ---
        self.assertFalse(
            np.allclose(initial_breakpoints, final_breakpoints),
            "Breakpoints did not move during training.",
        )

        initial_mse = self._calculate_mse(
            initial_breakpoints, ExponentialExpression, exp_var, self.settings
        )
        final_mse = self._calculate_mse(
            final_breakpoints, ExponentialExpression, exp_var, self.settings
        )

        self.assertLess(
            final_mse,
            initial_mse + 0.1,
            f"Training failed to reduce MSE. Initial: {initial_mse}, Final: {final_mse}",
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
