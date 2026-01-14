# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
# pylint: disable=redefined-outer-name, protected-access, no-member

import pytest
import numpy as np

import alpaca.breakpoints.breakpoint_neural_network as bnn


class DummySettings:
    """
    Mock settings object simulating alpaca.settings.UserSettings.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self):
        self.number_of_breakpoints = 5
        self.feature_nnbp_learning_rate = 0.1
        self.feature_nnbp_nr_of_samples = 20
        self.feature_nnbp_queue_size = 5
        self.feature_nnbp_time_limit = 0.5
        self.feature_nnbp_convergence_tol = 1e-4


class DummyExpression:
    """
    Mock expression object simulating alpaca.expressions.one_dim_expression.
    Default behavior is f(x) = x.
    """

    # pylint: disable=too-few-public-methods
    def f(self, x):
        """Returns the function value (identity)."""
        return x

    def f_derivative(self, x):  # pylint: disable=unused-argument
        """Returns the derivative value (constant 1)."""
        return 1.0


class DummyVariable:
    """
    Mock variable object simulating alpaca.model_data.variable.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self, expression):
        self.lb = 0.0
        self.ub = 4.0
        self.occurring_in = {"expr1": expression}


class TestableBNN(bnn.BreakpointNeuralNetwork):
    """
    Subclass that disables the automatic training on initialization
    to allow for unit testing of individual components without
    running the time-consuming training loop.
    """

    def _build_and_train_model(self):
        """Override to prevent training during initialization."""


# --- Fixtures ---


@pytest.fixture
def mock_settings():
    """Fixture for DummySettings."""
    return DummySettings()


@pytest.fixture
def mock_expression():
    """Fixture for DummyExpression."""
    return DummyExpression()


@pytest.fixture
def mock_variable(mock_expression):
    """Fixture for DummyVariable."""
    return DummyVariable(mock_expression)


@pytest.fixture
def net(mock_variable, mock_settings):
    """
    Returns an instance of BreakpointNeuralNetwork with training bypassed.
    Note: __init__ sets breakpoints BEFORE calling _build_and_train_model,
    so breakpoints are already initialized, but weights are None.
    """
    network = TestableBNN(mock_variable, mock_settings)
    return network


# --- Tests ---


class TestActivationFunctions:
    """Tests for static activation methods."""

    def test_relu_standard(self):
        """Test standard ReLU: max(0, x)."""
        arr = np.array([-1, 0, 1])
        result = bnn.BreakpointNeuralNetwork._relu(arr, negative_slope=False)
        np.testing.assert_array_equal(result, np.array([0, 0, 1]))

    def test_relu_negative_slope(self):
        """Test negative slope ReLU: min(0, x)."""
        arr = np.array([-1, 0, 1])
        result = bnn.BreakpointNeuralNetwork._relu(arr, negative_slope=True)
        np.testing.assert_array_equal(result, np.array([-1, 0, 0]))

    def test_relu_derivative_standard(self):
        """Test derivative of standard ReLU."""
        arr = np.array([-5, 5])
        result = bnn.BreakpointNeuralNetwork._relu_derivative(arr, negative_slope=False)
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_relu_derivative_negative_slope(self):
        """Test derivative of negative slope ReLU."""
        arr = np.array([-5, 5])
        result = bnn.BreakpointNeuralNetwork._relu_derivative(arr, negative_slope=True)
        np.testing.assert_array_equal(result, np.array([1, 0]))


class TestNetworkLogic:
    """Tests for network initialization, forward pass, and backward pass logic."""

    def test_initialization_breakpoints(self, mock_variable, mock_settings):
        """Test that breakpoints are initialized linearly between lb and ub."""
        # TestableBNN skips training, but runs the rest of __init__
        network = TestableBNN(mock_variable, mock_settings)
        expected = np.linspace(0.0, 4.0, 5)
        np.testing.assert_array_almost_equal(network.breakpoints, expected)

    def test_configure_network_linear_function(self, net):
        """
        Test that configuring the network for f(x) = 2x results in correct weights.
        """

        # Setup f(x) = 2x
        def linear_func(x):
            return 2.0 * x

        net._configure_network_from_breakpoints(linear_func)

        weights_biases = net.weights_and_biases

        # Check output weights (should be all 1s as per implementation)
        assert np.all(weights_biases.weights_output == 1.0)

        # Check hidden weights
        # w[0] = slope[0] - 0 = 2
        first_weight = weights_biases.weights_hidden[0, 0]
        other_weights = weights_biases.weights_hidden[0, 1:]

        assert first_weight == pytest.approx(2.0)
        np.testing.assert_allclose(other_weights, 0.0, atol=1e-10)

    def test_forward_pass_manual_weights(self, net):
        """Test forward pass with manually injected weights."""
        x_in = np.array([[2.0]])

        # Inject weights manually into the data class
        # Network: y = ReLU(x * 3 + 1) * 0.5 + 2
        net.weights_and_biases.weights_hidden = np.array([[3.0]])
        net.weights_and_biases.bias_hidden = np.array([[1.0]])
        net.weights_and_biases.weights_output = np.array([[0.5]])
        net.weights_and_biases.bias_output = np.array([[2.0]])

        # Hidden input = 2 * 3 + 1 = 7 -> ReLU -> 7
        # Output = 7 * 0.5 + 2 = 5.5

        output, hidden_out = net._forward_pass(x_in)

        assert output[0, 0] == pytest.approx(5.5)
        assert hidden_out[0, 0] == pytest.approx(7.0)

    def test_forward_pass_negative_weights_logic(self, net):
        """Test that negative hidden weights trigger the negative slope ReLU."""
        x_in = np.array([[2.0]])

        # weight is negative -> should use negative_slope ReLU logic
        net.weights_and_biases.weights_hidden = np.array([[-3.0]])
        net.weights_and_biases.bias_hidden = np.array([[1.0]])
        # hidden input = 2 * -3 + 1 = -5

        # Negative slope ReLU: min(0, x) -> -5

        net.weights_and_biases.weights_output = np.array([[1.0]])
        net.weights_and_biases.bias_output = np.array([[0.0]])

        _, hidden_out = net._forward_pass(x_in)

        assert hidden_out[0, 0] == pytest.approx(-5.0)

    def test_backward_pass_updates_breakpoints(self, net, mock_expression):
        """Test that backward pass actually changes the breakpoints."""
        x_train = np.array([[1.0], [2.0], [3.0]])
        net._configure_network_from_breakpoints(mock_expression.f)

        error = np.array([[0.5], [-0.5], [0.5]])

        _, hidden_layer_output = net._forward_pass(x_train)

        old_breakpoints = net.breakpoints.copy()

        net._backward_pass(x_train, error, hidden_layer_output, mock_expression)

        # Internal breakpoints should change
        internal_changed = not np.allclose(net.breakpoints[1:-1], old_breakpoints[1:-1])

        assert internal_changed
        assert net.breakpoints[0] == old_breakpoints[0]
        assert net.breakpoints[-1] == old_breakpoints[-1]

    def test_training_loop_execution(self, mock_variable, mock_settings):
        """Test that the full training loop runs without crashing."""
        # Here we use the REAL class, not the TestableBNN subclass,
        # to ensure the actual loop runs.

        mock_settings.feature_nnbp_time_limit = 0.1
        mock_settings.feature_nnbp_nr_of_samples = 5

        network = bnn.BreakpointNeuralNetwork(mock_variable, mock_settings)

        assert network.breakpoints is not None
        assert len(network.breakpoints) == mock_settings.number_of_breakpoints
        assert np.all(np.diff(network.breakpoints) > 0)
