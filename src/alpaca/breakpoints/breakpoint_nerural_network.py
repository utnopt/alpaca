# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses
import random
import matplotlib.pyplot as plt
import numpy as np

from alpaca.utils.logger import logger
import alpaca.model_data.variable as var


class BreakpointNeuralNetwork:
    """Generates breakpoints using a neural network approach."""

    def __init__(
        self,
        variable: var.Variable,
        nr_of_breakpoints: int,
        learning_rate=0.0001,
        epochs=20,
    ):
        self.number_of_breakpoints = nr_of_breakpoints
        self.variable = variable
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Network architecture
        # One input neuron, one output neuron, and a hidden layer
        self.weights_and_biases = WeightsAndBiases(
            weights_hidden=None, bias_hidden=None, weights_output=None, bias_output=None
        )
        self._build_and_train_model()

    @staticmethod
    def _relu(x):
        """ReLU activation function."""
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x):
        """Derivative of the ReLU function."""
        return np.where(x > 0, 1, 0)

    def _forward_pass(self, x):
        """Performs a forward pass through the network."""
        # Hidden layer
        hidden_layer_input = (
            np.dot(x, self.weights_and_biases.weights_hidden)
            + self.weights_and_biases.bias_hidden
        )
        hidden_layer_output = self._relu(hidden_layer_input)

        # Output layer
        output_layer_input = (
            np.dot(hidden_layer_output, self.weights_and_biases.weights_output)
            + self.weights_and_biases.bias_output
        )

        return output_layer_input, hidden_layer_output

    def _backward_pass(self, x, y, y_pred, hidden_layer_output):
        """Performs a backward pass (backpropagation) to update weights."""
        error = y - y_pred

        # Gradients for output layer
        d_weights_output = np.dot(hidden_layer_output.T, error)
        d_bias_output = np.sum(error, axis=0, keepdims=True)

        # Gradients for hidden layer
        error_hidden = np.dot(
            error, self.weights_and_biases.weights_output.T
        ) * self._relu_derivative(hidden_layer_output)
        d_weights_hidden = np.dot(x.T, error_hidden)
        d_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_and_biases.weights_output += self.learning_rate * d_weights_output
        self.weights_and_biases.bias_output += self.learning_rate * d_bias_output
        self.weights_and_biases.weights_hidden += self.learning_rate * d_weights_hidden
        self.weights_and_biases.bias_hidden += self.learning_rate * d_bias_hidden

    def _plot_pwl_function(self, epoch, x_train, y_train):
        """Plots the current PWL function approximated by the network."""
        plt.clf()

        # Plot training data
        plt.scatter(x_train, y_train, s=10, alpha=0.5, label="Training Data")

        # Create a range of x values for plotting the learned function
        x_range = np.linspace(self.variable.lb, self.variable.ub, 200).reshape(-1, 1)
        y_pred, _ = self._forward_pass(x_range)

        plt.plot(
            x_range, y_pred, color="red", linewidth=2, label="NN Approximation (PWL)"
        )

        # Plot original functions
        for name, func in self.variable.occurring_in.items():
            y_func = np.vectorize(func)(x_range)
            plt.plot(x_range, y_func, linestyle="--", label=f"Original: {name}")

        plt.title(f"Epoch {epoch + 1}/{self.epochs}")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()
        plt.grid(True)
        plt.ylim(np.min(y_train) - 1, np.max(y_train) + 1)  # Adjust y-axis limits
        plt.pause(0.01)

    def _initialize_weights_and_biases(self):
        """Initializes weights and biases for the neural network.
        This is the piecewise linear approximation for uniformly distributed breakpoints."""
        initial_function = list(self.variable.occurring_in.values())[0]
        initial_breakpoints = np.linspace(
            self.variable.lb, self.variable.ub, self.number_of_breakpoints
        )
        initial_slopes = [
            (
                initial_function(initial_breakpoints[i + 1])
                - initial_function(initial_breakpoints[i])
            )
            / (initial_breakpoints[i + 1] - initial_breakpoints[i])
            for i in range(len(initial_breakpoints) - 1)
        ]
        weights_output = [np.sign(initial_slopes[0])]
        for pre_slope, slope in zip(initial_slopes[:-1], initial_slopes[1:]):
            weights_output.append(
                np.sign(slope - pre_slope) if slope - pre_slope != 0 else 1
            )
        bias_output = initial_function(initial_breakpoints[0])
        weights_hidden = []
        bias_hidden = []
        previous_slope = 0
        for i, bp in enumerate(initial_breakpoints[:-1]):
            weights_hidden.append(
                (initial_slopes[i] - previous_slope) / weights_output[i]
            )

            bias_hidden.append(-weights_hidden[i] * bp)
            previous_slope = initial_slopes[i]

        self.weights_and_biases.bias_output = np.array([[bias_output]])
        self.weights_and_biases.weights_hidden = np.array([weights_hidden])
        self.weights_and_biases.bias_hidden = np.array([bias_hidden])
        self.weights_and_biases.weights_output = np.array([weights_output]).T

    def _build_and_train_model(self):
        """Builds and trains the neural network."""
        self._initialize_weights_and_biases()
        functions = [list(self.variable.occurring_in.values())[0]]

        num_samples = 1000
        x_train = np.random.uniform(
            self.variable.lb, self.variable.ub, (num_samples, 1)
        )
        y_train_list = []
        for x_val in x_train:
            random_func = random.choice(functions)
            y_train_list.append(random_func(x_val[0]))
        y_train = np.array(y_train_list).reshape(-1, 1)

        plt.ion()  # Turn on interactive mode for plotting
        logger.info(self.weights_and_biases)
        # Training loop
        for epoch in range(self.epochs):
            # Forward pass on the entire batch
            y_pred, hidden_layer_output = self._forward_pass(x_train)
            # Calculate loss for logging
            self._plot_pwl_function(epoch, x_train, y_train)
            total_loss = np.mean((y_train - y_pred) ** 2)

            # Backward pass and weight update on the entire batch
            self._backward_pass(x_train, y_train, y_pred, hidden_layer_output)
            logger.info(
                "Epoch %d/%d, Average Loss: %.4f",
                epoch + 1,
                self.epochs,
                total_loss,
            )

        plt.ioff()
        plt.show()

    def generate_breakpoints(self) -> list[float]:
        """
        Generates breakpoints using the trained neural network.
        The breakpoints of a ReLU network are where the input to a neuron is zero.
        """
        breakpoints = set()
        for i in range(self.number_of_breakpoints - 1):
            weight = self.weights_and_biases.weights_hidden[0, i]
            bias = self.weights_and_biases.bias_hidden[0, i]
            if weight != 0:
                breakpoint_val = -bias / weight
                if self.variable.lb <= breakpoint_val <= self.variable.ub:
                    breakpoints.add(breakpoint_val)

        # Add the bounds of the variable as breakpoints
        breakpoints.add(self.variable.lb)
        breakpoints.add(self.variable.ub)

        sorted_breakpoints = sorted(list(breakpoints))
        return sorted_breakpoints


@dataclasses.dataclass
class WeightsAndBiases:
    """Holds weights and biases for the neural network."""

    weights_hidden: np.ndarray | None
    bias_hidden: np.ndarray | None
    weights_output: np.ndarray | None
    bias_output: np.ndarray | None
