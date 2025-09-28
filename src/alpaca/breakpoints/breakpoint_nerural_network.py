# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses
import random
import copy
import io
import imageio
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
            learning_rate=0.007,
            epochs=300,
    ):
        self.number_of_breakpoints = nr_of_breakpoints
        self.variable = variable
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.minimum_distance_from_zero = 0.0

        # Network architecture
        # One input neuron, one output neuron, and a hidden layer
        self.weights_and_biases = WeightsAndBiases(
            weights_hidden=None, bias_hidden=None, weights_output=None, bias_output=None
        )
        self._build_and_train_model()

    @staticmethod
    def _relu(x, negative_slope=False):
        """ReLU activation function."""
        if negative_slope:
            return np.minimum(0, x)
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x, negative_slope=False):
        """Derivative of the ReLU function."""
        if negative_slope:
            return np.where(x < 0, 1, 0)
        return np.where(x > 0, 1, 0)

    def _forward_pass(self, x, weights_and_biases=None):
        """
        Performs a forward pass through the network.
        Accepts an optional weights_and_biases object to plot historical states.
        """
        wb = weights_and_biases if weights_and_biases is not None else self.weights_and_biases

        # Hidden layer
        hidden_layer_input = (
                np.dot(x, wb.weights_hidden) + wb.bias_hidden
        )

        # For columns where the corresponding hidden weight is negative, use the
        # negative slope ReLU. Otherwise, use the standard ReLU.
        hidden_layer_output = np.where(
            wb.weights_hidden < 0,
            self._relu(hidden_layer_input, negative_slope=True),
            self._relu(hidden_layer_input, negative_slope=False),
        )

        # Output layer
        output_layer_input = (
                np.dot(hidden_layer_output, wb.weights_output)
                + wb.bias_output
        )

        return output_layer_input, hidden_layer_output

    def _backward_pass(self, x, y, y_pred, hidden_layer_output):
        """Performs a backward pass (backpropagation) to update weights."""
        error = y - y_pred

        # Gradients for output layer
        d_bias_output = np.sum(error, axis=0, keepdims=True)

        # Gradients for hidden layer
        # Apply the ReLU derivative conditionally based on the hidden weights
        relu_derivative_output = np.where(
            self.weights_and_biases.weights_hidden < 0,
            self._relu_derivative(hidden_layer_output, negative_slope=True),
            self._relu_derivative(hidden_layer_output, negative_slope=False),
        )
        error_hidden = np.dot(
            error, self.weights_and_biases.weights_output.T
        ) * relu_derivative_output
        d_weights_hidden = np.dot(x.T, error_hidden)
        d_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True)
        # Update weights and biases
        learning_rate = self.learning_rate / abs(sum(d_bias_hidden[0]))
        self.weights_and_biases.bias_output += learning_rate * d_bias_output
        self.weights_and_biases.weights_hidden += learning_rate * d_weights_hidden
        self.push_weights_from_zero(self.weights_and_biases.weights_hidden)
        self.weights_and_biases.bias_hidden += learning_rate * d_bias_hidden

    def push_weights_from_zero(self, weights):
        """
        Pushes weights away from zero if they are within a certain threshold.
        """
        close_to_zero_mask = np.abs(weights) < self.minimum_distance_from_zero
        signs = np.sign(weights)
        signs[signs == 0] = 1
        new_values = signs * self.minimum_distance_from_zero
        np.copyto(weights, new_values, where=close_to_zero_mask)

    def _plot_frame(self, epoch, x_train, y_train, weights_and_biases):
        """Plots a single frame for the animation."""
        fig, ax = plt.subplots()

        # Plot training data
        ax.scatter(x_train, y_train, s=10, alpha=0.5, label="Training Data")

        # Create a range of x values for plotting the learned function
        x_range = np.linspace(self.variable.lb, self.variable.ub, 200).reshape(-1, 1)
        y_pred, _ = self._forward_pass(x_range, weights_and_biases=weights_and_biases)

        ax.plot(
            x_range, y_pred, color="red", linewidth=2, label="NN Approximation (PWL)"
        )

        # Plot first original function
        for name, func in self.variable.occurring_in.items():
            y_func = np.vectorize(func)(x_range)
            ax.plot(x_range, y_func, linestyle="--", label=f"Original: {name}")
            break

        ax.set_title(f"Epoch {epoch + 1}/{self.epochs}")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(np.min(y_train) - 1, np.max(y_train) + 1)

        return fig

    def _save_training_animation(self, history, x_train, y_train):
        """Creates and saves a GIF from the training history."""
        logger.info("Creating training animation GIF...")

        # Limit frames for very long trainings to keep GIF size reasonable
        num_frames = min(self.epochs, 200)
        frame_indices = np.linspace(0, self.epochs - 1, num_frames, dtype=int)

        with imageio.get_writer("training_animation.gif", fps=10) as writer:
            for i, epoch_idx in enumerate(frame_indices):
                wb = history[epoch_idx]
                fig = self._plot_frame(epoch_idx, x_train, y_train, wb)

                # Save plot to an in-memory buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                image = imageio.imread(buf)
                writer.append_data(image)
                plt.close(fig)  # Close the figure to free memory

                logger.info(f"Generated frame {i + 1}/{num_frames}")

        logger.info("Training animation saved as training_animation.gif")

    def _initialize_weights_and_biases(self):
        """Initializes weights and biases for the neural network."""
        initial_function = list(self.variable.occurring_in.values())[0]
        initial_breakpoints = nonlinear_space(
            self.variable.lb, self.variable.ub, num=self.number_of_breakpoints
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
        weights_output = [1 for _ in initial_slopes]
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

        history = []
        logger.info(self.weights_and_biases)

        # Training loop
        for epoch in range(self.epochs):
            y_pred, hidden_layer_output = self._forward_pass(x_train)

            # Store a deep copy of the weights and biases for animation creation
            history.append(copy.deepcopy(self.weights_and_biases))

            total_loss = np.mean((y_train - y_pred) ** 2)

            self._backward_pass(x_train, y_train, y_pred, hidden_layer_output)
            logger.info(
                "Epoch %d/%d, Average Loss: %.4f",
                epoch + 1,
                self.epochs,
                total_loss,
            )

        # After training, create and save the animation if requested
        self._save_training_animation(history, x_train, y_train)

    def generate_breakpoints(self) -> list[float]:
        """
        Generates breakpoints using the trained neural network.
        """
        breakpoints = set()
        for i in range(self.number_of_breakpoints - 1):
            weight = self.weights_and_biases.weights_hidden[0, i]
            bias = self.weights_and_biases.bias_hidden[0, i]
            if weight != 0:
                breakpoint_val = -bias / weight
                if self.variable.lb <= breakpoint_val <= self.variable.ub:
                    breakpoints.add(breakpoint_val)

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


def nonlinear_space(start, stop, num=50, method='quadratic', power=2):
    """
    Similar to np.linspace, but distributes points unevenly over [start, stop].
    """
    t = np.linspace(0, 1, num)

    if method == 'quadratic':
        t = t ** 2
    elif method == 'sqrt':
        t = np.sqrt(t)
    elif method == 'power':
        t = t ** power
    elif method == 'log':
        if start <= 0 or stop <= 0:
            raise ValueError("Log spacing requires start, stop > 0")
        return np.logspace(np.log10(start), np.log10(stop), num)
    else:
        raise ValueError(f"Unknown method: {method}")

    return start + (stop - start) * t
