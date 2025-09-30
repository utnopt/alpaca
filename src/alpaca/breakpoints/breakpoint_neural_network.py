# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import dataclasses
from typing import Callable, Optional
import numpy as np

import alpaca.model_data.variable as var
import alpaca.settings as s
import alpaca.expressions.one_dim_expression as ode


@dataclasses.dataclass
class WeightsAndBiases:
    """Holds weights and biases for the neural network."""

    weights_hidden: Optional[np.ndarray]
    bias_hidden: Optional[np.ndarray]
    weights_output: Optional[np.ndarray]
    bias_output: Optional[np.ndarray]


class BreakpointNeuralNetwork:
    """Generates breakpoints using a neural network approach."""

    variable: var.Variable
    settings: s.UserSettings
    weights_and_biases: WeightsAndBiases
    breakpoints: np.ndarray

    def __init__(self, variable: var.Variable, settings: s.UserSettings) -> None:
        self.variable = variable
        self.weights_and_biases = WeightsAndBiases(
            weights_hidden=None, bias_hidden=None, weights_output=None, bias_output=None
        )
        self.breakpoints = np.linspace(
            self.variable.lb, self.variable.ub, num=settings.number_of_breakpoints
        )
        self.settings = settings
        self._build_and_train_model()

    @staticmethod
    def _relu(x: np.ndarray, negative_slope: bool = False) -> np.ndarray:
        """ReLU activation function."""
        if negative_slope:
            return np.minimum(0, x)
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray, negative_slope: bool = False) -> np.ndarray:
        """Derivative of the ReLU function."""
        if negative_slope:
            return np.where(x < 0, 1, 0)
        return np.where(x > 0, 1, 0)

    def _forward_pass(
        self, x: np.ndarray, weights_and_biases: Optional[WeightsAndBiases] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass through the network.
        Accepts an optional weights_and_biases object to plot historical states.
        """
        wb = (
            weights_and_biases
            if weights_and_biases is not None
            else self.weights_and_biases
        )
        hidden_layer_input = np.dot(x, wb.weights_hidden) + wb.bias_hidden
        # For columns where the corresponding hidden weight is negative, use the
        # negative slope ReLU. Otherwise, use the standard ReLU.
        hidden_layer_output = np.where(
            wb.weights_hidden < 0,
            self._relu(hidden_layer_input, negative_slope=True),
            self._relu(hidden_layer_input, negative_slope=False),
        )
        output_layer_input = (
            np.dot(hidden_layer_output, wb.weights_output) + wb.bias_output
        )
        return output_layer_input, hidden_layer_output

    # pylint: disable=too-many-locals
    def _backward_pass(
        self,
        x: np.ndarray,
        error: np.ndarray,
        hidden_layer_output: np.ndarray,
        function_class: "ode.OneDimExpression",
    ) -> None:
        """
        Performs a backward pass (backpropagation) to update breakpoints.
        This version is corrected and refactored for clarity and correctness, using a loop.
        """
        # Gradients for hidden layer (assuming this part is correct)
        relu_derivative_output = np.where(
            self.weights_and_biases.weights_hidden < 0,
            self._relu_derivative(hidden_layer_output, negative_slope=True),
            self._relu_derivative(hidden_layer_output, negative_slope=False),
        )
        error_hidden: np.ndarray = (
            error * relu_derivative_output
        )  # Shape: (n_samples, n_hidden_units)

        # --- Step 2: Pre-compute values for efficiency ---
        num_breakpoints: int = len(self.breakpoints)
        f_b: np.ndarray = np.array([function_class.f(bp) for bp in self.breakpoints])
        fd_b: np.ndarray = np.array(
            [function_class.f_derivative(bp) for bp in self.breakpoints]
        )

        # Differences between adjacent breakpoint values
        delta_b: np.ndarray = np.diff(self.breakpoints)
        delta_f: np.ndarray = np.diff(f_b)

        # Initialize gradients for breakpoints as zeros.
        # Using np.zeros_like is safer and handles shape correctly.
        d_breakpoints: np.ndarray = np.zeros_like(self.breakpoints, dtype=np.float64)

        # --- Step 3: Loop over INTERNAL breakpoints to calculate gradients ---
        # The loop should go from the first internal breakpoint to the last one.
        denominator_right_sq: np.ndarray = np.array(0.0)
        for i in range(1, num_breakpoints - 1):
            # --- Gradient contribution from the left segment (i-1, i) ---
            denominator_left_sq = delta_b[i - 1] ** 2
            grad_signal_left = np.sum(
                error_hidden[:, i - 1] * (self.breakpoints[i - 1] - x)
            )
            product_term = (
                -delta_b[i - 1] * fd_b[i] + delta_f[i - 1]
            ) / denominator_left_sq
            d_breakpoints[i] += grad_signal_left * product_term
            # --- Gradient contribution from the right segment (i, i+1) ---
            denominator_right_sq = delta_b[i] ** 2
            if i < num_breakpoints - 2:  # Ensure we don't go out of bounds
                grad_signal_right = np.sum(
                    error_hidden[:, i + 1] * (x - self.breakpoints[i + 1])
                )
                product_term = (
                    delta_b[i] * fd_b[i] - delta_f[i]
                ) / denominator_right_sq
                d_breakpoints[i] += grad_signal_right * product_term
            # --- Gradient contribution from the current segment (i) ---
            grad_signal_current = np.sum(error_hidden[:, i] * (x - self.breakpoints[i]))
            product_term = (
                -fd_b[i] / delta_b[i - 1]
                - fd_b[i] / delta_b[i]
                + delta_f[i - 1] / denominator_left_sq
                + delta_f[i] / denominator_right_sq
            )
            d_breakpoints[i] += (
                grad_signal_current * product_term
                + delta_f[i - 1] / delta_b[i - 1]
                + delta_f[i] / delta_b[i]
            )

        # --- Step 4: Update internal breakpoints ---
        # We only update the internal breakpoints, not the boundaries.
        learning_rate: float = self.settings.feature_nnbp_learning_rate
        self.breakpoints[1:-1] += learning_rate * d_breakpoints[1:-1]

        while not np.all(np.diff(self.breakpoints) > 0):
            learning_rate = learning_rate / 2
            self.breakpoints[1:-1] -= learning_rate * d_breakpoints[1:-1]

    def _configure_network_from_breakpoints(
        self, observed_function: Callable[[np.ndarray], float]
    ) -> None:
        slopes: list[np.ndarray] = [
            (
                observed_function(self.breakpoints[i + 1])
                - observed_function(self.breakpoints[i])
            )
            / (self.breakpoints[i + 1] - self.breakpoints[i])
            for i in range(len(self.breakpoints) - 1)
        ]
        bias_output = observed_function(self.breakpoints[0])
        weights_hidden = []
        bias_hidden = []
        previous_slope = 0.0
        for i, bp in enumerate(self.breakpoints[:-1]):
            weights_hidden.append((slopes[i] - previous_slope))

            bias_hidden.append(-weights_hidden[i] * bp)
            previous_slope = slopes[i]

        self.weights_and_biases.bias_output = np.array([[bias_output]])
        self.weights_and_biases.weights_hidden = np.array([weights_hidden])
        self.weights_and_biases.bias_hidden = np.array([bias_hidden])
        self.weights_and_biases.weights_output = np.array(
            [[1.0 for _ in range(len(slopes))]]
        ).T

    def _build_and_train_model(self) -> None:
        """Builds and trains the neural network."""
        function_classes: list["ode.OneDimExpression"] = list(
            self.variable.occurring_in.values()
        )
        x_train: np.ndarray = np.random.uniform(
            self.variable.lb,
            self.variable.ub,
            (self.settings.feature_nnbp_nr_of_samples, 1),
        )
        y_train: list[np.ndarray] = []
        for function_class in function_classes:
            y_train_list: list[float] = []
            for x_val in x_train:
                y_train_list.append(function_class.f(x_val[0]))
            y_train.append(np.array(y_train_list).reshape(-1, 1))
        total_loss_queue: list[float] = [
            np.inf for _ in range(self.settings.feature_nnbp_queue_size)
        ]
        epoch: int = 0
        while epoch < self.settings.feature_nnbp_max_epochs:
            if (
                epoch > self.settings.feature_nnbp_queue_size
                and total_loss_queue[0] != 0
                and (total_loss_queue[0] - total_loss_queue[-1])
                / abs(total_loss_queue[0])
                < self.settings.feature_nnbp_convergence_tol
            ):
                break
            epoch += 1
            total_loss: list[np.floating] = []
            for function_index, function_class in enumerate(function_classes):
                self._configure_network_from_breakpoints(function_class.f)
                y_prediction, hidden_layer_output = self._forward_pass(x_train)
                total_loss.append(
                    np.mean((y_train[function_index] - y_prediction) ** 2)
                )
                error: np.ndarray = y_train[function_index] - y_prediction
                self._backward_pass(
                    x_train,
                    error,
                    hidden_layer_output,
                    function_class,
                )
            if total_loss:
                total_loss_queue = total_loss_queue[1:] + [sum(total_loss)]
