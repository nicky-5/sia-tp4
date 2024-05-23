import numpy as np
from src.perceptron import Perceptron
import time


class LinearPerceptron(Perceptron):

    def activation_func(self, value):
        """Identity function"""
        return value

    def compute_deltas(self, indexes: [int]) -> np.array:
        # Extract input data for the given indexes
        input_data = self.data[indexes]

        # Compute s = x * w
        s = np.inner(input_data, self.weights)

        # Compute delta_w using Oja's rule
        delta_w = self.learning_rate * s * (input_data - s * self.weights)

        return delta_w

    def compute_error(self):
        return 1
        # """Mean Squared Error - MSE"""
        # p = self.data.shape[0]
        # output_errors = self.expected_value - self.get_outputs()
        # return np.power(output_errors, 2).sum() / p

    def is_converged(self):
        return False
