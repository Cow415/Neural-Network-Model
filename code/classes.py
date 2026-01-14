"""
This script includes the declarable classses for various neural network layers and components.
"""

import numpy as np
class dense_layer:
    """A fully connected neural network layer."""
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        """Perform the forward pass."""
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        """Perform the backward pass."""
        weights_gradient = np.dot(self.inputs.T, output_gradient)
        inputs_gradient = np.dot(output_gradient, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return inputs_gradient
