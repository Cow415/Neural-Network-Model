"""
Dense Layer Class
"""
import numpy as np

class Dense:
    """
    A fully connected dense layer for neural networks
    """
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size))
        self.input_data = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, input_data):
        """
        Compute the forward pass
        """
        # Perform the forward pass
        self.input_data = input_data  # Store input for backpropagation
        forward_output = np.dot(input_data, self.weights.T) + self.biases
        return forward_output

    def backward(self, grad_output):
        """
        Compute gradients with respect to weights, biases, and input data
        """
        # Perform the backward pass
        if self.input_data is None:
            raise ValueError("backward() called before forward()")
        self.grad_weights = np.dot(grad_output.T, self.input_data) / self.input_data.shape[0]
        self.grad_biases = np.mean(grad_output, axis=0)

    def params(self):
        """
        Return weights and biases
        """
        # Return weights and biases
        return [self.weights, self.biases]

    def gradients(self):
        """
        Return gradients of weights and biases
        """
        # Return gradients of weights and biases
        return [self.grad_weights, self.grad_biases]
