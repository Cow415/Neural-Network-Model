"""
    Contains ReLU activation layer implementation.
    And Leaky ReLU variant.
"""
import numpy as np

class ReLU:
    """
        ReLU activation layer.
    """
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        """
            Forward pass of ReLU activation.
        """
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        """
            Backward pass of ReLU activation.
        """
        relu_grad = self.input > 0
        return output_gradient * relu_grad

class LeakyReLU:
    """
        Leaky ReLU activation layer.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input = None

    def forward(self, input_data):
        """
            Forward pass of Leaky ReLU activation.
        """
        self.input = input_data
        return np.where(input_data > 0, input_data, self.alpha * input_data)

    def backward(self, output_gradient):
        """
            Backward pass of Leaky ReLU activation.
        """
        leaky_relu_grad = np.where(self.input > 0, 1, self.alpha)
        return output_gradient * leaky_relu_grad