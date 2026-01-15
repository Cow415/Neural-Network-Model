"""
Dense Layer Class
"""
import numpy as np

class Dense: 
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size))

    def forward(self, input_data):
        # Perform the forward pass
        self.input_data = input_data  # Store input for backpropagation
        forward_output = np.dot(input_data, self.weights.T) + self.biases
        return forward_output
    
    def backward(self, grad_output, learning_rate):