"""
    Contains ReLU activation layer implementation.
    And Leaky ReLU variant.
"""

class ReLU:
    """
        ReLU activation layer.
    """
    def __init__(self):
        self.mask = None

    def forward(self, input_data):
        """
            Forward pass of ReLU activation.
        """
        self.mask = input_data > 0
        return input_data * self.mask
    def backward(self, output_gradient):
        """
            Backward pass of ReLU activation.
        """
        return output_gradient * self.mask
    def parameters(self):
        """Null Parameters method"""
        return []  # No parameters in ReLU layer
    def gradients(self):
        """Null Gradients method"""
        return []   # No gradients in ReLU layer
