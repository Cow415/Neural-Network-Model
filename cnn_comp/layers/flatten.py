"""
    Flatten Layer for CNNs
"""
class Flatten:
    """
    A layer that flattens the input tensor into a 2D array.
    """
    def __init__(self):
        self.shape = None

    def forward(self, x):
        """
        Forward pass to flatten the input tensor.
        """
        self.shape = x.shape
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, output_gradient):
        """
        Backward pass to reshape the gradient to the original input shape.
        """
        return output_gradient.reshape(self.shape)

    def parameters(self):
        """
        Flatten layer has no parameters.
        """
        return []
    def gradients(self):
        """
        Flatten layer has no gradients.
        """
        return []
