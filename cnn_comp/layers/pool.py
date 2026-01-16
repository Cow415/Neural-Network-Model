""" 
    For pooling layers.
    No parameter & reduction in dimensionality.
    Route gradient to max index.
"""
import numpy as np

class MaxPool2D:
    """
        Max Pooling layer for 2D inputs.
    """
    def __init__(self, k, stride=2):
        self.k = k
        self.stride = stride
        self.input = None
        self.argmax = {}  # To store the indices of max values

    def forward(self, input_data):
        """
            Forward pass of Max Pooling.
        """
        self.input = input_data
        batch_size, height, width, channels = input_data.shape
        out_height = height // self.k
        out_width = width // self.k

        out = np.zeros((batch_size, out_height, out_width, channels))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h = i * self.stride
                        w= j * self.stride

                        patch = input_data[b, h:h+self.k, w:w+self.k]
                        idx = np.argmax(patch)
                        out[b, i, j, c] = patch.flatten()[idx]
                        self.argmax[(b,c,i,j)] = (h + idx // self.k, w + idx % self.k)
        return out

    def backward(self, d_out):
        """
            Backward pass of Max Pooling.
        """
        dx = np.zeros_like(self.input)
        for (b,c,i,j), (h,w) in self.argmax.items():
            dx[b, h, w, c] += d_out[b, i, j, c]
        return dx

    def parameters(self):
        """Null Parameters method"""
        return []  # No parameters in pooling layer

    def gradients(self):
        """Null Gradients method"""
        return []   # No gradients in pooling layer
