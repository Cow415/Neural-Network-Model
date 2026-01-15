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
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output_shape = None
        self.max_indices = None

    def forward(self, input_data):
        """
            Forward pass of Max Pooling.
        """
        self.input = input_data
        batch_size, height, width, channels = input_data.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        self.output_shape = (batch_size, out_height, out_width, channels)
        
        output = np.zeros(self.output_shape)
        self.max_indices = np.zeros(self.output_shape + (2,), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        pool_region = input_data[b, h_start:h_end, w_start:w_end, c]
                        max_val = np.max(pool_region)
                        output[b, i, j, c] = max_val

                        max_index = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        self.max_indices[b, i, j, c] = (h_start + max_index[0], w_start + max_index[1])

        return output

    def backward(self, output_gradient):
        """
            Backward pass of Max Pooling.
        """
        batch_size, height, width, channels = self.input.shape
        input_gradient = np.zeros_like(self.input)

        out_height, out_width = output_gradient.shape[1:3]

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_index, w_index = self.max_indices[b, i, j, c]
                        input_gradient[b, h_index, w_index, c] += output_gradient[b, i, j, c]

        return input_gradient