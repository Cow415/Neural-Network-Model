"""
2D Convolutional Layer Class
"""
import numpy as np

class Conv2D:
    """2D Convolutional layer."""
    def __init__(self, in_c, out_c, k, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.w = np.random.randn(out_c, in_c, k, k) * np.sqrt(2. / (in_c * k * k))
        self.b = np.zeros((out_c, 1))
    
    def forward(self, x):
        self.input = x  # Store input for backpropagation
        
