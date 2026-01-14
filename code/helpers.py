"""
This python file contains helper functions for various tasks that comes to computing.
"""

# Necessary imports
import numpy as np

def relu_activate(x):
    """Apply the ReLU activation function."""
    return np.maximum(0, x)

def sigmoid_activate(x):
    """Apply the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
