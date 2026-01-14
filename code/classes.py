"""
This script includes the declarable classses for various neural network layers and components.
"""
# Necessary imports
import numpy as np

# Network elements
def neuron_layer(inputs, weights, bias):
    """Compute the output of a neuron layer."""
    return np.dot(inputs, weights.T) + bias
