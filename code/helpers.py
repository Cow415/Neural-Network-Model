"""
This python file contains helper functions for various tasks that comes to computing.
"""

# Necessary imports
import numpy as np

# Activation functions
def relu_activate(x):
    """Apply the ReLU activation function."""
    return np.maximum(0, x)

def sigmoid_activate(x):
    """Apply the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def softmax_activate(x):
    """Apply the softmax activation function."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Loss functions
def mean_squared_error(y_true, y_pred):
    """Compute the Mean Squared Error loss."""
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    """Compute the Cross-Entropy loss."""
    m = y_true.shape[0]
    p = softmax_activate(y_pred)
    log_likelihood = -np.log(p[range(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# Optimization functions
def gradient_descent(weights, gradients, learning_rate):
    """Update weights using gradient descent."""
    return weights - learning_rate * gradients
