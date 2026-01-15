"""
Gradient checking utility for verifying backpropagation implementations.
"""
import numpy as np

def grad_check(f, x, analytic_grad, epsilon=1e-5):
    """
    Perform gradient checking on function f at point x.
    
    Parameters:
    - f: function that takes x and returns loss
    - x: point (numpy array) to check the gradient at
    - analytic_grad: analytically computed gradient at x
    - epsilon: small perturbation for numerical gradient computation
    
    Returns:
    - relative_error: relative error between analytical and numerical gradients
    """
    numerical_grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        x[idx] = old_value + epsilon
        fx_plus_eps = f(x)

        x[idx] = old_value - epsilon
        fx_minus_eps = f(x)

        numerical_grad[idx] = (fx_plus_eps - fx_minus_eps) / (2 * epsilon)

        x[idx] = old_value  # Restore original value
        it.iternext()

    # Compute relative error
    numerator = np.linalg.norm(analytic_grad - numerical_grad)
    denominator = np.linalg.norm(analytic_grad) + np.linalg.norm(numerical_grad)
    relative_error = numerator / denominator

    return relative_error
