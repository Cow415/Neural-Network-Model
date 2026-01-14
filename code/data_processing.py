""" 
This script includes all fuctions used to pull and process data.
"""

# Necessary imports
import numpy as np
from PIL import Image

def load_image(image_path):
    """Load an image from the specified path into a numpy array."""
    loaded = Image.open(image_path)
    return np.array(loaded)

def normalize_data(data):
    """Normalize the input data to have values between 0 and 1."""
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = (data - data_min) / (data_max - data_min)
    return normalized

def flatten_data(data):
    """Flatten the input data to a 1D array."""
    return data.flatten()

def pooling_layer(inputs, pool_size, stride):
    """Apply a simple max pooling operation."""
    (input_height, input_width) = inputs.shape
    new_height = int(1 + (input_height - pool_size) / stride)
    new_width = int(1 + (input_width - pool_size) / stride)
    pooled = np.zeros((new_height, new_width))
    for h in range(new_height):
        for w in range(new_width):
            vert_start = h * stride
            vert_end = vert_start + pool_size
            horiz_start = w * stride
            horiz_end = horiz_start + pool_size

            a_slice = inputs[vert_start:vert_end, horiz_start:horiz_end]
            pooled[h, w] = np.max(a_slice)
    return pooled
