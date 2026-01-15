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

def padding_layer(inputs, pad_width):
    """Apply zero-padding to the input data."""
    return np.pad(inputs, pad_width, mode='constant', constant_values=0)

def convolution_layer(inputs, filter_kernel, bias, stride=1):
    """Apply a simple convolution operation, condenses inputs with given kernel"""
    (input_height, input_width) = inputs.shape
    (filter_height, filter_width) = filter_kernel.shape
    new_height = int((input_height - filter_height) / stride) + 1
    new_width = int((input_width - filter_width) / stride) + 1
    convolved = np.zeros((new_height, new_width))
    for h in range(new_height):
        for w in range(new_width):
            vert_start = h * stride
            vert_end = vert_start + filter_height
            horiz_start = w * stride
            horiz_end = horiz_start + filter_width

            a_slice = inputs[vert_start:vert_end, horiz_start:horiz_end]
            convolved[h, w] = np.sum(a_slice * filter_kernel) + bias
    return convolved
