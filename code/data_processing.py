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

