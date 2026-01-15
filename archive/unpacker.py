import numpy as np
import struct
import gzip
import os

def _parse_ubyte(f):
    # Read the magic number and dimensions
    f.seek(0)
    # The '>' indicates big-endian byte order
    magic = struct.unpack('>4B', f.read(4)) 

    # Check if it is an image file (magic[2] == 3) or label file (magic[2] == 1)
    if magic[2] == 3:
        # Image file: read dimensions
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        # Read the rest of the data as a 1D array of uint8 and reshape
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape((num_images, num_rows, num_cols))
        return data
    elif magic[2] == 1:
        # Label file: read number of items
        num_items = struct.unpack('>I', f.read(4))[0]
        # Read the rest of the data as a 1D array of uint8
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(num_items)
        return data
    else:
        raise ValueError("Unknown ubyte file format")
