"""" 
Unpack MNIST dataset from raw ubyte files and save as a compressed .npz file.
"""
# Import necessary libraries
import numpy as np
import struct
from pathlib import Path

def load_images(path):
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(n, rows, cols)
        return images

def load_labels(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def main():
    raw = Path("data/raw")

    x_train = load_images(raw / "train-images-idx3-ubyte")
    y_train = load_labels(raw / "train-labels-idx1-ubyte")
    x_test  = load_images(raw / "t10k-images-idx3-ubyte")
    y_test  = load_labels(raw / "t10k-labels-idx1-ubyte")

    # Normalize and reshape for CNN
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0

    x_train = x_train[:, None, :, :]   # (N, 1, 28, 28)
    x_test  = x_test[:, None, :, :]

    np.savez_compressed(
        "data/mnist.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    print("Saved data/mnist.npz")

if __name__ == "__main__":
    main()
