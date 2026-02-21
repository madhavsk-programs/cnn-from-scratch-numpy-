import numpy as np
import os
import struct

def load_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, 1, rows, cols)
        images = images.astype(np.float32) / 255.0  # Normalize to 0-1
    return images

def load_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)

def load_data(data_dir="data/raw"):
    train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    print("Loading MNIST dataset from IDX files...")

    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    print(f"Train Images Shape: {X_train.shape}")
    print(f"Train Labels Shape: {y_train.shape}")
    print(f"Test Images Shape: {X_test.shape}")
    print(f"Test Labels Shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()