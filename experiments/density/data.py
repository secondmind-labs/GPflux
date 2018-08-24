import os
import numpy as np


def _convert_to_numpy_and_store(path, name):
    path_train = os.path.join(path, "binarized_mnist_train.amat")
    path_test = os.path.join(path, "binarized_mnist_test.amat")
    path_store = os.path.join(path, name)

    if not (os.path.exists(path_test) and os.path.exists(path_train)):
        url = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/" \
              "binarized_mnist_{test/train}.amat"
        raise ValueError("Dataset not available, download from: {}".format(url))

    def lines_to_np_ndarray(lines):
        data = np.array([[int(i) for i in line.split(" ")] for line in lines])
        data = np.concatenate(data)
        return np.reshape(data, [-1, 28 ** 2])

    with open(path_train, 'r') as f:
        data_train = lines_to_np_ndarray(f.readlines()).astype(np.float64)

    with open(path_test, 'r') as f:
        data_test = lines_to_np_ndarray(f.readlines()).astype(np.float64)

    np.savez(path_store, train=data_train, test=data_test)
    return {"train": data_train, "test": data_test}


def fixed_binarized_mnist(path):
    name = "binarized_mnist.npz"
    path_dataset = os.path.join(path, name)
    try:
        dataset = np.load(path_dataset)
    except FileNotFoundError:
        dataset = _convert_to_numpy_and_store(path, name)

    return dataset["train"], dataset["test"]

