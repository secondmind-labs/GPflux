import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def _shuffle_X_and_Y(X, Y):
    assert len(X) == len(Y)
    shuffled_idx = np.random.permutation(len(X))
    return X[shuffled_idx], Y[shuffled_idx]

def mnist():
    """
    returns:
     - Train images (X) 55k x 784
     - Train labels (Y) 55k x 1
     - Test images (Xs) 10k x 784
     - Test labels (Ys) 10k x 1
    """
    PATH = "./data_mnist"
    mnist = input_data.read_data_sets(PATH, one_hot=False)
    X, Y = mnist.train.images, mnist.train.labels
    Xs, Ys = mnist.test.images, mnist.test.labels
    X = X.astype(float)
    Xs = Xs.astype(float)
    Y = Y.astype(float).reshape(-1, 1)
    Ys = Ys.astype(float).reshape(-1, 1)
    return X, Y, Xs, Ys

def mnist01():
    """
    returns:
     - Train images (X)
     - Train labels (Y)
     - Test images (Xs)
     - Test labels (Ys)
    """
    X_new, Y_new, Xs_new, Ys_new = [], [], [], []
    X, Y, Xs, Ys = mnist()
    for digit in [0, 1]:
        # train
        train_indices = (Y == digit).flatten()
        X_new.append(X[train_indices, :])
        Y_new.append(Y[train_indices, :])

        # test
        test_indices = (Ys == digit).flatten()
        Xs_new.append(Xs[test_indices, :])
        Ys_new.append(Ys[test_indices, :])

    X = np.concatenate(X_new, axis=0)
    Y = np.concatenate(Y_new, axis=0)
    X, Y = _shuffle_X_and_Y(X, Y)
    Xs = np.concatenate(Xs_new, axis=0)
    Ys = np.concatenate(Ys_new, axis=0)
    Xs, Ys = _shuffle_X_and_Y(Xs, Ys)
    return X, Y, Xs, Ys
