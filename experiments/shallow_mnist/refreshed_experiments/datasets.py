import numpy as np

from experiments.shallow_mnist.refreshed_experiments.utils import rgb2gray


class mnist:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        (train_features, train_targets), (test_features, test_targets) = mnist.load_data()
        return (train_features, train_targets), (test_features, test_targets)


class grey_cifar10:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) = cifar10.load_data()
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


class grey_cifar100:

    @staticmethod
    def load_data():
        from keras.datasets import cifar100
        (train_features, train_targets), (test_features, test_targets) = cifar100.load_data()
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


def _get_dataset_fraction(dataset, fraction):
    (train_features, train_targets), (test_features, test_targets) = dataset.load_data()
    seed = np.random.get_state()
    # fix the seed for numpy
    np.random.seed(0)
    train_ind = np.random.permutation(range(train_features.shape[0]))[
                :int(train_features.shape[0] * fraction)]
    train_features, train_targets = train_features[train_ind], train_targets[train_ind]
    np.random.set_state(seed)
    return (train_features, train_targets), (test_features, test_targets)


class mnist_5percent:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return _get_dataset_fraction(mnist, 0.05)


class mnist_10percent:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return _get_dataset_fraction(mnist, 0.1)


class mnist_25percent:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return _get_dataset_fraction(mnist, 0.5)


class grey_cifar10_5percent:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) \
            = _get_dataset_fraction(cifar10, 0.05)
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


class grey_cifar10_10percent:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) \
            = _get_dataset_fraction(cifar10, 0.1)
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


class grey_cifar10_25percent:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) \
            = _get_dataset_fraction(cifar10, 0.25)
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)
