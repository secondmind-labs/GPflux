import pickle
from pathlib import Path

from refreshed_experiments.utils import rgb2gray, get_dataset_fraction, \
    get_dataset_fixed_examples_per_class

_CACHE_DIR = Path('/tmp/.datasets')


def get_cached(name):
    cached = set(_CACHE_DIR.iterdir())
    if name in cached:
        with (_CACHE_DIR / Path(name)).open(mode='rb') as f_handle:
            return pickle.load(f_handle)


class mnist:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        (train_features, train_targets), (test_features, test_targets) = mnist.load_data()
        return (train_features, train_targets), (test_features, test_targets)


class mnist_5percent:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return get_dataset_fraction(mnist, 0.05)


class mnist_10percent:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return get_dataset_fraction(mnist, 0.1)


class mnist_25percent:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return get_dataset_fraction(mnist, 0.5)


class mnist_1epc:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return get_dataset_fixed_examples_per_class(mnist, 1)


class mnist_10epc:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return get_dataset_fixed_examples_per_class(mnist, 10)


class mnist_100epc:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return get_dataset_fixed_examples_per_class(mnist, 100)


class mnist_500epc:

    @staticmethod
    def load_data():
        from keras.datasets import mnist
        return get_dataset_fixed_examples_per_class(mnist, 500)


class grey_cifar10:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) = cifar10.load_data()
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


class grey_cifar10_5percent:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) \
            = get_dataset_fraction(cifar10, 0.05)
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


class grey_cifar10_10percent:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) \
            = get_dataset_fraction(cifar10, 0.1)
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


class grey_cifar10_25percent:

    @staticmethod
    def load_data():
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) \
            = get_dataset_fraction(cifar10, 0.25)
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


class fashion_mnist:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        (train_features, train_targets), (test_features, test_targets) = fashion_mnist.load_data()
        return (train_features, train_targets), (test_features, test_targets)


class fashion_mnist_5percent:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        return get_dataset_fraction(fashion_mnist, 0.05)


class fashion_mnist_10percent:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        return get_dataset_fraction(fashion_mnist, 0.10)


class fashion_mnist_25percent:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        return get_dataset_fraction(fashion_mnist, 0.25)


class fashion_mnist_1epc:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        return get_dataset_fixed_examples_per_class(fashion_mnist, 1)


class fashion_mnist_10epc:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        return get_dataset_fixed_examples_per_class(fashion_mnist, 10)


class fashion_mnist_100epc:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        return get_dataset_fixed_examples_per_class(fashion_mnist, 100)


class fashion_mnist_500epc:

    @staticmethod
    def load_data():
        from keras.datasets import fashion_mnist
        return get_dataset_fixed_examples_per_class(fashion_mnist, 500)
