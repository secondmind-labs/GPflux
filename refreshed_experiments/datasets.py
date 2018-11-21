import os
import subprocess
from scipy.io import loadmat
import numpy as np

from refreshed_experiments.data_infrastructure import ImageClassificationDataset, \
    MaxNormalisingPreprocessor
from refreshed_experiments.utils import rgb2gray, get_dataset_fraction, \
    get_dataset_fixed_examples_per_class


def _get_max_normalised(data, name):
    (train_features, train_targets), (test_features, test_targets) = data
    dataset = \
        ImageClassificationDataset.from_train_test_split(name,
                                                         train_features=train_features,
                                                         train_targets=train_targets,
                                                         test_features=test_features,
                                                         test_targets=test_targets)
    return MaxNormalisingPreprocessor.preprocess(dataset)


class mnist:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(mnist.load_data(), cls.__name__)


class mnist_5percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = get_dataset_fraction(mnist, 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_10percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = get_dataset_fraction(mnist, 0.10)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_25percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = get_dataset_fraction(mnist, 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_1epc:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = get_dataset_fixed_examples_per_class(mnist, 1)
        return _get_max_normalised(data_fraction, cls.__name__)


class grey_cifar10:

    @classmethod
    def load_data(cls):
        from keras.datasets import cifar10
        (train_features, train_targets), (test_features, test_targets) = cifar10.load_data()
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        dataset = \
            ImageClassificationDataset.from_train_test_split(cls.__name__,
                                                             train_features=train_features,
                                                             train_targets=train_targets,
                                                             test_features=test_features,
                                                             test_targets=test_targets)
        return MaxNormalisingPreprocessor.preprocess(dataset)


class fashion_mnist:

    @classmethod
    def load_data(cls):
        from keras.datasets import fashion_mnist
        return _get_max_normalised(fashion_mnist.load_data(), cls.__name__)


class svhn:

    @classmethod
    def load_data(cls):
        if not os.path.exists('/tmp/svhn_train.mat'):
            subprocess.call(
                ["wget", "-O", "/tmp/svhn_train.mat",
                 "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"])
        if not os.path.exists('/tmp/svhn_test.mat'):
            subprocess.call(
                ["wget", "-O", "/tmp/svhn_test.mat",
                 "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"])
        train_data = loadmat('/tmp/svhn_train.mat')
        test_data = loadmat('/tmp/svhn_test.mat')
        x_train, y_train = train_data['X'], train_data['y']
        x_test, y_test = test_data['X'], test_data['y']
        x_train, x_test = np.transpose(x_train, [3, 0, 1, 2]), np.transpose(x_test, [3, 0, 1, 2])
        x_train, x_test = rgb2gray(x_train), rgb2gray(x_test)
        num_classes = len(set(y_train.ravel()))
        y_train, y_test = y_train == np.arange(num_classes)[None, :], y_test == np.arange(
            num_classes)[None, :]
        data = (x_train, y_train), (x_test, y_test)
        return _get_max_normalised(data, cls.__name__)
