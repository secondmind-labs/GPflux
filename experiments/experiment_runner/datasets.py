# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np

from experiments.experiment_runner.utils import _get_max_normalised, _mix_train_test, load_svhn, \
    load_grey_cifar, numpy_fixed_seed


def _get_dataset_fraction(dataset, fraction):
    (train_features, train_targets), (test_features, test_targets) = dataset
    seed = np.random.get_state()
    train_ind = np.random.permutation(range(train_features.shape[0]))[
                :int(train_features.shape[0] * fraction)]
    train_features, train_targets = train_features[train_ind], train_targets[train_ind]
    np.random.set_state(seed)
    return (train_features, train_targets), (test_features, test_targets)


def _get_dataset_fixed_examples_per_class(dataset, num_examples):
    (train_features, train_targets), (test_features, test_targets) = dataset
    selected_examples = []
    selected_targets = []
    classes = set(train_targets.ravel())
    for i in classes:
        indices = (train_targets == i)
        selected_examples.append(train_features[indices][:num_examples])
        selected_targets.append(train_targets[indices][:num_examples])
    selected_examples = np.vstack(selected_examples)
    selected_targets = np.hstack(selected_targets)
    return (selected_examples, selected_targets), (test_features, test_targets)


class mnist:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(mnist.load_data(), cls.__name__)


class mixed_mnist1:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=10),
                                   cls.__name__)


class mixed_mnist2:

    @classmethod
    @numpy_fixed_seed(2)
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=11),
                                   cls.__name__)


class mixed_mnist3:

    @classmethod
    @numpy_fixed_seed(3)
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=12),
                                   cls.__name__)


class mixed_mnist4:

    @classmethod
    @numpy_fixed_seed(4)
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=13),
                                   cls.__name__)


class mnist_5percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist.load_data(), 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_10percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist.load_data(), 0.10)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_25percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist.load_data(), 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_100epc:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fixed_examples_per_class(mnist.load_data(), 100)
        return _get_max_normalised(data_fraction, cls.__name__)


class random_mnist_5percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist.load_data(), 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class random_mnist_10percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist.load_data(), 0.10)
        return _get_max_normalised(data_fraction, cls.__name__)


class random_mnist_25percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist.load_data(), 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)


class random_mnist_100epc:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fixed_examples_per_class(mnist.load_data(), 100)
        return _get_max_normalised(data_fraction, cls.__name__)


class grey_cifar10:

    @classmethod
    def load_data(cls):
        dataset = load_grey_cifar()
        return _get_max_normalised(dataset, cls.__name__)


class grey_cifar10_100epc:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fixed_examples_per_class(load_grey_cifar(), 100)
        return _get_max_normalised(data_fraction, cls.__name__)


class grey_cifar10_5percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fraction(load_grey_cifar(), 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class grey_cifar10_10percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fraction(load_grey_cifar(), 0.1)
        return _get_max_normalised(data_fraction, cls.__name__)


class grey_cifar10_25percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fraction(load_grey_cifar(), 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)


class fashion_mnist:

    @classmethod
    def load_data(cls):
        from keras.datasets import fashion_mnist
        return _get_max_normalised(fashion_mnist.load_data(), cls.__name__)


class fashion_mnist_5percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fraction(fashion_mnist.load_data(), 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class fashion_mnist_10percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fraction(fashion_mnist.load_data(), 0.1)
        return _get_max_normalised(data_fraction, cls.__name__)


class fashion_mnist_25percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fraction(fashion_mnist.load_data(), 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)


class fashion_mnist_100epc:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fixed_examples_per_class(fashion_mnist.load_data(), 100)
        return _get_max_normalised(data_fraction, cls.__name__)


class svhn:

    @classmethod
    def load_data(cls):
        return _get_max_normalised(load_svhn(), cls.__name__)


class svhn_100epc:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fixed_examples_per_class(load_svhn(), 100)
        return _get_max_normalised(data_fraction, cls.__name__)


class svhn_5percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fraction(load_svhn(), 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class svhn_10percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fraction(load_svhn(), 0.1)
        return _get_max_normalised(data_fraction, cls.__name__)


class svhn_25percent:

    @classmethod
    @numpy_fixed_seed(1)
    def load_data(cls):
        data_fraction = _get_dataset_fraction(load_svhn(), 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)
