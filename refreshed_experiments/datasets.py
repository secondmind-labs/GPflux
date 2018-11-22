import numpy as np

from refreshed_experiments.data_infrastructure import ImageClassificationDataset, \
    MaxNormalisingPreprocessor
from refreshed_experiments.utils import rgb2gray, _get_max_normalised, _mix_train_test, load_svhn


def _get_dataset_fraction(dataset, fraction):
    (train_features, train_targets), (test_features, test_targets) = dataset.load_data()
    seed = np.random.get_state()
    # fix the seed for numpy, so we always get the same fraction of examples
    np.random.seed(0)
    train_ind = np.random.permutation(range(train_features.shape[0]))[
                :int(train_features.shape[0] * fraction)]
    train_features, train_targets = train_features[train_ind], train_targets[train_ind]
    np.random.set_state(seed)
    return (train_features, train_targets), (test_features, test_targets)


def _get_dataset_fixed_examples_per_class(dataset, num_examples):
    (train_features, train_targets), (test_features, test_targets) = dataset.load_data()
    selected_examples = []
    selected_targets = []
    num_classes = set(train_targets)
    for i in num_classes:
        indices = train_targets == i
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
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=10),
                                   cls.__name__)


class mixed_mnist2:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=11),
                                   cls.__name__)


class mixed_mnist3:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=12),
                                   cls.__name__)


class mixed_mnist4:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        return _get_max_normalised(_mix_train_test(mnist.load_data(), random_state=13),
                                   cls.__name__)


class mnist_5percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist, 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_10percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist, 0.10)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_25percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fraction(mnist, 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)


class mnist_100epc:

    @classmethod
    def load_data(cls):
        from keras.datasets import mnist
        data_fraction = _get_dataset_fixed_examples_per_class(mnist, 100)
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


class fashion_mnist_5percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fraction(fashion_mnist, 0.05)
        return _get_max_normalised(data_fraction, cls.__name__)


class fashion_mnist_10percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fraction(fashion_mnist, 0.1)
        return _get_max_normalised(data_fraction, cls.__name__)


class fashion_mnist_25percent:

    @classmethod
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fraction(fashion_mnist, 0.25)
        return _get_max_normalised(data_fraction, cls.__name__)


class fashion_mnist_100epc:

    @classmethod
    def load_data(cls):
        from keras.datasets import fashion_mnist
        data_fraction = _get_dataset_fixed_examples_per_class(fashion_mnist, 100)
        return _get_max_normalised(data_fraction, cls.__name__)


class svhn:

    @classmethod
    def load_data(cls):
        return _get_max_normalised(load_svhn(), cls.__name__)
