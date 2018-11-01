from keras.datasets import cifar10, mnist, fashion_mnist

from experiments.shallow_mnist.refreshed_experiments.utils import rgb2gray


class grey_cifar10:

    @staticmethod
    def load_data():
        (train_features, train_targets), (test_features, test_targets) = cifar10.load_data()
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)