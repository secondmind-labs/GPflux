import keras

from experiments.shallow_mnist.refreshed_experiments.utils import Configuration


class NNConfig(Configuration):
    epochs = 1
    batch_size = 128
    optimiser = keras.optimizers.Adam()


class MNISTCNNConfiguration(NNConfig):
    epochs = 100
    batch_size = 128
    optimiser = keras.optimizers.Adadelta()


class CifarCNNConfiguration(NNConfig):
    epochs = 100
    batch_size = 128
    optimiser = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
