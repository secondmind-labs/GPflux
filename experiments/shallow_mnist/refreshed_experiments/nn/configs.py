import keras

from experiments.shallow_mnist.refreshed_experiments.utils import Configuration


class NNConfig(Configuration):
    num_updates = 128
    batch_size = 128
    optimiser = keras.optimizers.Adam()


class MNISTCNNConfiguration(NNConfig):
    num_updates = 300
    batch_size = 128
    optimiser = keras.optimizers.Adadelta()


class CifarCNNConfiguration(NNConfig):
    num_updates = 500
    batch_size = 128
    optimiser = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
