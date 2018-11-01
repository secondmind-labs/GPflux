import keras

from experiments.shallow_mnist.refreshed_experiments.utils import Configuration


class MNISTCNNConfiguration(Configuration):
    num_updates = 300
    batch_size = 128
    optimiser = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


class CifarCNNConfiguration(Configuration):
    num_updates = 500
    batch_size = 128
    optimiser = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
