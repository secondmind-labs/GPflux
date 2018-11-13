import keras

from refreshed_experiments.utils import Configuration


class NNConfig(Configuration):
    def __init__(self):
        self.epochs = 1
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adam()
        self.validation_proportion = 0.2
        self.early_stopping = True


class MNISTCNNConfiguration(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adadelta()


class CifarCNNConfiguration(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.batch_size = 128
        self.optimiser_parameters = {'lr': 0.0001, 'decay': 1e-6}
        self.optimiser = keras.optimizers.rmsprop(**self.optimiser_parameters)
