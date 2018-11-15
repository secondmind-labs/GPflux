import gpflow
import keras

import gpflux


class Configuration:

    def summary(self):
        summary_str = []
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                # discard protected and private members
                continue
            if hasattr(value, '__call__'):
                summary_str.append('{} {}\n'.format(name, value.__name__))
            else:
                summary_str.append('{} {}\n'.format(name, str(value)))
        return ''.join(summary_str)

    @classmethod
    @property
    def name(cls):
        return cls.__name__


class GPConfig(Configuration):
    def __init__(self):
        super().__init__()
        self.batch_size = 150
        self.num_epochs = 4
        self.monitor_stats_fraction = 1000
        self.lr = 0.01

    def get_optimiser(self, step):
        lr = self.lr * 1.0 / (1 + step // 5000 / 3)
        return gpflow.train.AdamOptimizer(lr)


class ConvGPConfig(GPConfig):
    def __init__(self):
        super().__init__()
        self.patch_shape = [5, 5]
        self.num_inducing_points = 1000
        self.base_kern = "RBF"
        self.with_weights = True
        self.with_indexing = True
        self.init_patches = "patches-unique"  # 'patches', 'random'

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)


class NNConfig(Configuration):
    def __init__(self):
        self.epochs = 1
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adam()
        self.validation_proportion = 0.1
        self.early_stopping = True


class MNISTCNNConfiguration(NNConfig):
    def __init__(self, early_stopping=True):
        super().__init__()
        self.epochs = 20
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adadelta()
        self.early_stopping = True


class CifarCNNConfiguration(NNConfig):
    def __init__(self, early_stopping=True):
        super().__init__()
        self.epochs = 200
        self.batch_size = 128
        self.optimiser_parameters = {'lr': 0.0001, 'decay': 1e-6}
        self.optimiser = keras.optimizers.rmsprop(**self.optimiser_parameters)
        self.early_stopping = early_stopping
