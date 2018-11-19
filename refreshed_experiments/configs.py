import gpflow
import keras

import gpflux


class Configuration:

    def summary(self):
        summary_str = ['Configuration parameters:\n']
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                # discard protected and private members
                continue
            if hasattr(value, '__call__'):
                summary_str.append('{} {}\n'.format(name, value.__name__))
            else:
                summary_str.append('{} {}\n'.format(name, str(value)))
        return ''.join(summary_str)

    @property
    def name(self):
        return self.__class__.__name__


class GPConfig(Configuration):
    def __init__(self):
        super().__init__()
        self.batch_size = 110
        self.num_epochs = 550
        self.num_inducing_points = 100
        self.monitor_stats_fraction = 1000
        self.lr = 0.01
        self.store_frequency = 1000

    def get_optimiser(self, step):
        lr = self.lr * 1.0 / (1 + step // 5000 / 3)
        return gpflow.train.AdamOptimizer(lr)


class RBFGPConfig(GPConfig):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.num_epochs = 500
        self.num_inducing_points = 1000
        self.monitor_stats_fraction = 1000
        self.lr = 0.01
        self.store_frequency = 1000

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
        self.with_indexing = False
        self.init_patches = "patches-unique"  # 'patches', 'random'

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)


class TickConvGPConfig(ConvGPConfig):
    def __init__(self):
        super().__init__()
        self.with_indexing = True


class NNConfig(Configuration):
    def __init__(self):
        self.epochs = 1
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adam()
        self.validation_proportion = 0.00
        self.early_stopping = False


class BasicCNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 12
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adam()
        self.early_stopping = False


class MNISTCNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 20
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adadelta()
        self.early_stopping = False


class CifarCNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 200
        self.batch_size = 128
        self.optimiser_parameters = {'lr': 0.0001, 'decay': 1e-6}
        self.optimiser = keras.optimizers.rmsprop(**self.optimiser_parameters)
        self.early_stopping = False
