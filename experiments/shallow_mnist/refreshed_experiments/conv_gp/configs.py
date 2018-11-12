import gpflow

import gpflux
from experiments.shallow_mnist.refreshed_experiments.utils import Configuration


class GPConfig(Configuration):
    batch_size = 150
    num_epochs = 200

    @staticmethod
    def get_monitor_tasks():
        raise NotImplementedError()

    @staticmethod
    def get_optimiser():
        raise NotImplementedError()


class ConvGPConfig(GPConfig):
    lr = 0.01
    patch_shape = [5, 5]
    num_inducing_points = 1000
    base_kern = "RBF"
    with_weights = True
    with_indexing = True
    init_patches = "patches-unique"  # 'patches', 'random'
    stats_fraction = 1000

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)

    @staticmethod
    def get_optimiser(step):
        lr = ConvGPConfig.lr * 1.0 / (1 + step // 5000 / 3)
        return gpflow.train.AdamOptimizer(lr)
