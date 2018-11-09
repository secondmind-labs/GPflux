import gpflow

import gpflux
from experiments.shallow_mnist.refreshed_experiments.utils import Configuration


class GPConfig(Configuration):
    batch_size = 128
    num_epochs = 150

    @staticmethod
    def get_monitor_tasks():
        raise NotImplementedError()

    @staticmethod
    def get_optimiser():
        raise NotImplementedError()


class ConvGPConfig(GPConfig):

    lr_cfg = {
        "decay": "custom",
        "lr": 1e-4
    }

    patch_shape = [5, 5]
    num_inducing_points = 1000
    base_kern = "RBF"
    with_weights = True
    with_indexing = True
    init_patches = "patches-unique"  # 'patches', 'random'

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)

    @staticmethod
    def get_optimiser(step):
        if ConvGPConfig.lr_cfg['decay'] == "custom":
            lr = ConvGPConfig.lr_cfg['lr'] * 1.0 / (1 + step // 5000 / 3)
        else:
            lr = ConvGPConfig.lr_cfg['lr']
        return gpflow.train.AdamOptimizer(lr)
