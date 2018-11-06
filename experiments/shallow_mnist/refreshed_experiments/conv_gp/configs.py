import gpflow

import gpflux
from experiments.shallow_mnist.refreshed_experiments.utils import Configuration


class GPConfig(Configuration):
    batch_size = 128
    optimiser = gpflow.train.AdamOptimizer(0.001)
    iterations = 5000

    def get_monitor_tasks(self):
        raise NotImplementedError()

    def get_optimiser(self):
        raise NotImplementedError()


class ConvGPConfig(GPConfig):
    model_type = "convgp"

    lr_cfg = {
        "decay": "custom",
        "lr": 1e-4
    }

    iterations = 50000
    patch_shape = [5, 5]
    batch_size = 128
    num_inducing_points = 1000
    base_kern = "RBF"
    with_weights = True
    with_indexing = True
    init_patches = "patches-unique"  # 'patches', 'random'
    restore = False

    # print hz
    hz = {
        'slow': 1000,
        'short': 50
    }

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)

    def get_monitor_tasks(self):
        pass

    def get_optimiser(self):
        pass
