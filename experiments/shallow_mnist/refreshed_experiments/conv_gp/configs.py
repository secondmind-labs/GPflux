import gpflow

import gpflux
from experiments.shallow_mnist.refreshed_experiments.utils import Configuration
import gpflow.training.monitor as mon
import numpy as np


class GPConfig(Configuration):
    batch_size = 128
    optimiser = gpflow.train.AdamOptimizer(0.001)
    num_epochs = 500

    @staticmethod
    def get_monitor_tasks():
        raise NotImplementedError()

    @staticmethod
    def get_optimiser():
        raise NotImplementedError()


class ConvGPConfig(GPConfig):
    model_type = "convgp"

    lr_cfg = {
        "decay": "custom",
        "lr": 1e-4
    }

    num_updates = 50000
    stats_freq = 500
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

    @staticmethod
    def get_optimiser(step):
        if ConvGPConfig.lr_cfg['decay'] == "custom":
            print("Custom decaying lr")
            lr = ConvGPConfig.lr_cfg['lr'] * 1.0 / (1 + step // 5000 / 3)
        else:
            lr = ConvGPConfig.lr_cfg['lr']
        return gpflow.train.AdamOptimizer(lr)
