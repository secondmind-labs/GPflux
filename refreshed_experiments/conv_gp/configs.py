import gpflow

import gpflux
from refreshed_experiments.utils import Configuration


class GPConfig(Configuration):
    def __init__(self):
        super().__init__()
        self.batch_size = 150
        self.num_epochs = 400
        self.monitor_stats_fraction = 1000


class ConvGPConfig(GPConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.01
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

    def get_optimiser(self, step):
        lr = self.lr * 1.0 / (1 + step // 5000 / 3)
        return gpflow.train.AdamOptimizer(lr)
