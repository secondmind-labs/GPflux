import gpflow

from experiments.shallow_mnist.refreshed_experiments.utils import Configuration


class ConvGPConfig(Configuration):
    num_inducing_points = 1000
    patch_shape = [5, 5]
    with_indexing = True
    with_weights = True
    batch_size = 128
    optimiser = gpflow.train.AdamOptimizer(0.001)
    iterations = 5000