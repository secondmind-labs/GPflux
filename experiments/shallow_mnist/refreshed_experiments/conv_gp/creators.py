import gpflow
import numpy as np

import gpflux
from experiments.shallow_mnist.refreshed_experiments.conv_gp.configs import ConvGPConfig
from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import Dataset, \
    ImageClassificationDataset

"""
One would implement a model creator and corresponding config to try a new model. The rest should
be done automatically.
"""


@gpflow.defer_build()
def convgp_creator(dataset: ImageClassificationDataset, config: ConvGPConfig):
    x, y = dataset.train_features, dataset.train_targets
    num_classes = y.shape[1]
    # DeepGP class expects 2d inputs and labels encoded with integers
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1).astype(np.int32).argmax(axis=-1)[
        ..., None]

    h = int(x.shape[1] ** .5)
    likelihood = gpflow.likelihoods.SoftMax(num_classes)

    num_latents = likelihood.num_classes if hasattr(likelihood, 'num_classes') else 1

    patches = config.patch_initializer(x[:100], h, h, config.init_patches)

    layer0 = gpflux.layers.WeightedSumConvLayer(
        [h, h],
        config.num_inducing_points,
        config.patch_shape,
        num_latents=num_latents,
        with_indexing=config.with_indexing,
        with_weights=config.with_weights,
        patches_initializer=patches)

    layer0.kern.basekern.variance = 25.0
    layer0.kern.basekern.lengthscales = 1.2

    if config.with_indexing:
        layer0.kern.index_kernel.variance = 25.0
        layer0.kern.index_kernel.lengthscales = 3.0

    # break symmetry in variational parameters
    layer0.q_sqrt = layer0.q_sqrt.read_value()
    layer0.q_mu = np.random.randn(*(layer0.q_mu.read_value().shape))

    model = gpflux.DeepGP(x, y,
                          layers=[layer0],
                          likelihood=likelihood,
                          batch_size=config.batch_size,
                          name="my_deep_gp")
    return model
