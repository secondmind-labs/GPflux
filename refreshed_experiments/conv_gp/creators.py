import gpflow
import numpy as np

import gpflux
from refreshed_experiments import ConvGPConfig
from refreshed_experiments.utils import reshape_to_2d, \
    labels_onehot_to_int


@gpflow.defer_build()
def convgp_creator(dataset, config: ConvGPConfig):
    x_train, y_train = dataset.train_features, dataset.train_targets
    num_classes = y_train.shape[1]
    # DeepGP class expects 2d inputs and labels encoded with integers
    x_train, y_train = reshape_to_2d(dataset.train_features), \
                       labels_onehot_to_int(reshape_to_2d(dataset.train_targets))
    h = int(x_train.shape[1] ** .5)
    likelihood = gpflow.likelihoods.SoftMax(num_classes)
    patches = config.patch_initializer(x_train[:100], h, h, config.init_patches)

    layer0 = gpflux.layers.WeightedSumConvLayer(
        [h, h],
        config.num_inducing_points,
        config.patch_shape,
        num_latents=num_classes,
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
    layer0.q_mu = np.random.randn(*layer0.q_mu.read_value().shape)
    model = gpflux.DeepGP(x_train, y_train,
                          layers=[layer0],
                          likelihood=likelihood,
                          batch_size=config.batch_size,
                          name="my_deep_gp")
    return model
