import gpflow
import numpy as np

import gpflux


@gpflow.defer_build()
def convgp_creator(dataset, config):
    x = dataset.train_features
    y = dataset.train_targets
    H = int(x.shape[1] ** .5)

    likelihood = gpflow.likelihoods.SoftMax(10)
    num_latents = likelihood.num_classes if hasattr(likelihood, 'num_classes') else 1
    init_patches = 'patches-unique'
    if init_patches == "random":
        patches = gpflux.init.NormalInitializer()
    else:
        unique = init_patches == "patches-unique"
        patches = gpflux.init.PatchSamplerInitializer(x[:100], width=H, height=H, unique=unique)

    layer0 = gpflux.layers.WeightedSumConvLayer(
        [H, H],
        config.num_inducing_points,
        config.patch_shape,
        num_latents=num_latents,
        with_indexing=config.with_indexing,
        with_weights=config.with_weights,
        patches_initializer=patches)

    # layer0.kern.basekern.variance = 25.0
    # layer0.kern.basekern.lengthscales = 1.2
    #
    # # init kernel
    # if config.with_indexing:
    #     layer0.kern.index_kernel.variance = 25.0
    #     layer0.kern.index_kernel.lengthscales = 3.0
    #
    # # break symmetry in variational parameters
    # layer0.q_sqrt = layer0.q_sqrt.read_value()
    # layer0.q_mu = np.random.randn(*(layer0.q_mu.read_value().shape))

    x = x.reshape(dataset.train_features.shape[0], -1)
    y = y.astype(np.int32)
    print(x.shape, y.shape)

    model = gpflux.DeepGP(x, y,
                          layers=[layer0],
                          likelihood=likelihood,
                          batch_size=config.batch_size,
                          name="my_deep_gp")
    return model