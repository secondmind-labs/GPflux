import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow

from gpflux.layers import LikelihoodLayer
from gpflux.losses import LikelihoodLoss


def test_likelihood_layer_and_likelihood_loss_give_equal_results():
    np.random.seed(123)
    f_mean = np.random.randn(7, 1)
    f_scale = np.random.randn(7, 1) ** 2
    targets = np.random.randn(7, 1)

    f_dist = tfp.distributions.MultivariateNormalDiag(loc=f_mean, scale_diag=f_scale)
    likelihood = gpflow.likelihoods.Gaussian(0.123)

    # evaluate layer object
    likelihood_layer = LikelihoodLayer(likelihood)
    _ = likelihood_layer(f_dist, targets=targets, training=True)
    [layer_loss] = likelihood_layer.losses

    # evaluate loss object
    likelihood_loss = LikelihoodLoss(likelihood)
    loss_loss = likelihood_loss(targets, f_dist)

    np.testing.assert_allclose(layer_loss, loss_loss)


# additional tests are in tests/integration/test_svgp_equivalence.py
# comparing both the keras.Sequential/LikelihoodLoss and DeepGP/LikelihoodLayer
# against gpflow.models.SVGP
