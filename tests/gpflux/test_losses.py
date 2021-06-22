import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow

from gpflux.layers import LikelihoodLayer
from gpflux.losses import LikelihoodLoss

tf.keras.backend.set_floatx("float64")


@pytest.fixture(name="fix_random_seed", scope="session")
def _fix_random_seed_fixture():
    old_seed = np.random.seed(123)

    yield

    np.random.seed(old_seed)


@pytest.fixture(name="f_dist", scope="session")
def _f_dist_fixture(fix_random_seed):
    f_mean = np.random.randn(7, 1)
    f_scale = np.random.randn(7, 1) ** 2
    return tfp.distributions.MultivariateNormalDiag(loc=f_mean, scale_diag=f_scale)


def test_likelihood_layer_and_likelihood_loss_give_equal_results(f_dist):
    targets = np.random.randn(7, 1)
    likelihood = gpflow.likelihoods.Gaussian(0.123)

    # evaluate layer object
    likelihood_layer = LikelihoodLayer(likelihood)
    _ = likelihood_layer(f_dist, targets=targets, training=True)
    [layer_loss] = likelihood_layer.losses

    # evaluate loss object
    likelihood_loss = LikelihoodLoss(likelihood)
    loss_loss = likelihood_loss(targets, f_dist)

    np.testing.assert_allclose(layer_loss, loss_loss)


def test_likelihood_loss_from_distribution(f_dist):
    targets = np.random.randn(7, 1)
    likelihood = gpflow.likelihoods.Gaussian(0.123)
    likelihood_loss = LikelihoodLoss(likelihood)
    actual_loss = likelihood_loss.call(targets, f_dist)

    expected_loss = -likelihood.variational_expectations(
        f_dist.loc, f_dist.scale.diag ** 2, targets
    )

    np.testing.assert_allclose(actual_loss, expected_loss)


def test_likelihood_loss_from_samples():
    f_samples = np.random.randn(7, 1)
    targets = np.random.randn(7, 1)
    likelihood = gpflow.likelihoods.Gaussian(0.123)
    likelihood_loss = LikelihoodLoss(likelihood)
    actual_loss = likelihood_loss.call(targets, f_samples)

    expected_loss = -likelihood.log_prob(f_samples, targets)

    np.testing.assert_allclose(actual_loss, expected_loss)


# additional tests are in tests/integration/test_svgp_equivalence.py
# comparing both the keras.Sequential/LikelihoodLoss and DeepGP/LikelihoodLayer
# against gpflow.models.SVGP
