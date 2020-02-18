import numpy as np
import pytest

import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, ReLU

from gpflow.kernels import RBF
from gpflow.kullback_leiblers import gauss_kl

from gpflux.layers import LatentVariableLayer
from gpflux.encoders import DirectlyParameterizedNormalDiag

tf.keras.backend.set_floatx("float64")

############
# Utilities
############


@pytest.fixture
def test_data():
    x_dim, y_dim, w_dim = 2, 1, 2
    points = 200
    x_data = np.random.random((points, x_dim)) * 5
    w_data = np.random.random((points, w_dim))
    w_data[: (points // 2), :] = 0.2 * w_data[: (points // 2), :] + 5

    input_data = np.concatenate([x_data, w_data], axis=1)
    y_data = np.random.multivariate_normal(
        mean=np.zeros(points), cov=RBF(variance=0.1).K(input_data), size=y_dim
    ).T
    return x_data[:, :x_dim], y_data


def get_distributions_with_w_dim():
    distributions = []
    for d in [1, 5]:

        mean = np.zeros(d)
        scale_tri_l = np.eye(d)
        mvn = tfp.distributions.MultivariateNormalTriL(mean, scale_tri_l)

        std = np.ones(d)
        mvn_diag = tfp.distributions.MultivariateNormalDiag(mean, std)

        distributions.append((mvn, d))
        distributions.append((mvn_diag, d))
    return distributions


############
# Tests
############


@pytest.mark.parametrize("distribution, w_dim", get_distributions_with_w_dim())
def test_latent_prior_sample(distribution, w_dim):
    lv = LatentVariableLayer(encoder=None, prior=distribution)

    sample_shape = (500, 2)
    assert lv.sample_prior(sample_shape).shape == sample_shape + (w_dim,)


@pytest.mark.parametrize("w_dim", [1, 5])
def test_latent_posterior_sample(test_data, w_dim):
    x_data, y_data = test_data
    x_dim, y_dim = x_data.shape[1], y_data.shape[1]
    num_data = len(x_data)

    means = np.zeros((num_data, w_dim))
    encoder = DirectlyParameterizedNormalDiag(num_data, w_dim, means)

    mean = np.zeros(w_dim)
    var = np.ones(w_dim)
    mvn = tfp.distributions.MultivariateNormalDiag(mean, var)

    lv = LatentVariableLayer(encoder=encoder, prior=mvn)

    recog_data = np.concatenate([x_data, y_data], axis=-1)
    w_data = lv(recog_data, training=False)

    assert w_data.shape == (num_data, w_dim)


@pytest.mark.parametrize("distribution, w_dim", get_distributions_with_w_dim())
def test_local_kls(distribution, w_dim):
    lv = LatentVariableLayer(encoder=None, prior=distribution)

    # test kl is 0 when posteriors == priors
    posterior = distribution
    assert lv.local_kls(posterior) == 0

    # test kl > 0 when posteriors != priors
    batch_size = 10
    params = distribution.parameters
    posterior_params = {
        k: [v + 0.5 for _ in range(batch_size)]
        for k, v in params.items()
        if isinstance(v, np.ndarray)
    }
    posterior = lv.distribution(**posterior_params)
    local_kls = lv.local_kls(posterior)
    assert np.all(local_kls > 0)
    assert local_kls.shape == (batch_size,)


@pytest.mark.parametrize("w_dim", [1, 5])
def test_local_kl_gpflow_consistency(w_dim):
    num_data = 400
    means = np.random.randn(num_data, w_dim)
    encoder = DirectlyParameterizedNormalDiag(num_data, w_dim, means)

    # Prior is N(0, I)
    prior_mean = np.zeros(w_dim)
    prior_var = np.ones(w_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_var)

    lv = LatentVariableLayer(encoder=encoder, prior=prior)
    samples, posteriors = lv.sample_posteriors(recognition_data=None)

    q_mu = posteriors.parameters["loc"]
    q_sqrt = posteriors.parameters["scale_diag"]

    gpflow_local_kls = gauss_kl(q_mu, q_sqrt)
    tfp_local_kls = tf.reduce_sum(lv.local_kls(posteriors))

    np.testing.assert_allclose(tfp_local_kls, gpflow_local_kls, rtol=1e-10)


@pytest.mark.parametrize("w_dim", [1, 5])
def test_add_losses(w_dim):
    num_data = 400
    x_dim = 3
    means = np.random.randn(num_data, w_dim)
    encoder = DirectlyParameterizedNormalDiag(num_data, w_dim, means)

    # Prior is N(0, I)
    prior_mean = np.zeros(w_dim)
    prior_var = np.ones(w_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_var)

    lv = LatentVariableLayer(encoder=encoder, prior=prior)

    inputs = tf.zeros((num_data, x_dim), dtype=tf.float64)

    _ = lv(inputs=inputs)
    assert lv.losses == [0.0]

    _ = lv(inputs=inputs, training=True)
    _, posteriors = lv.sample_posteriors(recognition_data=None)
    local_kls = [tf.reduce_mean(lv.local_kls(posteriors))]
    np.testing.assert_allclose(lv.losses, local_kls)
    assert lv.losses > [0.0]
