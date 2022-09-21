#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import abc

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.kullback_leiblers import gauss_kl

from gpflux.encoders import DirectlyParameterizedNormalDiag
from gpflux.layers import LatentVariableLayer, LayerWithObservations, TrackableLayer

tf.keras.backend.set_floatx("float64")

############
# Utilities
############


def _zero_one_normal_prior(w_dim):
    """N(0, I) prior"""
    return tfp.distributions.MultivariateNormalDiag(loc=np.zeros(w_dim), scale_diag=np.ones(w_dim))


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
def test_local_kls(distribution, w_dim):
    lv = LatentVariableLayer(encoder=None, prior=distribution)

    # test kl is 0 when posteriors == priors
    posterior = distribution
    assert lv._local_kls(posterior) == 0

    # test kl > 0 when posteriors != priors
    batch_size = 10
    params = distribution.parameters
    posterior_params = {
        k: [v + 0.5 for _ in range(batch_size)]
        for k, v in params.items()
        if isinstance(v, np.ndarray)
    }
    posterior = lv.distribution_class(**posterior_params)
    local_kls = lv._local_kls(posterior)
    assert np.all(local_kls > 0)
    assert local_kls.shape == (batch_size,)


@pytest.mark.parametrize("w_dim", [1, 5])
def test_local_kl_gpflow_consistency(w_dim):
    num_data = 400
    means = np.random.randn(num_data, w_dim)
    encoder = DirectlyParameterizedNormalDiag(num_data, w_dim, means)

    lv = LatentVariableLayer(encoder=encoder, prior=_zero_one_normal_prior(w_dim))
    posteriors = lv._inference_posteriors(
        [np.random.randn(num_data, 3), np.random.randn(num_data, 2)]
    )

    q_mu = posteriors.parameters["loc"]
    q_sqrt = posteriors.parameters["scale_diag"]

    gpflow_local_kls = gauss_kl(q_mu, q_sqrt)
    tfp_local_kls = tf.reduce_sum(lv._local_kls(posteriors))

    np.testing.assert_allclose(tfp_local_kls, gpflow_local_kls, rtol=1e-10)


class ArrayMatcher:
    def __init__(self, expected):
        self.expected = expected

    def __eq__(self, actual):
        return np.allclose(actual, self.expected, equal_nan=True)


@pytest.mark.parametrize("w_dim", [1, 5])
def test_latent_variable_layer_losses(mocker, w_dim):
    num_data, x_dim, y_dim = 43, 3, 1

    prior_shape = (w_dim,)
    posteriors_shape = (num_data, w_dim)

    prior = tfp.distributions.MultivariateNormalDiag(
        loc=np.random.randn(*prior_shape),
        scale_diag=np.random.randn(*prior_shape) ** 2,
    )
    posteriors = tfp.distributions.MultivariateNormalDiag(
        loc=np.random.randn(*posteriors_shape),
        scale_diag=np.random.randn(*posteriors_shape) ** 2,
    )

    encoder = mocker.Mock(return_value=(posteriors.loc, posteriors.scale.diag))

    lv = LatentVariableLayer(encoder=encoder, prior=prior)

    inputs = np.full((num_data, x_dim), np.nan)
    targets = np.full((num_data, y_dim), np.nan)
    observations = [inputs, targets]
    encoder_inputs = np.concatenate(observations, axis=-1)

    _ = lv(inputs)
    encoder.assert_not_called()
    assert lv.losses == [0.0]

    _ = lv(inputs, observations=observations, training=True)

    # assert_called_once_with uses == for comparison which fails on arrays
    encoder.assert_called_once_with(ArrayMatcher(encoder_inputs), training=True)

    expected_loss = [tf.reduce_mean(posteriors.kl_divergence(prior))]
    np.testing.assert_equal(lv.losses, expected_loss)  # also checks shapes match


@pytest.mark.parametrize("w_dim", [1, 5])
@pytest.mark.parametrize("seed2", [None, 42])
def test_latent_variable_layer_samples(mocker, test_data, w_dim, seed2):
    seed = 123

    inputs, targets = test_data
    num_data, x_dim = inputs.shape

    prior_shape = (w_dim,)
    posteriors_shape = (num_data, w_dim)

    prior = tfp.distributions.MultivariateNormalDiag(
        loc=np.random.randn(*prior_shape),
        scale_diag=np.random.randn(*prior_shape) ** 2,
    )
    posteriors = tfp.distributions.MultivariateNormalDiag(
        loc=np.random.randn(*posteriors_shape),
        scale_diag=np.random.randn(*posteriors_shape) ** 2,
    )

    encoder = mocker.Mock(return_value=(posteriors.loc, posteriors.scale.diag))

    lv = LatentVariableLayer(prior=prior, encoder=encoder)

    tf.random.set_seed(seed)
    sample_prior = lv(inputs, seed=seed2)
    tf.random.set_seed(seed)
    prior_expected = np.concatenate([inputs, prior.sample(num_data, seed=seed2)], axis=-1)
    np.testing.assert_array_equal(sample_prior, prior_expected)

    tf.random.set_seed(seed)
    sample_posterior = lv(inputs, observations=[inputs, targets], training=True, seed=seed2)
    tf.random.set_seed(seed)
    posterior_expected = np.concatenate([inputs, posteriors.sample(seed=seed2)], axis=-1)
    np.testing.assert_array_equal(sample_posterior, posterior_expected)


def test_no_tensorflow_metaclass_overwritten():
    """
    LayerWithObservations is a subclass of tf.keras.layers.Layer (via TrackableLayer);
    this test ensures that TrackableLayer does not have a metaclass, and hence by adding
    the ABCMeta to LayerWithObservations we are not accidentally removing some required
    TensorFlow magic metaclass.
    """
    assert LayerWithObservations.__bases__ == (TrackableLayer,)
    assert type(TrackableLayer) is type
    assert type(LayerWithObservations) is abc.ABCMeta
