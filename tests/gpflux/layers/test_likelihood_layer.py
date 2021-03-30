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
import numpy as np
import pytest
import tensorflow as tf

from gpflow.kernels import Matern52
from gpflow.likelihoods import Bernoulli, Beta, Gaussian, Poisson
from gpflow.mean_functions import Zero

from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.layers import GPLayer, LikelihoodLayer
from gpflux.layers.likelihood_layer import LikelihoodOutputs
from gpflux.losses import LikelihoodLoss

TEST_GPFLOW_LIKELIHOODS = [Bernoulli, Beta, Gaussian, Poisson]


def setup_gp_layer_and_data(num_inducing: int, **gp_layer_kwargs):
    input_dim = 5
    output_dim = 3
    num_data = 13
    data = make_data(input_dim, output_dim, num_data=num_data)

    kernel = construct_basic_kernel(Matern52(), output_dim)
    inducing_vars = construct_basic_inducing_variables(num_inducing, input_dim, output_dim)
    mean_function = Zero(output_dim)

    gp_layer = GPLayer(
        kernel, inducing_vars, num_data, mean_function=mean_function, **gp_layer_kwargs
    )
    return gp_layer, data


def make_data(input_dim: int, output_dim: int, num_data: int):
    lim = [0, 20]
    sigma = 0.1

    X = np.random.random(size=(num_data, input_dim)) * lim[1]
    cov = Matern52()(X) + np.eye(num_data) * sigma ** 2
    Y = [np.random.multivariate_normal(np.zeros(num_data), cov)[:, None] for _ in range(output_dim)]
    Y = np.hstack(Y)
    return X, Y  # TODO: improve this test; for most of the likelihoods, Y won't actually be valid


@pytest.mark.parametrize("GPflowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_call_shapes(GPflowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood_layer = LikelihoodLayer(GPflowLikelihood())

    # Run tests with gp layer outputting f_mean, f_var
    f_distribution = gp_layer(X)
    y_dist_params = likelihood_layer(f_distribution)

    assert y_dist_params.y_mean.shape == f_distribution.shape
    assert y_dist_params.y_var.shape == f_distribution.scale.diag.shape
    # The mean might not change but the covariance should
    assert f_distribution.variance().shape == y_dist_params.y_var.shape
    assert np.all(y_dist_params.y_var != f_distribution.variance())
    np.testing.assert_array_equal(y_dist_params.f_var, f_distribution.variance())
    np.testing.assert_array_equal(y_dist_params.f_mean, f_distribution.mean())


@pytest.mark.parametrize("GPflowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_likelihood_layer_losses(GPflowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood = GPflowLikelihood()
    likelihood_layer = LikelihoodLayer(likelihood)

    # Run tests with gp layer output as distribution
    f_distribution = gp_layer(X)

    _ = likelihood_layer(f_distribution)
    [keras_loss] = likelihood_layer.losses

    assert keras_loss == 0.0

    _ = likelihood_layer(f_distribution, targets=Y, training=True)
    [keras_loss] = likelihood_layer.losses

    f_mean = f_distribution.loc
    f_var = f_distribution.scale.diag ** 2
    expected_loss = np.mean(-likelihood.variational_expectations(f_mean, f_var, Y))

    np.testing.assert_almost_equal(keras_loss, expected_loss, decimal=5)


@pytest.mark.parametrize("GPflowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_likelihood_loss(GPflowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood = GPflowLikelihood()
    likelihood_loss = LikelihoodLoss(likelihood)

    # 1. Run tests with gp layer output as distribution
    f_distribution = gp_layer(X)
    f_mean = f_distribution.loc
    f_var = f_distribution.scale.diag

    expected_loss = np.mean(-likelihood.variational_expectations(f_mean, f_var, Y))
    np.testing.assert_almost_equal(likelihood_loss(Y, f_distribution), expected_loss, decimal=5)

    # 2. Run tests with gp_layer output coerced to sample
    f_sample = tf.convert_to_tensor(gp_layer(X))

    expected_loss = np.mean(-likelihood.log_prob(f_sample, Y))
    np.testing.assert_almost_equal(likelihood_loss(Y, f_sample), expected_loss, decimal=5)


def test_tensor_coercible():
    shape = [5, 3]
    f_mean = tf.random.normal(shape)
    f_var = tf.random.normal(shape)
    y_mean = tf.random.normal(shape)
    y_var = tf.random.normal(shape)
    tensor_coercible = LikelihoodOutputs(f_mean, f_var, y_mean, y_var)

    np.testing.assert_array_equal(f_mean, tf.convert_to_tensor(tensor_coercible))
