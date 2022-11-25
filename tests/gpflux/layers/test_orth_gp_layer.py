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
import tensorflow_probability as tfp

from gpflow.kernels import RBF
from gpflow.mean_functions import Zero

from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.layers import OrthGPLayer
from gpflux.types import unwrap_dist


def setup_orth_gp_layer_and_data(num_inducing_u: int, num_inducing_v: int, **gp_layer_kwargs):
    input_dim = 30
    output_dim = 5
    num_data = 100
    data = make_data(input_dim, output_dim, num_data=num_data)

    kernel = construct_basic_kernel(RBF(), output_dim)
    inducing_vars_u = construct_basic_inducing_variables(num_inducing_u, input_dim, output_dim)
    inducing_vars_v = construct_basic_inducing_variables(num_inducing_v, input_dim, output_dim)
    mean_function = Zero(output_dim)

    gp_layer = OrthGPLayer(
        kernel,
        inducing_vars_u,
        inducing_vars_v,
        num_data,
        mean_function=mean_function,
        **gp_layer_kwargs
    )
    return gp_layer, data


def make_data(input_dim: int, output_dim: int, num_data: int):
    lim = [0, 20]
    sigma = 0.1

    X = np.random.random(size=(num_data, input_dim)) * lim[1]
    cov = RBF().K(X) + np.eye(num_data) * sigma ** 2
    Y = [np.random.multivariate_normal(np.zeros(num_data), cov)[:, None] for _ in range(output_dim)]
    Y = np.hstack(Y)
    return X, Y


def test_build():
    num_inducing_u = 5
    num_inducing_v = 10

    gp_layer, (X, Y) = setup_orth_gp_layer_and_data(num_inducing_u, num_inducing_v)
    output_dim = Y.shape[-1]

    gp_layer.build(X.shape)
    assert gp_layer.q_mu_u.shape == (num_inducing_u, output_dim)
    assert gp_layer.q_sqrt_u.shape == (output_dim, num_inducing_u, num_inducing_u)

    assert gp_layer.q_mu_v.shape == (num_inducing_v, output_dim)
    assert gp_layer.q_sqrt_v.shape == (output_dim, num_inducing_v, num_inducing_v)


def test_kl_change_q_mean():
    gp_layer, (X, Y) = setup_orth_gp_layer_and_data(num_inducing_u=5, num_inducing_v=5)

    gp_layer.build(X.shape)
    assert gp_layer.prior_kl() > 0.0

    q_mu_u_random = np.random.random(size=gp_layer.q_mu_u.shape)
    gp_layer.q_mu_u.assign(q_mu_u_random)

    q_mu_v_random = np.random.random(size=gp_layer.q_mu_v.shape)
    gp_layer.q_mu_v.assign(q_mu_v_random)

    assert gp_layer.prior_kl() > 0.0


def test_kl_change_q_sqrt():
    gp_layer, (X, Y) = setup_orth_gp_layer_and_data(num_inducing_u=5, num_inducing_v=5)

    gp_layer.build(X.shape)
    assert gp_layer.prior_kl() > 0.0

    gp_layer.q_sqrt_u.assign(gp_layer.q_sqrt_u * 3)
    gp_layer.q_sqrt_v.assign(gp_layer.q_sqrt_v * 3)
    assert gp_layer.prior_kl() > 0.0


def test_call_shapes():
    gp_layer, (X, Y) = setup_orth_gp_layer_and_data(num_inducing_u=5, num_inducing_v=5)
    gp_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]

    samples = tf.convert_to_tensor(gp_layer(X, training=False))
    assert samples.shape == (batch_size, output_dim)

    assert not gp_layer.full_cov and not gp_layer.full_output_cov

    distribution = gp_layer(X, training=False)
    assert isinstance(unwrap_dist(distribution), tfp.distributions.MultivariateNormalDiag)
    assert distribution.shape == (batch_size, output_dim)

    gp_layer.full_cov = True
    distribution = gp_layer(X, training=False)
    assert isinstance(unwrap_dist(distribution), tfp.distributions.MultivariateNormalTriL)
    assert distribution.shape == (batch_size, output_dim)
    assert distribution.covariance().shape == (output_dim, batch_size, batch_size)

    gp_layer.full_output_cov = True
    gp_layer.full_cov = False
    distribution = gp_layer(X, training=False)
    assert isinstance(unwrap_dist(distribution), tfp.distributions.MultivariateNormalTriL)
    assert distribution.shape == (batch_size, output_dim)
    assert distribution.covariance().shape == (batch_size, output_dim, output_dim)

    gp_layer.full_output_cov = True
    gp_layer.full_cov = True
    with pytest.raises(NotImplementedError):
        gp_layer(X)


def test_call_shapes_num_samples():
    num_samples = 10
    gp_layer, (X, Y) = setup_orth_gp_layer_and_data(
        num_inducing_u=5, num_inducing_v=5, num_samples=num_samples
    )
    gp_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]

    samples = tf.convert_to_tensor(gp_layer(X, training=False))
    assert samples.shape == (num_samples, batch_size, output_dim)

    gp_layer.full_cov = True
    samples = tf.convert_to_tensor(gp_layer(X, training=False))
    assert samples.shape == (num_samples, batch_size, output_dim)


def test_predict_shapes():
    gp_layer, (X, Y) = setup_orth_gp_layer_and_data(num_inducing_u=5, num_inducing_v=5)
    gp_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]

    mean, cov = gp_layer.predict(X)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim)

    mean, cov = gp_layer.predict(X, full_cov=True)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (output_dim, batch_size, batch_size)

    mean, cov = gp_layer.predict(X, full_output_cov=True)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim, output_dim)


def test_losses_are_added():
    gp_layer, (X, Y) = setup_orth_gp_layer_and_data(num_inducing_u=5, num_inducing_v=5)
    gp_layer.build(X.shape)

    # to make KL non-zero
    gp_layer.q_mu_u.assign(tf.ones_like(gp_layer.q_mu_u) * 3)
    assert gp_layer.prior_kl() > 0.0

    assert len(gp_layer.losses) == 0

    _ = gp_layer(X, training=True)
    assert gp_layer.losses == [gp_layer.prior_kl() / gp_layer.num_data]

    # Check loss is 0 when training is False
    _ = gp_layer(X, training=False)
    assert gp_layer.losses == [tf.zeros_like(gp_layer.losses[0])]

    # Check calling multiple times only adds one loss
    _ = gp_layer(X, training=True)
    _ = gp_layer(X, training=True)
    assert len(gp_layer.losses) == 1
    assert gp_layer.losses == [gp_layer.prior_kl() / gp_layer.num_data]


if __name__ == "__main__":
    test_call_shapes()
