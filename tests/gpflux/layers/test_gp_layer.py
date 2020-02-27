import tensorflow as tf
import numpy as np
import pytest

from gpflow.kernels import SeparateIndependent, RBF, Matern12, Matern32, Matern52
from gpflow.inducing_variables import (
    SeparateIndependentInducingVariables,
    InducingPoints,
)
from gpflow.mean_functions import Zero

from gpflux.exceptions import GPInitializationError
from gpflux.initializers import ZeroOneInitializer
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.layers import GPLayer


def setup_gp_layer_and_data(num_inducing: int, **gp_layer_kwargs):
    input_dim = 30
    output_dim = 5
    num_data = 100
    data = make_data(input_dim, output_dim, num_data=num_data)

    kernel = construct_basic_kernel(RBF(), output_dim)
    inducing_vars = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim
    )
    initializer = ZeroOneInitializer()
    mean_function = Zero(output_dim)

    gp_layer = GPLayer(
        kernel,
        inducing_vars,
        num_data,
        initializer=initializer,
        mean_function=mean_function,
        **gp_layer_kwargs
    )
    return gp_layer, data


def make_data(input_dim: int, output_dim: int, num_data: int):
    lim = [0, 20]
    sigma = 0.1

    X = np.random.random(size=(num_data, input_dim)) * lim[1]
    cov = RBF().K(X) + np.eye(num_data) * sigma ** 2
    Y = [
        np.random.multivariate_normal(np.zeros(num_data), cov)[:, None]
        for _ in range(output_dim)
    ]
    Y = np.hstack(Y)
    return X, Y


def test_build():
    num_inducing = 5

    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing)
    output_dim = Y.shape[-1]
    assert not gp_layer._initialized

    gp_layer.build(X.shape)
    assert gp_layer.q_mu.shape == (num_inducing, output_dim)
    assert gp_layer.q_sqrt.shape == (output_dim, num_inducing, num_inducing)
    assert gp_layer._initialized


def test_kl_change_q_mean():
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)

    gp_layer.build(X.shape)
    assert gp_layer.prior_kl() == 0.0

    q_mu_random = np.random.random(size=gp_layer.q_mu.shape)
    gp_layer.q_mu.assign(q_mu_random)
    assert gp_layer.prior_kl() > 0.0


def test_kl_change_q_sqrt():
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)

    gp_layer.build(X.shape)
    assert gp_layer.prior_kl() == 0.0

    gp_layer.q_sqrt.assign(gp_layer.q_sqrt * 3)
    assert gp_layer.prior_kl() > 0.0


def test_call_shapes():
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    gp_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]

    samples = gp_layer(X, training=False)
    assert samples.shape == (batch_size, output_dim)

    gp_layer.returns_samples = False
    assert not gp_layer.full_cov and not gp_layer.full_output_cov

    mean, cov = gp_layer(X, training=False)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim)

    gp_layer.full_cov = True
    mean, cov = gp_layer(X, training=False)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (output_dim, batch_size, batch_size)

    gp_layer.full_output_cov = True
    gp_layer.full_cov = False
    mean, cov = gp_layer(X, training=False)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim, output_dim)

    gp_layer.full_output_cov = True
    gp_layer.full_cov = True
    with pytest.raises(NotImplementedError):
        gp_layer(X)


def test_predict_shapes():
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    gp_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]
    num_samples = 10

    samples, mean, cov = gp_layer.predict(X)
    assert samples.shape == (batch_size, output_dim)

    samples, mean, cov = gp_layer.predict(X, num_samples=num_samples)
    assert samples.shape == (num_samples, batch_size, output_dim)

    samples, mean, cov = gp_layer.predict(X)
    assert samples.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim)

    samples, mean, cov = gp_layer.predict(X, full_cov=True)
    assert samples.shape == (batch_size, output_dim)
    assert cov.shape == (output_dim, batch_size, batch_size)

    samples, mean, cov = gp_layer.predict(X, full_output_cov=True)
    assert samples.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim, output_dim)

    with pytest.raises(NotImplementedError):
        gp_layer.predict(X, full_cov=True, full_output_cov=True)


def test_losses_are_added():
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    gp_layer.build(X.shape)

    # to make KL non-zero
    gp_layer.q_mu.assign(tf.ones_like(gp_layer.q_mu) * 3)
    assert gp_layer.prior_kl() > 0.0

    assert len(gp_layer.losses) == 0

    outputs = gp_layer(X, training=True)
    assert gp_layer.losses == [gp_layer.prior_kl() / gp_layer.num_data]

    # Check loss is 0 when training is False
    outputs = gp_layer(X, training=False)
    assert gp_layer.losses == [tf.zeros_like(gp_layer.losses[0])]

    # Check calling multiple times only adds one loss
    outputs = gp_layer(X, training=True)
    outputs = gp_layer(X, training=True)
    assert len(gp_layer.losses) == 1
    assert gp_layer.losses == [gp_layer.prior_kl() / gp_layer.num_data]


def test_initialization():
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    assert not gp_layer._initialized

    f_pred = gp_layer(X)
    assert gp_layer._initialized

    with pytest.raises(GPInitializationError) as e:
        gp_layer.initialize_inducing_variables()


if __name__ == "__main__":
    test_call_shapes()
