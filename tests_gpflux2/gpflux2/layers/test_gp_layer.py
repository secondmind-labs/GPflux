import numpy as np
import pytest

from gpflow.kernels import SeparateIndependent, RBF, Matern12, Matern32, Matern52
from gpflow.inducing_variables import (
    SeparateIndependentInducingVariables,
    InducingPoints,
)
from gpflow.mean_functions import Zero

from gpflux2.layers import GPLayer
from gpflux2.initializers import ZeroInitializer
from gpflux2.helpers import construct_basic_kernel, construct_basic_inducing_variables


def setup_data_and_gplayer(num_inducing: int, **kwargs):
    input_dim = 30
    output_dim = 5
    data = make_data(input_dim, output_dim, num_data=100)

    kernel = construct_basic_kernel(RBF(), output_dim)
    inducing_vars = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim
    )
    initializer = ZeroInitializer()
    mean_function = Zero(output_dim)

    gp_layer = GPLayer(
        kernel,
        inducing_vars,
        initializer=initializer,
        mean_function=mean_function,
        **kwargs
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
    num_inducing = 30
    num_data = 300

    gp_layer, (X, Y) = setup_data_and_gplayer(num_inducing)
    output_dim = Y.shape[-1]

    gp_layer.build(X.shape)
    assert gp_layer.q_mu.shape == (num_inducing, output_dim)
    assert gp_layer.q_sqrt.shape == (output_dim, num_inducing, num_inducing)


def test_kl_change_q_mean():
    gp_layer, (X, Y) = setup_data_and_gplayer(num_inducing=30)

    gp_layer.build(X.shape)
    assert gp_layer.prior_kl() == 0.0

    q_mu_random = np.random.random(size=gp_layer.q_mu.shape)
    gp_layer.q_mu.assign(q_mu_random)
    assert gp_layer.prior_kl() > 0.0


def test_kl_change_q_sqrt():
    gp_layer, (X, Y) = setup_data_and_gplayer(num_inducing=30)

    gp_layer.build(X.shape)

    assert gp_layer.prior_kl() == 0.0

    gp_layer.q_sqrt.assign(gp_layer.q_sqrt * 3)
    assert gp_layer.prior_kl() > 0.0


def test_call_shapes():
    gp_layer, (X, Y) = setup_data_and_gplayer(num_inducing=30)
    gp_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]
    num_samples = 10

    samples = gp_layer.call(X, training=False)
    assert samples.shape == (batch_size, output_dim)

    gp_layer.use_samples = False
    assert not gp_layer.full_cov and not gp_layer.full_output_cov

    mean, cov = gp_layer.call(X, training=False)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim)

    gp_layer.full_cov = True
    mean, cov = gp_layer.call(X, training=False)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (output_dim, batch_size, batch_size)

    gp_layer.full_output_cov = True
    gp_layer.full_cov = False
    mean, cov = gp_layer.call(X, training=False)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim, output_dim)

    gp_layer.full_output_cov = True
    gp_layer.full_cov = True
    with pytest.raises(NotImplementedError):
        gp_layer.call(X)


def test_predict_shapes():
    gp_layer, (X, Y) = setup_data_and_gplayer(num_inducing=30)

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


if __name__ == "__main__":
    test_call_shapes()
