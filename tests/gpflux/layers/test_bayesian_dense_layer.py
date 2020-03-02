import tensorflow as tf
import numpy as np
import pytest

from gpflow.kernels import RBF

from gpflux.layers import BayesianDenseLayer


is_mean_field_values = [True, False]


def setup_bnn_layer_and_data(is_mean_field: bool):
    input_dim = 12
    output_dim = 4
    num_data = 100
    w_mu = np.zeros(((input_dim + 1) * output_dim,))
    w_sqrt = np.eye((input_dim + 1) * output_dim) if not is_mean_field \
        else np.ones(((input_dim + 1) * output_dim,))
    activity_function = tf.nn.relu
    data = make_data(input_dim, output_dim, num_data=num_data)

    bnn_layer = BayesianDenseLayer(
        input_dim,
        output_dim,
        num_data,
        w_mu=w_mu,
        w_sqrt=w_sqrt,
        activity_function=activity_function,
        is_mean_field=is_mean_field
    )
    return bnn_layer, data


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


@pytest.mark.parametrize('is_mean_field', is_mean_field_values)
def test_build(is_mean_field):
    bnn_layer, (X, Y) = setup_bnn_layer_and_data(is_mean_field)
    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]
    num_data = X.shape[0]
    dim = (input_dim + 1) * output_dim
    assert not bnn_layer._initialized

    bnn_layer.build(X.shape)
    assert bnn_layer.input_dim == input_dim
    assert bnn_layer.output_dim == output_dim
    assert bnn_layer.num_data == num_data
    assert bnn_layer.is_mean_field == is_mean_field
    assert bnn_layer.dim == dim
    assert bnn_layer.full_output_cov == bnn_layer.full_cov == False
    assert bnn_layer.w_mu.shape == (dim,)
    if not is_mean_field:
        assert bnn_layer.w_sqrt.shape == (dim, dim)
    else:
        assert bnn_layer.w_sqrt.shape == (dim,)
    assert bnn_layer._initialized


@pytest.mark.parametrize('is_mean_field', is_mean_field_values)
def test_kl_change_w_mean(is_mean_field):
    bnn_layer, (X, Y) = setup_bnn_layer_and_data(is_mean_field)

    bnn_layer.build(X.shape)
    assert bnn_layer.prior_kl() == 0.0

    w_mu_random = np.random.random(size=bnn_layer.w_mu.shape)
    bnn_layer.w_mu.assign(w_mu_random)
    assert bnn_layer.prior_kl() > 0.0


@pytest.mark.parametrize('is_mean_field', is_mean_field_values)
def test_kl_change_w_sqrt(is_mean_field):
    bnn_layer, (X, Y) = setup_bnn_layer_and_data(is_mean_field)

    bnn_layer.build(X.shape)
    assert bnn_layer.prior_kl() == 0.0

    bnn_layer.w_sqrt.assign(bnn_layer.w_sqrt * 3)
    assert bnn_layer.prior_kl() > 0.0


@pytest.mark.parametrize('is_mean_field', is_mean_field_values)
def test_call_shapes(is_mean_field):
    bnn_layer, (X, Y) = setup_bnn_layer_and_data(is_mean_field)
    bnn_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]

    samples = bnn_layer(X, training=False)
    assert samples.shape == (batch_size, output_dim)

    bnn_layer.returns_samples = False
    assert not bnn_layer.full_cov and not bnn_layer.full_output_cov

    mean, cov = bnn_layer(X, training=False)
    assert mean.shape == (batch_size, output_dim)
    assert cov.shape == (batch_size, output_dim)


@pytest.mark.parametrize('is_mean_field', is_mean_field_values)
def test_predict_shapes(is_mean_field):
    bnn_layer, (X, Y) = setup_bnn_layer_and_data(is_mean_field)
    bnn_layer.build(X.shape)

    output_dim = Y.shape[-1]
    batch_size = X.shape[0]
    num_samples = 10

    samples, mean, cov = bnn_layer.predict(X)
    assert samples.shape == mean.shape == cov.shape == (batch_size, output_dim)

    samples, mean, cov = bnn_layer.predict(X, num_samples=num_samples)
    assert samples.shape == mean.shape == cov.shape == (num_samples, batch_size, output_dim)


@pytest.mark.parametrize('is_mean_field', is_mean_field_values)
def test_losses_are_added(is_mean_field):
    bnn_layer, (X, Y) = setup_bnn_layer_and_data(is_mean_field)
    bnn_layer.build(X.shape)

    # to make KL non-zero
    bnn_layer.w_mu.assign(tf.ones_like(bnn_layer.w_mu) * 3)
    assert bnn_layer.prior_kl() > 0.0

    assert len(bnn_layer.losses) == 0

    outputs = bnn_layer(X, training=True)
    assert bnn_layer.losses == [bnn_layer.temperature * bnn_layer.prior_kl() / bnn_layer.num_data]

    # Check loss is 0 when training is False
    outputs = bnn_layer(X, training=False)
    assert bnn_layer.losses == [tf.zeros_like(bnn_layer.losses[0])]

    # Check calling multiple times only adds one loss
    outputs = bnn_layer(X, training=True)
    outputs = bnn_layer(X, training=True)
    assert len(bnn_layer.losses) == 1
    assert bnn_layer.losses == [bnn_layer.temperature * bnn_layer.prior_kl() / bnn_layer.num_data]


@pytest.mark.parametrize('is_mean_field', is_mean_field_values)
def test_initialization(is_mean_field):
    bnn_layer, (X, Y) = setup_bnn_layer_and_data(is_mean_field)
    assert not bnn_layer._initialized

    f_pred = bnn_layer(X)
    assert bnn_layer._initialized
