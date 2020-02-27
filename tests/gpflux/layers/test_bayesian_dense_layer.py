import tensorflow as tf
import numpy as np
import pytest

from gpflow.kernels import RBF

from gpflux.layers import BayesianDenseLayer


is_mean_field_values = [True, False]


def setup_gp_layer_and_data(is_mean_field: bool):
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

    bnn_layer, (X, Y) = setup_gp_layer_and_data(is_mean_field)
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
    assert bnn_layer.w_mu.shape == (dim, 1)
    if not is_mean_field:
        assert bnn_layer.w_sqrt.shape == (1, dim, dim)
    else:
        assert bnn_layer.w_sqrt.shape == (dim, 1)
    assert bnn_layer._initialized
