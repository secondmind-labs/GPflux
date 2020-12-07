import numpy as np
import pytest
import tensorflow as tf

from gpflow.inducing_variables import InducingPoints

from gpflux.initializers import FeedForwardInitializer


@pytest.mark.parametrize("num_inducing, num_data", [(7, 5), (5, 7), (7, 7)])
@pytest.mark.parametrize("input_dim", [1, 2])
def test_feedforwardinitializer_init(num_inducing, num_data, input_dim):
    X = np.arange(num_data * input_dim, dtype=float).reshape(num_data, input_dim)
    iv = InducingPoints(np.zeros((num_inducing, input_dim)))

    initializer = FeedForwardInitializer()
    initializer.init_single_inducing_variable(iv, tf.convert_to_tensor(X))

    Z_value = iv.Z.numpy()

    assert Z_value.shape == (num_inducing, input_dim)
    Z_values = Z_value.flatten()
    X_values = np.arange(num_data * input_dim, dtype=float)
    if num_inducing == num_data:
        assert set(Z_values) == set(X_values)
    elif num_inducing < num_data:
        assert set(Z_values) < set(X_values)
    else:
        assert set(Z_values) > set(X_values)
