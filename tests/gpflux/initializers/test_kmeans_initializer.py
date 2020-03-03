import numpy as np

from gpflow.inducing_variables import InducingPoints

from gpflux.initializers import KmeansInitializer


class Data:
    input_dim = 1
    X = np.array([-0.5, 0.5, 10.0, 20.0]).reshape(-1, input_dim)


def get_initialized_Z(initializer, num_inducing):
    iv = InducingPoints(np.zeros((num_inducing, Data.input_dim)))
    initializer.init_single_inducing_variable(iv)
    return iv.Z.numpy()


def test_kmeansinitializer_kmeans():
    num_inducing = 3
    assert len(Data.X) > num_inducing

    initializer = KmeansInitializer(Data.X, num_inducing)
    Z = get_initialized_Z(initializer, num_inducing)

    assert sorted(Z) in ([-0.5, 0.5, 15.0], [0.0, 10.0, 20.0])


def test_kmeansinitializer_input():
    num_inducing = 4
    assert len(Data.X) == num_inducing

    initializer = KmeansInitializer(Data.X, num_inducing)
    Z = get_initialized_Z(initializer, num_inducing)

    np.testing.assert_equal(Z, Data.X)


def test_kmeansinitializer_padding():
    num_inducing = 6
    num_data = len(Data.X)
    assert num_data < num_inducing

    initializer = KmeansInitializer(Data.X, num_inducing)
    Z = get_initialized_Z(initializer, num_inducing)

    np.testing.assert_equal(Z[:num_data], Data.X)
    extra_rows = Z[num_data:]
    assert len(extra_rows) == num_inducing - num_data == 2
    assert np.any(extra_rows[0] != extra_rows[1])
