import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

from gpflow.kernels import SquaredExponential

# TODO: It would be great to make serialisation work in general. See:
# https://github.com/GPflow/GPflow/issues/1658
skip_serialization_tests = pytest.mark.skipif(
    Version(tf.__version__) >= Version("2.6"),
    reason="GPflow Parameter cannot be serialized in newer version of TensorFlow.",
)


@pytest.fixture
def test_data():
    x_dim, y_dim, w_dim = 2, 1, 2
    num_data = 31
    x_data = np.random.random((num_data, x_dim)) * 5
    w_data = np.random.random((num_data, w_dim))
    w_data[: (num_data // 2), :] = 0.2 * w_data[: (num_data // 2), :] + 5

    input_data = np.concatenate([x_data, w_data], axis=1)
    assert input_data.shape == (num_data, x_dim + w_dim)
    y_data = np.random.multivariate_normal(
        mean=np.zeros(num_data), cov=SquaredExponential(variance=0.1)(input_data), size=y_dim
    ).T
    assert y_data.shape == (num_data, y_dim)
    return x_data, y_data
