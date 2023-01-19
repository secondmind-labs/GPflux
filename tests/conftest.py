from functools import partial

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

from gpflow.kernels import SquaredExponential

from gpflux.architectures import DistConfig, build_dist_deep_gp
from gpflux.models.dist_deep_gp import DistDeepGP

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


@pytest.fixture(name="ddgp_model")
def _ddgp_model(linear_dataset_querypoints: tf.Tensor) -> DistDeepGP:
    def _buil_ddgp() -> DistDeepGP:
        config = DistConfig(
            num_inducing=len(linear_dataset_querypoints),
            inner_layer_qsqrt_factor=1e-5,
            likelihood_noise_variance=1e-2,
            whiten=True,
        )

        return build_dist_deep_gp(X=linear_dataset_querypoints, num_layers=2, config=config)

    return partial(_buil_dgp)
