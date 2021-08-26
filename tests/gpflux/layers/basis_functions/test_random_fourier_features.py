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
from tensorflow.python.keras.testing_utils import layer_test
from tensorflow.python.keras.utils.kernelized_utils import inner_product

import gpflow

from gpflux.layers.basis_functions.random_fourier_features import (
    RFF_SUPPORTED_KERNELS,
    RandomFourierFeatures,
)


@pytest.fixture(name="n_dims", params=[1, 2, 3])
def _n_dims_fixture(request):
    return request.param


@pytest.fixture(name="lengthscale", params=[0.1, 1.0, 5.0])
def _lengthscale_fixture(request):
    return request.param


@pytest.fixture(name="batch_size", params=[1, 10])
def _batch_size_fixture(request):
    return request.param


@pytest.fixture(name="n_features", params=[1, 2, 30])
def _n_features_fixture(request):
    return request.param


@pytest.fixture(name="kernel_class", params=list(RFF_SUPPORTED_KERNELS))
def _kernel_class_fixture(request):
    return request.param


def test_throw_for_unsupported_kernel():
    kernel = gpflow.kernels.Constant()
    with pytest.raises(AssertionError) as excinfo:
        RandomFourierFeatures(kernel, 1)

    assert "Unsupported Kernel" in str(excinfo.value)


def test_fourier_features_can_approximate_kernel_multidim(kernel_class, lengthscale, n_dims):
    n_features = 10000
    x_rows = 20
    y_rows = 30
    # ARD
    lengthscales = np.random.rand((n_dims)) * lengthscale

    kernel = kernel_class(lengthscales=lengthscales)
    fourier_features = RandomFourierFeatures(kernel, n_features, dtype=tf.float64)

    x = tf.random.uniform((x_rows, n_dims), dtype=tf.float64)
    y = tf.random.uniform((y_rows, n_dims), dtype=tf.float64)

    u = fourier_features(x)
    v = fourier_features(y)
    approx_kernel_matrix = inner_product(u, v)

    actual_kernel_matrix = kernel.K(x, y)

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix, atol=0.05)


def test_fourier_features_shapes(n_features, n_dims, batch_size):
    kernel = gpflow.kernels.SquaredExponential()
    fourier_features = RandomFourierFeatures(kernel, n_features, dtype=tf.float64)
    features = fourier_features(tf.ones(shape=(batch_size, n_dims)))

    np.testing.assert_equal(features.shape, [batch_size, n_features])


def test_fourier_feature_layer_compute_covariance_of_inducing_variables(batch_size):
    """
    Ensure that the random fourier feature map can be used to approximate the covariance matrix
    between the inducing point vectors of a sparse GP, with the condition that the number of latent
    GP models is greater than one.
    """
    n_features = 10000

    kernel = gpflow.kernels.SquaredExponential()
    fourier_features = RandomFourierFeatures(kernel, n_features, dtype=tf.float64)

    x_new = tf.ones(shape=(2 * batch_size + 1, 1), dtype=tf.float64)

    u = fourier_features(x_new)
    approx_kernel_matrix = inner_product(u, u)

    actual_kernel_matrix = kernel.K(x_new, x_new)

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix, atol=0.05)


def test_keras_testing_util_layer_test_1D(kernel_class, batch_size, n_features):
    kernel = kernel_class()

    tf.keras.utils.get_custom_objects()["RandomFourierFeatures"] = RandomFourierFeatures
    layer_test(
        RandomFourierFeatures,
        kwargs={
            "kernel": kernel,
            "output_dim": n_features,
            "input_dim": 1,
            "dtype": "float64",
            "dynamic": True,
        },
        input_shape=(batch_size, 1),
        input_dtype="float64",
    )


def test_keras_testing_util_layer_test_multidim(kernel_class, batch_size, n_dims, n_features):
    kernel = kernel_class()

    tf.keras.utils.get_custom_objects()["RandomFourierFeatures"] = RandomFourierFeatures
    layer_test(
        RandomFourierFeatures,
        kwargs={
            "kernel": kernel,
            "output_dim": n_features,
            "input_dim": n_dims,
            "dtype": "float64",
            "dynamic": True,
        },
        input_shape=(batch_size, n_dims),
        input_dtype="float64",
    )
