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

from gpflux.layers.basis_functions.fourier_features.random import (
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
    RandomFourierFeaturesCosine,
)
from gpflux.layers.basis_functions.fourier_features.random.base import RFF_SUPPORTED_KERNELS
from tests.conftest import skip_serialization_tests


@pytest.fixture(name="n_dims", params=[1, 2, 3, 5, 10, 20])
def _n_dims_fixture(request):
    return request.param


@pytest.fixture(name="variance", params=[0.5, 1.0, 2.0])
def _variance_fixture(request):
    return request.param


@pytest.fixture(name="lengthscale", params=[0.1, 1.0, 5.0])
def _lengthscale_fixture(request):
    return request.param


@pytest.fixture(name="batch_size", params=[1, 10])
def _batch_size_fixture(request):
    return request.param


@pytest.fixture(name="n_components", params=[1, 2, 4, 20, 100])
def _n_features_fixture(request):
    return request.param


@pytest.fixture(name="kernel_cls", params=list(RFF_SUPPORTED_KERNELS))
def _kernel_cls_fixture(request):
    return request.param


@pytest.fixture(
    name="random_basis_func_cls",
    params=[RandomFourierFeatures, RandomFourierFeaturesCosine],
)
def _random_basis_func_cls_fixture(request):
    return request.param


@pytest.fixture(
    name="basis_func_cls",
    params=[RandomFourierFeatures, RandomFourierFeaturesCosine, OrthogonalRandomFeatures],
)
def _basis_func_cls_fixture(request):
    return request.param


def test_throw_for_unsupported_kernel(basis_func_cls):
    kernel = gpflow.kernels.Constant()
    with pytest.raises(AssertionError) as excinfo:
        basis_func_cls(kernel, n_components=1)
    assert "Unsupported Kernel" in str(excinfo.value)


def test_random_fourier_features_can_approximate_kernel_multidim(
    random_basis_func_cls, kernel_cls, variance, lengthscale, n_dims
):
    n_components = 40000

    x_rows = 20
    y_rows = 30
    # ARD
    lengthscales = np.random.rand((n_dims)) * lengthscale

    kernel = kernel_cls(variance=variance, lengthscales=lengthscales)
    fourier_features = random_basis_func_cls(kernel, n_components, dtype=tf.float64)

    x = tf.random.uniform((x_rows, n_dims), dtype=tf.float64)
    y = tf.random.uniform((y_rows, n_dims), dtype=tf.float64)

    u = fourier_features(x)
    v = fourier_features(y)
    approx_kernel_matrix = inner_product(u, v)

    actual_kernel_matrix = kernel.K(x, y)

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix, atol=5e-2)


def test_orthogonal_random_features_can_approximate_kernel_multidim(variance, lengthscale, n_dims):
    n_components = 20000

    x_rows = 20
    y_rows = 30
    # ARD
    lengthscales = np.random.rand((n_dims)) * lengthscale

    kernel = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=lengthscales)
    fourier_features = OrthogonalRandomFeatures(kernel, n_components, dtype=tf.float64)

    x = tf.random.uniform((x_rows, n_dims), dtype=tf.float64)
    y = tf.random.uniform((y_rows, n_dims), dtype=tf.float64)

    u = fourier_features(x)
    v = fourier_features(y)
    approx_kernel_matrix = inner_product(u, v)

    actual_kernel_matrix = kernel.K(x, y)

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix, atol=5e-2)


def test_random_fourier_feature_layer_compute_covariance_of_inducing_variables(
    basis_func_cls, batch_size
):
    """
    Ensure that the random fourier feature map can be used to approximate the covariance matrix
    between the inducing point vectors of a sparse GP, with the condition that the number of latent
    GP models is greater than one.
    """
    n_components = 10000

    kernel = gpflow.kernels.SquaredExponential()
    fourier_features = basis_func_cls(kernel, n_components, dtype=tf.float64)

    x_new = tf.ones(shape=(2 * batch_size + 1, 1), dtype=tf.float64)

    u = fourier_features(x_new)
    approx_kernel_matrix = inner_product(u, u)

    actual_kernel_matrix = kernel.K(x_new, x_new)

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix, atol=5e-2)


def test_fourier_features_shapes(basis_func_cls, n_components, n_dims, batch_size):
    input_shape = (batch_size, n_dims)
    kernel = gpflow.kernels.SquaredExponential()
    feature_functions = basis_func_cls(kernel, n_components, dtype=tf.float64)
    output_shape = feature_functions.compute_output_shape(input_shape)
    features = feature_functions(tf.ones(shape=input_shape))
    np.testing.assert_equal(features.shape, output_shape)


@skip_serialization_tests
def test_keras_testing_util_layer_test_1D(kernel_cls, batch_size, n_components):
    kernel = kernel_cls()

    tf.keras.utils.get_custom_objects()["RandomFourierFeatures"] = RandomFourierFeatures
    layer_test(
        RandomFourierFeatures,
        kwargs={
            "kernel": kernel,
            "n_components": n_components,
            "input_dim": 1,
            "dtype": "float64",
            "dynamic": True,
        },
        input_shape=(batch_size, 1),
        input_dtype="float64",
    )


@skip_serialization_tests
def test_keras_testing_util_layer_test_multidim(kernel_cls, batch_size, n_dims, n_components):
    kernel = kernel_cls()

    tf.keras.utils.get_custom_objects()["RandomFourierFeatures"] = RandomFourierFeatures
    layer_test(
        RandomFourierFeatures,
        kwargs={
            "kernel": kernel,
            "n_components": n_components,
            "input_dim": n_dims,
            "dtype": "float64",
            "dynamic": True,
        },
        input_shape=(batch_size, n_dims),
        input_dtype="float64",
    )
