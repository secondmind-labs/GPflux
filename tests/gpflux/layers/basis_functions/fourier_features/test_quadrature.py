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
from gpflow.quadrature.gauss_hermite import NDiagGHQuadrature
from gpflow.utilities.ops import difference_matrix

from gpflux.layers.basis_functions.fourier_features.quadrature import QuadratureFourierFeatures
from gpflux.layers.basis_functions.fourier_features.quadrature.gaussian import QFF_SUPPORTED_KERNELS
from tests.conftest import skip_serialization_tests


@pytest.fixture(name="n_dims", params=[1, 2, 3])
def _n_dims_fixture(request):
    return request.param


@pytest.fixture(name="n_components", params=[1, 2, 4, 20])
def _n_features_fixture(request):
    return request.param


@pytest.fixture(name="variance", params=[0.5, 1.0, 2.0])
def _variance_fixture(request):
    return request.param


@pytest.fixture(name="lengthscale", params=[0.75, 1.0, 5.0])
def _lengthscale_fixture(request):
    return request.param


@pytest.fixture(name="batch_size", params=[1, 10])
def _batch_size_fixture(request):
    return request.param


@pytest.fixture(name="kernel_cls", params=list(QFF_SUPPORTED_KERNELS))
def _kernel_cls_fixture(request):
    return request.param


def test_throw_for_unsupported_kernel():
    kernel = gpflow.kernels.Constant()
    with pytest.raises(AssertionError) as excinfo:
        QuadratureFourierFeatures(kernel, n_components=1)
    assert "Unsupported Kernel" in str(excinfo.value)


def test_quadrature_fourier_features_can_approximate_kernel_multidim(
    kernel_cls, variance, lengthscale, n_dims
):

    """
    Compare finite feature approximation to analytical kernel expression.
    Approximation only holds for large enough lengthscales, as explained in TODO: notebook.
    """
    n_components = 128

    x_rows = 20
    y_rows = 30
    # ARD

    # small lengthscales can lead to large errors in this approximation
    lengthscales = np.random.uniform(low=0.75, size=n_dims) * lengthscale

    kernel = kernel_cls(variance=variance, lengthscales=lengthscales)
    fourier_features = QuadratureFourierFeatures(kernel, n_components, dtype=tf.float64)

    x = tf.random.uniform((x_rows, n_dims), dtype=tf.float64)
    y = tf.random.uniform((y_rows, n_dims), dtype=tf.float64)

    u = fourier_features(x)
    v = fourier_features(y)
    approx_kernel_matrix = inner_product(u, v)

    actual_kernel_matrix = kernel.K(x, y)

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix)


def test_feature_map_decomposition(kernel_cls, variance, lengthscale, n_dims, n_components):
    """
    Verify that the inner product of the feature map yields exactly the same
    result as that of the direct Gauss-Hermite quadrature scheme.
    """
    x_rows = 20
    y_rows = 30
    # ARD
    lengthscales = np.random.rand(n_dims) * lengthscale

    kernel = kernel_cls(variance=variance, lengthscales=lengthscales)
    fourier_features = QuadratureFourierFeatures(kernel, n_components, dtype=tf.float64)

    x = tf.random.uniform((x_rows, n_dims), dtype=tf.float64)
    y = tf.random.uniform((y_rows, n_dims), dtype=tf.float64)

    u = fourier_features(x)
    v = fourier_features(y)
    K_decomp = inner_product(u, v)

    x_scaled = tf.truediv(x, lengthscales)
    y_scaled = tf.truediv(y, lengthscales)
    r = difference_matrix(x_scaled, y_scaled)

    def eigen_func(w):
        return variance * tf.cos(tf.matmul(r, w, transpose_b=True))

    quadrature = NDiagGHQuadrature(dim=n_dims, n_gh=n_components)
    K_quadrature = quadrature(
        eigen_func,
        mean=tf.zeros((1, 1, n_dims), dtype=tf.float64),
        var=tf.ones((1, 1, n_dims), dtype=tf.float64),
    )
    K_quadrature = tf.squeeze(K_quadrature, axis=-1)

    np.testing.assert_allclose(K_decomp, K_quadrature, atol=1e-15)


def test_fourier_features_shapes(n_components, n_dims, batch_size):
    input_shape = (batch_size, n_dims)
    kernel = gpflow.kernels.SquaredExponential()
    feature_functions = QuadratureFourierFeatures(kernel, n_components, dtype=tf.float64)
    output_shape = feature_functions.compute_output_shape(input_shape)
    output_dim = output_shape[-1]
    assert output_dim == 2 * n_components ** n_dims
    features = feature_functions(tf.ones(shape=input_shape))
    np.testing.assert_equal(features.shape, output_shape)


@skip_serialization_tests
def test_keras_testing_util_layer_test_1D(kernel_cls, batch_size, n_components):
    kernel = kernel_cls()

    tf.keras.utils.get_custom_objects()["QuadratureFourierFeatures"] = QuadratureFourierFeatures
    layer_test(
        QuadratureFourierFeatures,
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

    tf.keras.utils.get_custom_objects()["QuadratureFourierFeatures"] = QuadratureFourierFeatures
    layer_test(
        QuadratureFourierFeatures,
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
