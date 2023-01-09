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
from gpflow.kernels import SeparateIndependent, SharedIndependent

from gpflux import feature_decomposition_kernels
from gpflux.feature_decomposition_kernels.multioutput import (
    SeparateMultiOutputKernelWithFeatureDecomposition,
    SharedMultiOutputKernelWithFeatureDecomposition,
)
from gpflux.layers.basis_functions.fourier_features.multioutput.random import (
    MultiOutputOrthogonalRandomFeatures,
)
from gpflux.layers.basis_functions.fourier_features.random.orthogonal import ORF_SUPPORTED_KERNELS
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


@pytest.fixture(name="base_kernel_cls", params=list(ORF_SUPPORTED_KERNELS))
def _base_kernel_cls_fixture(request):
    return request.param


def test_orthogonal_fourier_features_can_approximate_multi_output_separate_kernel_multidim(
    base_kernel_cls, variance, lengthscale, n_dims
):
    n_components = 40000

    x_rows = 20
    y_rows = 30
    # ARD
    lengthscales = np.random.rand((n_dims)) * lengthscale

    print("size of sampled lengthscales")
    print(lengthscales.shape)

    base_kernel = base_kernel_cls(variance=variance, lengthscales=lengthscales)

    kernel = SeparateIndependent(kernels=[base_kernel, base_kernel])

    x = tf.random.uniform((x_rows, n_dims), dtype=tf.float64)
    y = tf.random.uniform((y_rows, n_dims), dtype=tf.float64)

    actual_kernel_matrix = kernel.K(x, y, full_output_cov=False).numpy()

    # fourier_features = random_basis_func_cls(kernel, n_components, dtype=tf.float64)
    fourier_features = MultiOutputOrthogonalRandomFeatures(kernel, n_components, dtype=tf.float64)

    feature_coefficients = np.ones((2, 2 * n_components, 1), dtype=np.float64)

    kernel = SeparateMultiOutputKernelWithFeatureDecomposition(
        kernel=None,
        feature_functions=fourier_features,
        feature_coefficients=feature_coefficients,
        output_dim=2,
    )

    approx_kernel_matrix = kernel(x, y).numpy()

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix, atol=5e-2)


def test_orthogonal_fourier_features_can_approximate_multi_output_shared_kernel_multidim(
    base_kernel_cls, variance, lengthscale, n_dims
):
    n_components = 40000

    x_rows = 20
    y_rows = 30
    # ARD
    lengthscales = np.random.rand((n_dims)) * lengthscale

    print("size of sampled lengthscales")
    print(lengthscales.shape)

    base_kernel = base_kernel_cls(variance=variance, lengthscales=lengthscales)

    kernel = SharedIndependent(kernel=base_kernel, output_dim=2)

    x = tf.random.uniform((x_rows, n_dims), dtype=tf.float64)
    y = tf.random.uniform((y_rows, n_dims), dtype=tf.float64)

    actual_kernel_matrix = kernel.K(x, y, full_output_cov=False).numpy()

    # fourier_features = random_basis_func_cls(kernel, n_components, dtype=tf.float64)
    fourier_features = MultiOutputOrthogonalRandomFeatures(kernel, n_components, dtype=tf.float64)

    feature_coefficients = np.ones((2, 2 * n_components, 1), dtype=np.float64)

    kernel = SharedMultiOutputKernelWithFeatureDecomposition(
        kernel=None,
        feature_functions=fourier_features,
        feature_coefficients=feature_coefficients,
        output_dim=2,
    )

    approx_kernel_matrix = kernel(x, y).numpy()

    np.testing.assert_allclose(approx_kernel_matrix, actual_kernel_matrix, atol=5e-2)


"""
#TODO -- have a look at what layer_test actually does
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

"""
