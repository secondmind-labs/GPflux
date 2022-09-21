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

import gpflow
from gpflow.config import default_float, default_jitter

from gpflux.feature_decomposition_kernels import (
    SeparateMultiOutputKernelWithFeatureDecomposition,
    SharedMultiOutputKernelWithFeatureDecomposition,
)
from gpflux.layers.basis_functions.fourier_features import MultiOutputRandomFourierFeaturesCosine
from gpflux.sampling.sample import Sample, efficient_sample


@pytest.fixture(name="base_kernel")
def _base_kernel_fixture():

    return gpflow.kernels.SquaredExponential()


def _get_shared_kernel(base_kernel):

    return gpflow.kernels.SharedIndependent(kernel=base_kernel, output_dim=2)


def _get_separate_kernel(base_kernel):

    return gpflow.kernels.SeparateIndependent(kernels=[base_kernel, base_kernel])


@pytest.fixture(name="shared_inducing_variable")
def _shared_inducing_variable_fixture():
    Z = np.linspace(-1, 1, 10).reshape(-1, 1)

    ind_var = gpflow.inducing_variables.InducingPoints(Z)

    return gpflow.inducing_variables.SharedIndependentInducingVariables(inducing_variable=ind_var)


@pytest.fixture(name="separate_inducing_variable")
def _separate_inducing_variable_fixture():
    Z = np.linspace(-1, 1, 10).reshape(-1, 1)

    ind_var = gpflow.inducing_variables.InducingPoints(Z)

    return gpflow.inducing_variables.SeparateIndependentInducingVariables(
        inducing_variable_list=[ind_var, ind_var]
    )


@pytest.fixture(name="whiten", params=[True])
def _whiten_fixture(request):
    return request.param


def _get_shared_qmu_qsqrt(kernel, inducing_variable):
    """Returns q_mu and q_sqrt for a kernel and inducing_variable"""
    Z = inducing_variable.inducing_variable.Z.numpy()
    Kzz = kernel(Z, full_cov=True, full_output_cov=False).numpy()

    q_sqrt = np.linalg.cholesky(Kzz) + default_jitter() * np.eye(Z.shape[0])[None, ...]
    q_mu = q_sqrt @ np.random.randn(2, Z.shape[0], 1)

    return np.transpose(q_mu[..., 0]), q_sqrt


def _get_separate_qmu_qsqrt(kernel, inducing_variable):
    """Returns q_mu and q_sqrt for a kernel and inducing_variable"""
    Z = inducing_variable.inducing_variable_list[0].Z.numpy()
    Kzz = kernel(Z, full_cov=True, full_output_cov=False).numpy()

    q_sqrt = np.linalg.cholesky(Kzz) + default_jitter() * np.eye(Z.shape[0])[None, ...]
    q_mu = q_sqrt @ np.random.randn(2, Z.shape[0], 1)

    return np.transpose(q_mu[..., 0]), q_sqrt


def test_shared_wilson_efficient_sample(base_kernel, shared_inducing_variable, whiten):
    """Smoke and consistency test for efficient sampling using Wilson"""
    kernel = _get_shared_kernel(base_kernel)

    eigenfunctions = MultiOutputRandomFourierFeaturesCosine(kernel, 100, dtype=default_float())
    eigenvalues = np.ones((2, 100, 1), dtype=default_float())
    # To apply Wilson sampling we require the features and eigenvalues of the kernel
    kernel2 = SharedMultiOutputKernelWithFeatureDecomposition(kernel, eigenfunctions, eigenvalues)
    q_mu, q_sqrt = _get_shared_qmu_qsqrt(kernel, shared_inducing_variable)

    sample_func = efficient_sample(
        shared_inducing_variable,
        kernel2,
        tf.convert_to_tensor(q_mu),
        q_sqrt=1e-3 * tf.convert_to_tensor(q_sqrt),
        whiten=whiten,
    )

    X = np.linspace(-1, 0, 100).reshape(-1, 1)
    # Check for consistency - i.e. evaluating the sample at the
    # same locations (X) returns the same value
    np.testing.assert_array_almost_equal(
        sample_func(X),
        sample_func(X),
    )


def test_separate_wilson_efficient_sample(base_kernel, separate_inducing_variable, whiten):
    """Smoke and consistency test for efficient sampling using Wilson"""
    kernel = _get_separate_kernel(base_kernel)

    eigenfunctions = MultiOutputRandomFourierFeaturesCosine(kernel, 100, dtype=default_float())
    eigenvalues = np.ones((2, 100, 1), dtype=default_float())
    # To apply Wilson sampling we require the features and eigenvalues of the kernel
    kernel2 = SeparateMultiOutputKernelWithFeatureDecomposition(kernel, eigenfunctions, eigenvalues)
    q_mu, q_sqrt = _get_separate_qmu_qsqrt(kernel, separate_inducing_variable)

    sample_func = efficient_sample(
        separate_inducing_variable,
        kernel2,
        tf.convert_to_tensor(q_mu),
        q_sqrt=1e-3 * tf.convert_to_tensor(q_sqrt),
        whiten=whiten,
    )

    X = np.linspace(-1, 0, 100).reshape(-1, 1)
    # Check for consistency - i.e. evaluating the sample at the
    # same locations (X) returns the same value
    np.testing.assert_array_almost_equal(
        sample_func(X),
        sample_func(X),
    )


"""
class SampleMock(Sample):
    def __init__(self, a):
        self.a = a

    def __call__(self, X):
        return self.a * X


def test_adding_samples():
    X = np.random.randn(100, 2)

    sample1 = SampleMock(1.0)
    sample2 = SampleMock(2.0)
    sample3 = sample1 + sample2
    np.testing.assert_array_almost_equal(sample3(X), sample1(X) + sample2(X))


def test_adding_sample_and_mean_function():
    X = np.random.randn(100, 2)

    mean_function = gpflow.mean_functions.Identity()
    sample = SampleMock(3.0)

    sample_and_mean_function = sample + mean_function

    np.testing.assert_array_almost_equal(sample_and_mean_function(X), sample(X) + mean_function(X))
"""
