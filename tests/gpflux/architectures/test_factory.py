#
# Copyright (c) 2022 The GPflux Contributors.
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
from typing import Any, Type
import pytest

import numpy as np
import tensorflow_probability as tfp
from gpflow.kernels import (
    Matern12,
    Matern32,
    Matern52,
    SquaredExponential,
    Stationary,
)

from gpflux.architectures.config import (
    GaussianLikelihoodConfig,
    HeteroSkedasticLikelihoodConfig,
    HyperParametersConfig,
    ModelHyperParametersConfig,
    OrthogonalModelHyperparametersConfig,
    StudenttLikelihoodConfig,
)
from gpflux.architectures.factory import build_constant_input_dim_architecture, build_kernel


@pytest.fixture(
    name="kernel_type",
    params=[
        SquaredExponential,
        Matern12,
        Matern32,
        Matern52,
    ]
)
def _kernel_type(request: Any) -> Type[Stationary]:
    return request.param


@pytest.fixture(name="is_last_layer", params=[False, True])
def _is_last_layer(request: Any) -> bool:
    return request.param


@pytest.mark.parametrize("input_dim", [-1, 0])
def test_build_kernel__raises_with_invalid_input_dim(input_dim) -> None:
    with pytest.raises(AssertionError):
        build_kernel(input_dim, False, SquaredExponential)


def test_build_kernel(kernel_type: Type[Stationary], is_last_layer: bool) -> None:
    kernel = build_kernel(3, is_last_layer, kernel_type)

    assert isinstance(kernel, kernel_type)
    expected_variance = 1.0 if is_last_layer else 1e-6
    expected_lengthscales = [1.0] * 3

    np.testing.assert_allclose(kernel.variance.numpy(), expected_variance)
    np.testing.assert_allclose(kernel.lengthscales, expected_lengthscales)


MODEL_CONFIGS = [
    ModelHyperParametersConfig(
        num_layers=3,
        kernel=SquaredExponential,
        likelihood=GaussianLikelihoodConfig(noise_variance=1e-2),
        inner_layer_qsqrt_factor=1e-3,
        whiten=True,
        num_inducing=7,
    ),
    ModelHyperParametersConfig(
        num_layers=3,
        kernel=SquaredExponential,
        likelihood=StudenttLikelihoodConfig(df=3, scale=1e-2),
        inner_layer_qsqrt_factor=1e-3,
        whiten=True,
        num_inducing=7,
    ),
    pytest.param(
        ModelHyperParametersConfig(
            num_layers=3,
            kernel=SquaredExponential,
            likelihood=HeteroSkedasticLikelihoodConfig(),
            inner_layer_qsqrt_factor=1e-3,
            whiten=True,
            num_inducing=7,
        ),
        marks=pytest.mark.xfail
    ),
    pytest.param(
        ModelHyperParametersConfig(
            num_layers=3,
            kernel=SquaredExponential,
            likelihood=HeteroSkedasticLikelihoodConfig(
                distribution_class=tfp.distributions.StudentT
            ),
            inner_layer_qsqrt_factor=1e-3,
            whiten=True,
            num_inducing=7,
        ),
        marks=pytest.mark.xfail
    ),
    OrthogonalModelHyperparametersConfig(
        num_layers=3,
        kernel=SquaredExponential,
        likelihood=GaussianLikelihoodConfig(noise_variance=1e-2),
        inner_layer_qsqrt_factor=1e-3,
        whiten=True,
        num_inducing_u=7,
        num_inducing_v=7,
    ),
    OrthogonalModelHyperparametersConfig(
        num_layers=3,
        kernel=SquaredExponential,
        likelihood=StudenttLikelihoodConfig(df=3, scale=1e-2),
        inner_layer_qsqrt_factor=1e-3,
        whiten=True,
        num_inducing_u=7,
        num_inducing_v=7,
    ),
    pytest.param(
        OrthogonalModelHyperparametersConfig(
            num_layers=3,
            kernel=SquaredExponential,
            likelihood=HeteroSkedasticLikelihoodConfig(),
            inner_layer_qsqrt_factor=1e-3,
            whiten=True,
            num_inducing_u=7,
            num_inducing_v=7,
        ),
        marks=pytest.mark.xfail
    ),
    pytest.param(
        OrthogonalModelHyperparametersConfig(
            num_layers=3,
            kernel=SquaredExponential,
            likelihood=HeteroSkedasticLikelihoodConfig(
                distribution_class=tfp.distributions.StudentT
            ),
            inner_layer_qsqrt_factor=1e-3,
            whiten=True,
            num_inducing_u=7,
            num_inducing_v=7,
        ),
        marks=pytest.mark.xfail
    )
]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
@pytest.mark.parametrize("model_config", MODEL_CONFIGS)
def test_build_constant_input_dim_architecture__raises_on_incorrect_dtype(
    dtype, model_config: HyperParametersConfig
) -> None:
    X = np.random.randn(13, 2).astype(dtype)

    with pytest.raises(ValueError):
        build_constant_input_dim_architecture(model_config, X)


@pytest.mark.parametrize("model_config", MODEL_CONFIGS)
def test_build_constant_input_dim_architecture__does_not_smoke(
    model_config: HyperParametersConfig
) -> None:
    X = np.random.randn(13, 2)
    Y = np.random.randn(13, 1)

    model = build_constant_input_dim_architecture(model_config, X)
    model_train = model.as_training_model()
    model_train.compile("Adam")
    model_train.fit((X, Y), epochs=1)
    model_test = model.as_prediction_model()
    _ = model_test(X)
