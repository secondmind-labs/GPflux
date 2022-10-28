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
from dataclasses import dataclass

import numpy as np
import pytest

import gpflow
from gpflow import mean_functions
from gpflow.inducing_variables import (
    InducingPoints,
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    Matern52,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
    SquaredExponential,
)

from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_gp_layer,
    construct_mean_function,
    make_dataclass_from_class,
    xavier_initialization_numpy,
)


def test_construct_kernel_separate_independent_custom_list():
    kernel_list = [SquaredExponential(), Matern52()]
    mok = construct_basic_kernel(kernel_list)

    assert isinstance(mok, MultioutputKernel)
    assert isinstance(mok, SeparateIndependent)
    assert mok.kernels == kernel_list


def test_construct_kernel_separate_independent_duplicates():
    kernel = Matern52(variance=5)
    output_dim = 3
    mok = construct_basic_kernel(kernel, output_dim=output_dim, share_hyperparams=False)

    assert isinstance(mok, MultioutputKernel)
    assert isinstance(mok, SeparateIndependent)
    # check all kernels same type
    assert all([isinstance(k, Matern52) for k in mok.kernels])
    # check kernels have same hyperparameter values but are independent
    assert mok.kernels[0] is not mok.kernels[-1]
    assert mok.kernels[0].variance.numpy() == mok.kernels[-1].variance.numpy()
    assert mok.kernels[0].lengthscales.numpy() == mok.kernels[-1].lengthscales.numpy()


def test_construct_kernel_shared_independent_duplicates():
    kernel = Matern52(variance=5)
    output_dim = 3
    mok = construct_basic_kernel(kernel, output_dim=output_dim, share_hyperparams=True)

    assert isinstance(mok, MultioutputKernel)
    assert isinstance(mok, SharedIndependent)

    assert isinstance(mok.kernel, Matern52)
    assert mok.kernel is kernel


@pytest.mark.parametrize("z_init", [True, False])
def test_construct_inducing_separate_independent_custom_list(z_init):
    num_inducing = [25, 35, 45]
    input_dim = 5

    if z_init:
        z_init = [xavier_initialization_numpy(m, input_dim) for m in num_inducing]
    else:
        z_init = None

    moiv = construct_basic_inducing_variables(num_inducing, input_dim, z_init=z_init)

    assert isinstance(moiv, SeparateIndependentInducingVariables)
    assert isinstance(moiv, MultioutputInducingVariables)
    for i, iv in enumerate(moiv.inducing_variable_list):
        assert iv.num_inducing == num_inducing[i]


@pytest.mark.parametrize("z_init", [True, False])
def test_construct_inducing_separate_independent_duplicates(z_init):
    num_inducing = 25
    input_dim = 5
    output_dim = 7

    if z_init:
        z_init = np.random.randn(output_dim, num_inducing, input_dim)
    else:
        z_init = None

    moiv = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim=output_dim, z_init=z_init
    )

    assert isinstance(moiv, SeparateIndependentInducingVariables)
    assert isinstance(moiv, MultioutputInducingVariables)
    for iv in moiv.inducing_variable_list:
        assert iv.num_inducing == num_inducing


@pytest.mark.parametrize("z_init", [True, False])
def test_construct_inducing_shared_independent_duplicates(z_init):
    num_inducing = 25
    input_dim = 5
    output_dim = 7

    if z_init:
        z_init = np.random.randn(num_inducing, input_dim)
    else:
        z_init = None

    moiv = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim=output_dim, share_variables=True, z_init=z_init
    )

    assert isinstance(moiv, SharedIndependentInducingVariables)
    assert isinstance(moiv, MultioutputInducingVariables)
    assert moiv.inducing_variable.num_inducing == num_inducing


def test_construct_mean_function_Identity():
    num_data, input_dim, output_dim = 11, 5, 5
    X = np.random.randn(num_data, input_dim)
    mean_functions = construct_mean_function(X, input_dim, output_dim)
    assert isinstance(mean_functions, gpflow.mean_functions.Identity)


def test_construct_mean_function_Linear():
    num_data, input_dim, output_dim = 11, 5, 7
    X = np.random.randn(num_data, input_dim)
    mean_functions = construct_mean_function(X, input_dim, output_dim)
    assert isinstance(mean_functions, gpflow.mean_functions.Linear)


def test_construct_gp_layer():
    num_data = 11
    num_inducing = 23
    input_dim = 5
    output_dim = 7

    # build layer
    layer = construct_gp_layer(num_data, num_inducing, input_dim, output_dim)

    # kernel
    assert isinstance(layer.kernel, SharedIndependent)
    assert isinstance(layer.kernel.kernel, SquaredExponential)
    assert len(layer.kernel.kernel.lengthscales.numpy()) == input_dim, "expected ARD kernel"

    # inducing variable
    assert isinstance(layer.inducing_variable, SharedIndependentInducingVariables)
    assert isinstance(layer.inducing_variable.inducing_variable, InducingPoints)
    assert layer.inducing_variable.inducing_variable.num_inducing == num_inducing

    # mean function
    assert isinstance(layer.mean_function, gpflow.mean_functions.Zero)

    # variational parameters
    assert layer.q_mu.numpy().shape == (num_inducing, output_dim)
    assert layer.q_sqrt.numpy().shape == (output_dim, num_inducing, num_inducing)


def test_make_dataclass_from_class():
    @dataclass
    class BlankDataclass:
        foo: int
        bar: str

    class PopulatedClass:
        foo = 42
        bar = "hello world"
        baz = "other stuff"

    overwritten = "overwritten"
    assert PopulatedClass.bar != overwritten

    result = make_dataclass_from_class(BlankDataclass, PopulatedClass(), bar=overwritten)
    assert isinstance(result, BlankDataclass)
    assert result.foo == PopulatedClass.foo
    assert result.bar == overwritten
