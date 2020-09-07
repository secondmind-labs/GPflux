from dataclasses import dataclass

import numpy as np
import gpflow

from gpflow.kernels import (
    Matern52,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
    SquaredExponential,
)
from gpflow.inducing_variables import (
    InducingPoints,
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)

from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_gp_layer,
    make_dataclass_from_class,
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


def test_construct_inducing_separate_independent_custom_list():
    num_inducing = [25, 35, 45]
    input_dim = 5

    moiv = construct_basic_inducing_variables(num_inducing, input_dim)

    assert isinstance(moiv, SeparateIndependentInducingVariables)
    assert isinstance(moiv, MultioutputInducingVariables)
    for i, iv in enumerate(moiv.inducing_variable_list):
        assert len(iv) == num_inducing[i]
        np.testing.assert_equal(iv.Z.numpy(), np.zeros((num_inducing[i], input_dim)))


def test_construct_inducing_separate_independent_duplicates():
    num_inducing = 25
    input_dim = 5
    output_dim = 7

    moiv = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim=output_dim
    )

    assert isinstance(moiv, SeparateIndependentInducingVariables)
    assert isinstance(moiv, MultioutputInducingVariables)
    for iv in moiv.inducing_variable_list:
        assert len(iv) == num_inducing
        np.testing.assert_equal(iv.Z.numpy(), np.zeros((num_inducing, input_dim)))


def test_construct_inducing_shared_independent_duplicates():
    num_inducing = 25
    input_dim = 5
    output_dim = 7

    moiv = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim=output_dim, share_variables=True
    )

    assert isinstance(moiv, SharedIndependentInducingVariables)
    assert isinstance(moiv, MultioutputInducingVariables)
    assert len(moiv.inducing_variable) == num_inducing
    np.testing.assert_equal(
        moiv.inducing_variable.Z.numpy(), np.zeros((num_inducing, input_dim))
    )


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
    assert (
        len(layer.kernel.kernel.lengthscales.numpy()) == input_dim
    ), "expected ARD kernel"

    # inducing variable
    assert isinstance(layer.inducing_variable, SharedIndependentInducingVariables)
    assert isinstance(layer.inducing_variable.inducing_variable, InducingPoints)
    assert len(layer.inducing_variable.inducing_variable) == num_inducing

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

    result = make_dataclass_from_class(
        BlankDataclass, PopulatedClass(), bar=overwritten
    )
    assert isinstance(result, BlankDataclass)
    assert result.foo == PopulatedClass.foo
    assert result.bar == overwritten
