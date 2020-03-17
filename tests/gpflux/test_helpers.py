import numpy as np

from gpflow.kernels import (
    RBF,
    Matern52,
    MultioutputKernel,
    SharedIndependent,
    SeparateIndependent,
)
from gpflow.inducing_variables import (
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)

from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables


def test_construct_kernel_separate_independent_custom_list():
    kernel_list = [RBF(), Matern52()]
    mok = construct_basic_kernel(kernel_list)

    assert isinstance(mok, MultioutputKernel)
    assert isinstance(mok, SeparateIndependent)
    assert mok.kernels == kernel_list


def test_construct_kernel_separate_independent_duplicates():
    kernel = RBF(variance=5)
    output_dim = 3
    mok = construct_basic_kernel(kernel, output_dim=output_dim, share_hyperparams=False)

    assert isinstance(mok, MultioutputKernel)
    assert isinstance(mok, SeparateIndependent)
    # check all kernels same type
    assert all([isinstance(k, RBF) for k in mok.kernels])
    # check kernels have same hyperparameter values but are independent
    assert mok.kernels[0] is not mok.kernels[-1]
    assert mok.kernels[0].variance == mok.kernels[-1].variance
    assert mok.kernels[0].lengthscales == mok.kernels[-1].lengthscales


def test_construct_kernel_shared_independent_duplicates():
    kernel = RBF(variance=5)
    output_dim = 3
    mok = construct_basic_kernel(kernel, output_dim=output_dim, share_hyperparams=True)

    assert isinstance(mok, MultioutputKernel)
    assert isinstance(mok, SharedIndependent)

    assert isinstance(mok.kernel, RBF)
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
    assert len(moiv.inducing_variable_shared) == num_inducing
    np.testing.assert_equal(
        moiv.inducing_variable_shared.Z.numpy(), np.zeros((num_inducing, input_dim))
    )
