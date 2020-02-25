# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import pytest

from gpflow.kernels import Matern52
from gpflow.mean_functions import Zero
from gpflow.inducing_variables import InducingPoints

from gpflux.utils.runtime_checks import verify_compatibility
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.exceptions import ShapeIncompatibilityError


# has no effect on compatibility in these tests
input_dim = 7
num_inducing = 35


def make_kernels(num_latent_k):
    return [
        construct_basic_kernel([Matern52() for _ in range(num_latent_k)]),
        construct_basic_kernel(
            Matern52(), output_dim=num_latent_k, share_hyperparams=False
        ),
        construct_basic_kernel(
            Matern52(), output_dim=num_latent_k, share_hyperparams=True
        ),
    ]


def make_inducing_variables(num_latent_iv):
    return [
        construct_basic_inducing_variables(
            num_inducing=[num_inducing for _ in range(num_latent_iv)],
            input_dim=input_dim,
        ),
        construct_basic_inducing_variables(
            num_inducing=num_inducing, input_dim=input_dim, output_dim=num_latent_iv
        ),
    ]


@pytest.mark.parametrize("kernel", make_kernels(3))
@pytest.mark.parametrize("inducing_variable", make_inducing_variables(3))
def test_verify_compatibility_num_latents_compatible(kernel, inducing_variable):
    mean_function = Zero()
    _, num_latents = verify_compatibility(kernel, mean_function, inducing_variable)
    assert num_latents == kernel.num_latents


@pytest.mark.parametrize("kernel", make_kernels(5))
@pytest.mark.parametrize("inducing_variable", make_inducing_variables(10))
def test_verify_compatibility_num_latents_incompatible(kernel, inducing_variable):
    mean_function = Zero()
    with pytest.raises(ShapeIncompatibilityError):
        verify_compatibility(kernel, mean_function, inducing_variable)


def test_verify_compatibility_type_errors():
    valid_inducing_variable = construct_basic_inducing_variables([35], input_dim=40)
    valid_kernel = construct_basic_kernel([Matern52()])
    valid_mean_function = Zero()  # all gpflow mean functions are currently valid

    with pytest.raises(TypeError):  # gpflow kernels must be MultioutputKernels
        verify_compatibility(Matern52(), valid_mean_function, valid_inducing_variable)

    Z = valid_inducing_variable.inducing_variable_list[0].Z
    inducing_variable = InducingPoints(Z)
    with pytest.raises(
        TypeError
    ):  # gpflow inducing_variables must be MultioutputInducingVariables
        verify_compatibility(valid_kernel, valid_mean_function, inducing_variable)
