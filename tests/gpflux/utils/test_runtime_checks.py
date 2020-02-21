# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import pytest

from gpflow.kernels import RBF
from gpflow.mean_functions import Zero
from gpflow.inducing_variables import InducingPoints

from gpflux.utils.runtime_checks import verify_compatibility
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.exceptions import ShapeIncompatibilityError


def make_kernels_and_inducing_variables():
    input_dim = 40  # has no effect on compatibility in these tests

    num_latents = [(3, 3), (5, 10)]  # (kernel_latents, iv_latents)
    test_combos = []
    for num_latent_k, num_latent_iv in num_latents:

        kernels = [
            construct_basic_kernel([RBF() for _ in range(num_latent_k)]),
            construct_basic_kernel(
                RBF(), output_dim=num_latent_k, share_hyperparams=False
            ),
            construct_basic_kernel(
                RBF(), output_dim=num_latent_k, share_hyperparams=True
            ),
        ]

        inducing_variables = [
            construct_basic_inducing_variables(
                [35 for _ in range(num_latent_iv)], input_dim=40
            ),
            construct_basic_inducing_variables(
                35, input_dim=40, output_dim=num_latent_iv
            ),
        ]

        are_compatible = num_latent_k == num_latent_iv
        test_combos.extend(
            [(k, iv, are_compatible) for k in kernels for iv in inducing_variables]
        )
    print(test_combos)
    return test_combos


@pytest.mark.parametrize(
    "kernel,inducing_variable,are_compatible", make_kernels_and_inducing_variables()
)
def test_verify_compatibility_num_latents(kernel, inducing_variable, are_compatible):
    mean_function = Zero()
    if are_compatible:
        _, num_latents = verify_compatibility(kernel, mean_function, inducing_variable)
        assert num_latents == kernel.num_latents
    else:
        with pytest.raises(ShapeIncompatibilityError):
            verify_compatibility(kernel, mean_function, inducing_variable)


def test_verify_compatibility_type_errors():
    valid_inducing_variable = construct_basic_inducing_variables([35], input_dim=40)
    valid_kernel = construct_basic_kernel([RBF()])
    valid_mean_function = Zero()  # all gpflow mean functions are currently valid

    with pytest.raises(TypeError):  # gpflow kernels must me MultioutputKernels
        verify_compatibility(RBF(), valid_mean_function, valid_inducing_variable)

    Z = valid_inducing_variable.inducing_variable_list[0].Z
    inducing_variable = InducingPoints(Z)
    with pytest.raises(
        TypeError
    ):  # gpflow inducing_variables must me MultioutputInducingVariables
        verify_compatibility(valid_kernel, valid_mean_function, inducing_variable)
