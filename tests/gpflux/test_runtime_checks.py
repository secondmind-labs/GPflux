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
import pytest

from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Matern52
from gpflow.mean_functions import Zero

from gpflux.exceptions import GPLayerIncompatibilityException
from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.runtime_checks import verify_compatibility

# has no effect on compatibility in these tests
input_dim = 7
num_inducing = 35


def make_kernels(num_latent_k):
    return [
        construct_basic_kernel([Matern52() for _ in range(num_latent_k)]),
        construct_basic_kernel(Matern52(), output_dim=num_latent_k, share_hyperparams=False),
        construct_basic_kernel(Matern52(), output_dim=num_latent_k, share_hyperparams=True),
    ]


def make_inducing_variables(num_latent_iv):
    return [
        construct_basic_inducing_variables(
            num_inducing=[num_inducing for _ in range(num_latent_iv)], input_dim=input_dim,
        ),
        construct_basic_inducing_variables(
            num_inducing=num_inducing, input_dim=input_dim, output_dim=num_latent_iv
        ),
    ]


@pytest.mark.parametrize("kernel", make_kernels(3))
@pytest.mark.parametrize("inducing_variable", make_inducing_variables(3))
def test_verify_compatibility_num_latent_gps_compatible(kernel, inducing_variable):
    mean_function = Zero()
    _, num_latent_gps = verify_compatibility(kernel, mean_function, inducing_variable)
    assert num_latent_gps == kernel.num_latent_gps


@pytest.mark.parametrize("kernel", make_kernels(5))
@pytest.mark.parametrize("inducing_variable", make_inducing_variables(10))
def test_verify_compatibility_num_latent_gps_incompatible(kernel, inducing_variable):
    mean_function = Zero()
    with pytest.raises(GPLayerIncompatibilityException):
        verify_compatibility(kernel, mean_function, inducing_variable)


def test_verify_compatibility_type_errors():
    valid_inducing_variable = construct_basic_inducing_variables([35], input_dim=40)
    valid_kernel = construct_basic_kernel([Matern52()])
    valid_mean_function = Zero()  # all gpflow mean functions are currently valid

    with pytest.raises(
        GPLayerIncompatibilityException
    ):  # gpflow kernels must be MultioutputKernels
        verify_compatibility(Matern52(), valid_mean_function, valid_inducing_variable)

    Z = valid_inducing_variable.inducing_variable_list[0].Z
    inducing_variable = InducingPoints(Z)
    with pytest.raises(
        GPLayerIncompatibilityException
    ):  # gpflow inducing_variables must be MultioutputInducingVariables
        verify_compatibility(valid_kernel, valid_mean_function, inducing_variable)
