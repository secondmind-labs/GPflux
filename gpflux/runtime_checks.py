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
""" Runtime checks """
from typing import Optional, Tuple

from gpflow.inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    MultioutputInducingVariables,
)
from gpflow.kernels import MultioutputKernel
from gpflow.mean_functions import MeanFunction

from gpflux.exceptions import GPLayerIncompatibilityException


def verify_compatibility(
    kernel: MultioutputKernel,
    mean_function: MeanFunction,
    inducing_variable: MultioutputInducingVariables,
) -> Tuple[int, int]:
    """
    Provide error checking on shapes at layer construction. This method will be
    made simpler by having enhancements to GPflow: eg by adding foo.output_dim
    attribute, where foo is a MultioutputInducingVariable

    :param kernel: The multioutput kernel for the layer
    :param inducing_variable: The inducing features for the layer
    :param mean_function: The mean function applied to the inputs.
    """
    if not isinstance(inducing_variable, MultioutputInducingVariables):
        raise GPLayerIncompatibilityException(
            "`inducing_variable` must be a `gpflow.inducing_variables.MultioutputInducingVariables`"
        )
    if not isinstance(kernel, MultioutputKernel):
        raise GPLayerIncompatibilityException(
            "`kernel` must be a `gpflow.kernels.MultioutputKernel`"
        )
    if not isinstance(mean_function, MeanFunction):
        raise GPLayerIncompatibilityException(
            "`kernel` must be a `gpflow.mean_functions.MeanFunction`"
        )

    latent_inducing_points: Optional[int] = None
    if isinstance(inducing_variable, FallbackSeparateIndependentInducingVariables):
        latent_inducing_points = len(inducing_variable.inducing_variable_list)

    num_latent_gps = kernel.num_latent_gps

    if latent_inducing_points is not None:
        if latent_inducing_points != num_latent_gps:
            raise GPLayerIncompatibilityException(
                f"The number of latent GPs ({num_latent_gps}) does not match "
                f"the number of separate independent inducing_variables ({latent_inducing_points})"
            )

    num_inducing_points = len(inducing_variable)  # currently the same for each dim
    return num_inducing_points, num_latent_gps
