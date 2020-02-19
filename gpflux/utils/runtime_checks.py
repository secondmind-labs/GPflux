# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from gpflow.inducing_variables import (
    MultioutputInducingVariables,
    FallbackSeparateIndependentInducingVariables,
)
from gpflow.kernels import MultioutputKernel
from gpflow.mean_functions import MeanFunction, Identity

from gpflux.exceptions import GPInitializationError


def verify_compatibility(
    kernel: MultioutputKernel,
    mean_function: MeanFunction,
    inducing_variable: MultioutputInducingVariables,
):
    """
        Provide error checking on shapes at layer construction. This method will be
        made simpler by having enhancements to GPflow: eg by adding foo.output_dim
        attribute, where foo is a MultioutputInducingPoints

        :param kernel: The multioutput kernel for the layer
        :param inducing_variable: The inducing features for the layer
        :param mean_function: The mean function applied to the inputs.
        """
    if not isinstance(inducing_variable, MultioutputInducingVariables):
        raise TypeError("`inducing_variable` must be a `MultioutputInducingVariables`")
    if not isinstance(kernel, MultioutputKernel):
        raise TypeError("`kernel` must be a `MultioutputKernel`")

    latent_inducing_points = None  # type: Optional[int]
    if isinstance(inducing_variable, FallbackSeparateIndependentInducingVariables):
        latent_inducing_points = len(inducing_variable.inducing_variable_list)

    num_latent_gps = kernel.num_latents

    if latent_inducing_points is not None:
        if latent_inducing_points != num_latent_gps:
            raise GPInitializationError(
                f"The number of latent GPs ({num_latent_gps}) does not match "
                f"the number of independent inducing_variables {latent_inducing_points}"
            )

    num_inducing_points = len(inducing_variable)  # currently the same for each dim
    return num_inducing_points, num_latent_gps
