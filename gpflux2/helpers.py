# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import numpy as np

from gpflow import default_float as dfloat
from gpflow.kernels import SeparateIndependent, SharedIndependent
from gpflow.inducing_variables import (
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
    InducingPoints,
)
from gpflow.utilities import deepcopy_components


def construct_basic_kernel(kernels, output_dim=None, share_hyperparams=False):
    if isinstance(kernels, list):
        mo_kern = SeparateIndependent(kernels)
    elif not share_hyperparams:
        copies = [deepcopy_components(kernels) for _ in range(output_dim)]
        mo_kern = SeparateIndependent(copies)
    else:
        mo_kern = SharedIndependent(kernels, output_dim)
    return mo_kern


def construct_basic_inducing_variables(
    num_inducing, input_dim, output_dim=None, share_variables=False
):
    if isinstance(num_inducing, list):
        inducing_variables = []
        for num_ind_var in num_inducing:
            empty_array = np.empty((num_ind_var, input_dim), dtype=dfloat())
            inducing_variables.append(InducingPoints(empty_array))
        return SeparateIndependentInducingVariables(inducing_variables)

    elif not share_variables:
        inducing_variables = []
        for _ in range(output_dim):
            empty_array = np.empty((num_inducing, input_dim), dtype=dfloat())
            inducing_variables.append(InducingPoints(empty_array))
        return SeparateIndependentInducingVariables(inducing_variables)

    else:
        empty_array = np.empty((num_inducing, input_dim), dtype=dfloat())
        shared_ip = InducingPoints(empty_array)
        return SharedIndependentInducingVariables(shared_ip)
