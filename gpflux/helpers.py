# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import inspect
from dataclasses import fields
from typing import Optional, Type

import numpy as np

import gpflow
from gpflow import default_float as dfloat
from gpflow.kernels import SeparateIndependent, SharedIndependent
from gpflow.inducing_variables import (
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.utilities import deepcopy
from gpflux.layers import GPLayer
from gpflux.initializers import Initializer


def construct_basic_kernel(kernels, output_dim=None, share_hyperparams=False):
    if isinstance(kernels, list):
        mo_kern = SeparateIndependent(kernels)
    elif not share_hyperparams:
        copies = [deepcopy(kernels) for _ in range(output_dim)]
        mo_kern = SeparateIndependent(copies)
    else:
        mo_kern = SharedIndependent(kernels, output_dim)
    return mo_kern


def construct_basic_inducing_variables(
    num_inducing, input_dim, output_dim=None, share_variables=False
):
    if isinstance(num_inducing, list):
        if output_dim is not None:
            # TODO: the following assert may clash with MixedMultiOutputFeatures
            # where the number of independent GPs can differ from the output
            # dimension
            assert output_dim == len(num_inducing)  # pragma: no cover
        assert share_variables is False

        inducing_variables = []
        for num_ind_var in num_inducing:
            empty_array = np.zeros((num_ind_var, input_dim), dtype=dfloat())
            inducing_variables.append(InducingPoints(empty_array))
        return SeparateIndependentInducingVariables(inducing_variables)

    elif not share_variables:
        inducing_variables = []
        for _ in range(output_dim):
            empty_array = np.zeros((num_inducing, input_dim), dtype=dfloat())
            inducing_variables.append(InducingPoints(empty_array))
        return SeparateIndependentInducingVariables(inducing_variables)

    else:
        # TODO: should we assert output_dim is None ?

        empty_array = np.zeros((num_inducing, input_dim), dtype=dfloat())
        shared_ip = InducingPoints(empty_array)
        return SharedIndependentInducingVariables(shared_ip)


def construct_gp_layer(
    num_data: int,
    num_inducing: int,
    input_dim: int,
    output_dim: int,
    kernel_class: Type[gpflow.kernels.Stationary] = gpflow.kernels.SquaredExponential,
    initializer: Optional[Initializer] = None,
) -> GPLayer:
    """
    Builds a vanilla GP layer with a single kernel shared among all outputs,
        shared inducing point variables and zero mean function.

    :param num_data: total number of datapoints in the dataset, `N`.
        Typically corresponds to `X.shape[0] == len(X)`.
    :param num_inducing: total number of inducing variables, `M`.
        This parameter can be freely chosen by the user. General advice
        is to pick it as high as possible, but smaller than `N`.
        The computational complexity of the layer is cubic in `M`.
    :param input_dim: dimensionality of the input data (or features) `X`.
        Typically corresponds to `X.shape[1]`.
    :param output_dim: dimensionality of the outputs (or targets) `Y`.
        Typically corresponds to `Y.shape[1]`.
    :param kernel_class: the kernel class used by the layer.
        Can be as simple as `gpflow.kernels.SquaredExponential`, or complexer
        as `lambda **_: gpflow.kernels.Linear() + gpflow.kernels.Periodic()`.

    :return: a preconfigured GPLayer.
    """

    lengthscale = float(input_dim) ** 0.5
    base_kernel = kernel_class(lengthscales=np.full(input_dim, lengthscale))
    kernel = construct_basic_kernel(
        base_kernel, output_dim=output_dim, share_hyperparams=True,
    )
    inducing_variable = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim=output_dim, share_variables=True,
    )
    gp_layer = GPLayer(
        kernel=kernel,
        inducing_variable=inducing_variable,
        num_data=num_data,
        mean_function=gpflow.mean_functions.Zero(),
        initializer=initializer,
    )
    return gp_layer


def make_dataclass_from_class(dataclass, instance, **updates):
    """
    Takes a regular object `instance` with a superset of fields for a
    `dataclass` (`@dataclass`-decorated class), i.e. it has all of the
    dataclass's fields but may also have more, as well as additional key=value
    keyword arguments (`**updates`), and returns an instance of the dataclass.
    """
    dataclass_keys = [f.name for f in fields(dataclass)]
    field_dict = {k: v for k, v in inspect.getmembers(instance) if k in dataclass_keys}
    field_dict.update(updates)
    return dataclass(**field_dict)


def xavier_initialization_numpy(input_dim, output_dim):
    return (
        np.random.randn(input_dim, output_dim) * (2.0 / (input_dim + output_dim)) ** 0.5
    )
