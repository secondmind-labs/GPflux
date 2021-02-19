# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import inspect
import warnings
from dataclasses import fields
from typing import List, Optional, Type, TypeVar, Union

import numpy as np

import gpflow
from gpflow import default_float
from gpflow.inducing_variables import (
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import SeparateIndependent, SharedIndependent
from gpflow.utilities import deepcopy

from gpflux.layers.gp_layer import GPLayer


def construct_basic_kernel(
    kernels: Union[gpflow.kernels.Kernel, List[gpflow.kernels.Kernel]],
    output_dim: Optional[int] = None,
    share_hyperparams: bool = False,
) -> gpflow.kernels.Kernel:
    if isinstance(kernels, list):
        mo_kern = SeparateIndependent(kernels)
    elif not share_hyperparams:
        copies = [deepcopy(kernels) for _ in range(output_dim)]
        mo_kern = SeparateIndependent(copies)
    else:
        mo_kern = SharedIndependent(kernels, output_dim)
    return mo_kern


def construct_basic_inducing_variables(
    num_inducing: Union[int, List[int]],
    input_dim: int,
    output_dim: Optional[int] = None,
    share_variables: bool = False,
    z_init: Optional[np.ndarray] = None,
) -> gpflow.inducing_variables.InducingVariables:

    if z_init is None:
        warnings.warn(
            "No `z_init` has been specified in `construct_basic_inducing_variables`."
            "Default initialization using random normal N(0, 1) will be used."
        )

    z_init_is_given = z_init is not None

    if isinstance(num_inducing, list):
        if output_dim is not None:
            # TODO: the following assert may clash with MixedMultiOutputFeatures
            # where the number of independent GPs can differ from the output
            # dimension
            assert output_dim == len(num_inducing)  # pragma: no cover
        assert share_variables is False

        inducing_variables = []
        for i, num_ind_var in enumerate(num_inducing):
            if z_init_is_given:
                assert len(z_init[i]) == num_ind_var
                z_init_i = z_init[i]
            else:
                z_init_i = np.random.randn(num_ind_var, input_dim).astype(dtype=default_float())
            assert z_init_i.shape == (num_ind_var, input_dim)
            inducing_variables.append(InducingPoints(z_init_i))
        return SeparateIndependentInducingVariables(inducing_variables)

    elif not share_variables:
        inducing_variables = []
        for o in range(output_dim):
            if z_init_is_given:
                if z_init.shape != (output_dim, num_inducing, input_dim):
                    raise ValueError(
                        "When not sharing variables, z_init must have shape"
                        "[output_dim, num_inducing, input_dim]"
                    )
                z_init_o = z_init[o]
            else:
                z_init_o = np.random.randn(num_inducing, input_dim).astype(dtype=default_float())
            inducing_variables.append(InducingPoints(z_init_o))
        return SeparateIndependentInducingVariables(inducing_variables)

    else:
        # TODO: should we assert output_dim is None ?

        z_init = (
            z_init
            if z_init_is_given
            else np.random.randn(num_inducing, input_dim).astype(dtype=default_float())
        )
        shared_ip = InducingPoints(z_init)
        return SharedIndependentInducingVariables(shared_ip)


def construct_mean_function(
    X: np.ndarray, D_in: int, D_out: int
) -> gpflow.mean_functions.MeanFunction:
    """
    Returns Identity mean function when D_in and D_out are equal. Otherwise uses
    the principal components in `X` to build a Linear mean function.

    The returned mean function is set to be untrainable. To change this,
    use `gpflow.set_trainable`.
    """
    assert X.shape[-1] == D_in
    if D_in == D_out:
        mean_function = gpflow.mean_functions.Identity()
    else:
        if D_in > D_out:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            W = V[:D_out, :].T
        else:
            W = np.concatenate([np.eye(D_in), np.zeros((D_in, D_out - D_in))], axis=1)

        assert W.shape == (D_in, D_out)
        mean_function = gpflow.mean_functions.Linear(W)
        gpflow.set_trainable(mean_function, False)

    return mean_function


def construct_gp_layer(
    num_data: int,
    num_inducing: int,
    input_dim: int,
    output_dim: int,
    kernel_class: Type[gpflow.kernels.Stationary] = gpflow.kernels.SquaredExponential,
    z_init: Optional[np.ndarray] = None,
    name: Optional[str] = None,
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
    :param z_init: initialisation for inducing variable inputs.
    :param name: name for the GP layer.

    :return: a preconfigured GPLayer.
    """

    lengthscale = float(input_dim) ** 0.5
    base_kernel = kernel_class(lengthscales=np.full(input_dim, lengthscale))
    kernel = construct_basic_kernel(base_kernel, output_dim=output_dim, share_hyperparams=True)
    inducing_variable = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim=output_dim, share_variables=True, z_init=z_init,
    )
    gp_layer = GPLayer(
        kernel=kernel,
        inducing_variable=inducing_variable,
        num_data=num_data,
        mean_function=gpflow.mean_functions.Zero(),
        name=name,
    )
    return gp_layer


T = TypeVar("T")


def make_dataclass_from_class(dataclass: Type[T], instance: object, **updates: object) -> T:
    """
    Takes a regular object `instance` with a superset of fields for a
    `dataclass` (`@dataclass`-decorated class), i.e. it has all of the
    dataclass's fields but may also have more, as well as additional key=value
    keyword arguments (`**updates`), and returns an instance of the dataclass.
    """
    dataclass_keys = [f.name for f in fields(dataclass)]
    field_dict = {k: v for k, v in inspect.getmembers(instance) if k in dataclass_keys}
    field_dict.update(updates)
    return dataclass(**field_dict)  # type: ignore


def xavier_initialization_numpy(input_dim: int, output_dim: int) -> np.ndarray:
    return np.random.randn(input_dim, output_dim) * (2.0 / (input_dim + output_dim)) ** 0.5
