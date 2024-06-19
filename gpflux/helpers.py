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
r"""
This module contains helper functions for constructing :class:`~gpflow.kernels.MultioutputKernel`,
:class:`~gpflow.inducing_variables.MultioutputInducingVariables`,
:class:`~gpflow.mean_functions.MeanFunction`, and :class:`~gpflux.layers.GPLayer` objects.
"""

import inspect
import warnings
from dataclasses import fields
from typing import Any, List, Optional, Type, TypeVar, Union

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
) -> gpflow.kernels.MultioutputKernel:
    r"""
    Construct a :class:`~gpflow.kernels.MultioutputKernel` to use
    in :class:`GPLayer`\ s.

    :param kernels: A single kernel or list of :class:`~gpflow.kernels.Kernel`\ s.
        - When a single kernel is passed, the same kernel is used for all
        outputs. Depending on ``share_hyperparams``, the hyperparameters will be
        shared across outputs. You must also specify ``output_dim``.
        - When a list of kernels is passed, each kernel in the list is used on a separate
        output dimension and a :class:`gpflow.kernels.SeparateIndependent` is returned.
    :param output_dim: The number of outputs. This is equal to the number of latent GPs
        in the :class:`GPLayer`. When only a single kernel is specified for ``kernels``,
        you must also specify ``output_dim``. When a list of kernels is specified for ``kernels``,
        we assume that ``len(kernels) == output_dim``, and ``output_dim`` is not required.
    :param share_hyperparams: If `True`, use the type of kernel and the same hyperparameters
        (variance and lengthscales) for the different outputs. Otherwise, the
        same type of kernel (Squared-Exponential, Matern12, and so on) is used for
        the different outputs, but the kernel can have different hyperparameter values for each.
    """
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
) -> gpflow.inducing_variables.MultioutputInducingVariables:
    r"""
    Construct a compatible :class:`~gpflow.inducing_variables.MultioutputInducingVariables`
    to use in :class:`GPLayer`\ s.

    :param num_inducing: The total number of inducing variables, ``M``.
        This parameter can be freely chosen by the user. General advice
        is to set it as high as possible, but smaller than the number of datapoints.
        The computational complexity of the layer is cubic in ``M``.
        If a list is passed, each element in the list specifies the number of inducing
        variables to use for each ``output_dim``.
    :param input_dim: The dimensionality of the input data (or features) ``X``.
        Typically, this corresponds to ``X.shape[-1]``.
        For :class:`~gpflow.inducing_variables.InducingPoints`, this specifies the dimensionality
        of ``Z``.
    :param output_dim: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]`` or the number of latent GPs.
        The parameter is used to determine the number of inducing variable sets
        to create when a different set is used for each output. The parameter
        is redundant when ``num_inducing`` is a list, because the code assumes
        that ``len(num_inducing) == output_dim``.
    :param share_variables: If `True`, use the same inducing variables for different
        outputs. Otherwise, create a different set for each output. Set this parameter to
        `False` when ``num_inducing`` is a list, because otherwise the two arguments
        contradict each other. If you set this parameter to `True`, you must also specify
        ``output_dim``, because that is used to determine the number of inducing variable
        sets to create.
    :param z_init: Raw values to use to initialise
        :class:`gpflow.inducing_variables.InducingPoints`. If `None` (the default), values
        will be initialised from ``N(0, 1)``. The shape of ``z_init`` depends on the other
        input arguments. If a single set of inducing points is used for all outputs (that
        is, if ``share_variables`` is `True`), ``z_init`` should be rank two, with the
        dimensions ``[M, input_dim]``. If a different set of inducing points is used for
        the outputs (ithat is, if ``num_inducing`` is a list, or if ``share_variables`` is
        `False`), ``z_init`` should be a rank three tensor with the dimensions
        ``[output_dim, M, input_dim]``.
    """

    if z_init is None:
        warnings.warn(
            "No `z_init` has been specified in `construct_basic_inducing_variables`. "
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
    Return :class:`gpflow.mean_functions.Identity` when ``D_in`` and ``D_out`` are
    equal. Otherwise, use the principal components of the inputs matrix ``X`` to build a
    :class:`~gpflow.mean_functions.Linear` mean function.

    .. note::
        The returned mean function is set to be untrainable.
        To change this, use :meth:`gpflow.set_trainable`.

    :param X: A data array with the shape ``[N, D_in]`` used to determine the principal
        components to use to create a :class:`~gpflow.mean_functions.Linear` mean function
        when ``D_in != D_out``.
    :param D_in: The dimensionality of the input data (or features) ``X``.
        Typically, this corresponds to ``X.shape[-1]``.
    :param D_out: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]`` or the number of latent GPs in the layer.
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

    :param num_data: total number of datapoints in the dataset, *N*.
        Typically corresponds to ``X.shape[0] == len(X)``.
    :param num_inducing: total number of inducing variables, *M*.
        This parameter can be freely chosen by the user. General advice
        is to pick it as high as possible, but smaller than *N*.
        The computational complexity of the layer is cubic in *M*.
    :param input_dim: dimensionality of the input data (or features) X.
        Typically, this corresponds to ``X.shape[-1]``.
    :param output_dim: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]``.
    :param kernel_class: The kernel class used by the layer.
        This can be as simple as :class:`gpflow.kernels.SquaredExponential`, or more complex,
        for example, ``lambda **_: gpflow.kernels.Linear() + gpflow.kernels.Periodic()``.
        It will be passed a ``lengthscales`` keyword argument.
    :param z_init: The initial value for the inducing variable inputs.
    :param name: The name for the GP layer.
    """
    lengthscale = float(input_dim) ** 0.5
    base_kernel = kernel_class(lengthscales=np.full(input_dim, lengthscale))
    kernel = construct_basic_kernel(base_kernel, output_dim=output_dim, share_hyperparams=True)
    inducing_variable = construct_basic_inducing_variables(
        num_inducing,
        input_dim,
        output_dim=output_dim,
        share_variables=True,
        z_init=z_init,
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


# HACK to get mypy to pass, should be (dataclass: Type[T], ...) -> T:
# mypy said: gpflux/helpers.py:271: error: Argument 1 to "fields" has incompatible type "Type[T]";
# expected "Union[DataclassInstance, Type[DataclassInstance]]"  [arg-type]
def make_dataclass_from_class(dataclass: Any, instance: object, **updates: object) -> Any:
    """
    Take a regular object ``instance`` with a superset of fields for a
    :class:`dataclasses.dataclass` (``@dataclass``-decorated class), and return an
    instance of the dataclass. The ``instance`` has all of the dataclass's fields
    but might also have more. ``key=value`` keyword arguments supersede the fields in ``instance``.
    """
    dataclass_keys = [f.name for f in fields(dataclass)]
    field_dict = {k: v for k, v in inspect.getmembers(instance) if k in dataclass_keys}
    field_dict.update(updates)
    return dataclass(**field_dict)  # type: ignore


def xavier_initialization_numpy(input_dim: int, output_dim: int) -> np.ndarray:
    r"""
    Generate initial weights for a neural network layer with the given input and output
    dimensionality using the Xavier Glorot normal initialiser. From:

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
    feedforward neural networks." Proceedings of the thirteenth international
    conference on artificial intelligence and statistics. JMLR Workshop and Conference
    Proceedings, 2010.

    Draw samples from a normal distribution centred on :math:`0` with standard deviation
    :math:`\sqrt(2 / (\text{input_dim} + \text{output_dim}))`.
    """
    return np.random.randn(input_dim, output_dim) * (2.0 / (input_dim + output_dim)) ** 0.5
