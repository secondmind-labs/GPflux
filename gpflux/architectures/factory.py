#
# Copyright (c) 2022 The GPflux Contributors.
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
"""This module defines factories for various Deep GP architectures"""
from functools import singledispatch
from typing import Type, cast

import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2

from gpflow import default_float
from gpflow.kernels import Stationary
from gpflow.mean_functions import Zero

from gpflux.architectures.config import (
    HyperParametersConfig,
    ModelHyperParametersConfig,
    OrthogonalModelHyperparametersConfig,
)
from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)
from gpflux.layers.gp_layer import GPLayer, OrthGPLayer
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.models import DeepGP, OrthDeepGP


def build_kernel(input_dim: int, is_last_layer: bool, kernel: Type[Stationary]) -> Stationary:
    """
    Return a :class:`gpflow.kernels.Stationary` kernel with ARD lengthscales set to
    1.0 and a small kernel variance of 1e-6 if the kernel is part of a hidden layer;
    otherwise, the kernel variance is set to 1.0.

    :param input_dim: The input dimensionality of the layer.
    :param is_last_layer: Whether the kernel is part of the last layer in the Deep GP.
    :param kernel: the :class:`~gpflow.kernels.Stationary` type of the kernel
    """
    assert input_dim > 0, "Cannot have non positive input dimension"

    variance = 1e-6 if not is_last_layer else 1.0
    lengthscales = [1.0] * input_dim

    return kernel(lengthscales=lengthscales, variance=variance)


@singledispatch
def build_constant_input_dim_architecture(
    model_config: HyperParametersConfig, X: np.ndarray
) -> DeepGP:
    r"""
    Build a Deep GP consisting of a number of :class:`GPLayer`\ s.
    All the hidden layers have the same input dimension as the data, that is, ``X.shape[1]``.

    The architecture is largely based on :cite:t:`salimbeni2017doubly`, with
    the most notable difference being that we keep the hidden dimension equal
    to the input dimensionality of the data.

    .. note::
        This architecture might be slow for high-dimensional data.

    :param model_config: The configuration for (hyper)parameters.
    :param X: The training input data, used to retrieve the number of datapoints and
        the input dimension and to initialise the inducing point locations using k-means. A
        tensor of rank two with the dimensions ``[num_data, input_dim]``.

    :return: an instance of a DeepGP model
    :raises ValueError: If the config type is not registered.
    """
    raise ValueError(
        f"Don't know how to create model from config of type: {type(HyperParametersConfig)}"
    )


@build_constant_input_dim_architecture.register
def build_constant_input_dim_deep_gp(
    model_config: ModelHyperParametersConfig, X: np.ndarray
) -> DeepGP:
    if X.dtype != default_float():
        raise ValueError(
            f"X needs to have dtype according to gpflow.default_float() = {default_float()} "
            f"however got X with {X.dtype} dtype."
        )

    num_data, input_dim = X.shape
    X_running = X

    gp_layers = []
    centroids, _ = kmeans2(X, k=model_config.num_inducing, minit="points")

    num_layers = model_config.num_layers
    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else input_dim

        # Pass in kernels, specify output dim (shared hyperparams/variables)

        inducing_var = construct_basic_inducing_variables(
            num_inducing=model_config.num_inducing,
            input_dim=D_in,
            share_variables=True,
            z_init=centroids,
        )

        kernel = construct_basic_kernel(
            kernels=build_kernel(D_in, is_last_layer, model_config.kernel),
            output_dim=D_out,
            share_hyperparams=True,
        )

        assert model_config.whiten is True, "non-whitened case not implemented yet"

        if is_last_layer:
            mean_function = Zero()
            q_sqrt_scaling = 1.0
        else:
            mean_function = construct_mean_function(X_running, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = cast(tf.Tensor, X_running).numpy()
            q_sqrt_scaling = model_config.inner_layer_qsqrt_factor

        layer = GPLayer(
            kernel,
            inducing_var,
            num_data,
            mean_function=mean_function,
            name=f"gp_{i_layer}",
        )
        layer.q_sqrt.assign(layer.q_sqrt * q_sqrt_scaling)
        gp_layers.append(layer)

    likelihood = model_config.likelihood.create()

    return DeepGP(gp_layers, LikelihoodLayer(likelihood))


@build_constant_input_dim_architecture.register
def build_constant_input_dim_orthogonal_deep_gp(
    model_config: OrthogonalModelHyperparametersConfig, X: np.ndarray
) -> OrthDeepGP:
    if X.dtype != default_float():
        raise ValueError(
            f"X needs to have dtype according to gpflow.default_float() = {default_float()} "
            f"however got X with {X.dtype} dtype."
        )

    num_data, input_dim = X.shape
    X_running = X

    num_inducing_u = model_config.num_inducing_u
    num_inducing_v = model_config.num_inducing_v
    gp_layers = []
    centroids, _ = kmeans2(X, k=min(num_inducing_u + num_inducing_v, X.shape[0]), minit="points")

    centroids_u = centroids[:num_inducing_u, ...]
    centroids_v = centroids[num_inducing_u:, ...]

    num_layers = model_config.num_layers
    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else input_dim

        # Pass in kernels, specify output dim (shared hyperparams/variables)

        inducing_var_u = construct_basic_inducing_variables(
            num_inducing=num_inducing_u,
            input_dim=D_in,
            share_variables=True,
            z_init=centroids_u,
        )

        inducing_var_v = construct_basic_inducing_variables(
            num_inducing=num_inducing_v,
            input_dim=D_in,
            share_variables=True,
            z_init=centroids_v,
        )

        kernel = construct_basic_kernel(
            kernels=build_kernel(D_in, is_last_layer, model_config.kernel),
            output_dim=D_out,
            share_hyperparams=True,
        )

        assert model_config.whiten is True, "non-whitened case not implemented yet"

        if is_last_layer:
            mean_function = Zero()
            q_sqrt_scaling = 1.0
        else:
            mean_function = construct_mean_function(X_running, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = cast(tf.Tensor, X_running).numpy()
            q_sqrt_scaling = model_config.inner_layer_qsqrt_factor

        layer = OrthGPLayer(
            kernel,
            inducing_var_u,
            inducing_var_v,
            num_data,
            mean_function=mean_function,
            name=f"orth_gp_{i_layer}",
            num_latent_gps=D_out,
        )
        layer.q_sqrt_u.assign(layer.q_sqrt_u * q_sqrt_scaling)
        layer.q_sqrt_v.assign(layer.q_sqrt_v * q_sqrt_scaling)
        gp_layers.append(layer)

    likelihood = model_config.likelihood.create()

    return OrthDeepGP(gp_layers, LikelihoodLayer(likelihood))
