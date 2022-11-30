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
"""
This module provides :func:`build_constant_input_dim_deep_gp` to build a Deep GP of
arbitrary depth where each hidden layer has the same input dimensionality as the data.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2

import gpflow
from gpflow.kernels import SquaredExponential
from gpflow.likelihoods import Gaussian

from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)
from gpflux.layers.gp_layer import OrthGPLayer
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.models import OrthDeepGP


@dataclass
class Config:
    """
    The configuration used by :func:`build_constant_input_dim_deep_gp`.
    """

    num_inducing_u: int
    """
    The number of inducing variables, *M*. The Deep GP uses the same number
    of inducing variables in each layer.
    """

    num_inducing_v: int
    """
    The number of inducing variables, *M*. The Deep GP uses the same number
    of inducing variables in each layer.
    """

    inner_layer_qsqrt_factor: float
    """
    A multiplicative factor used to rescale the hidden layers'
    :attr:`~gpflux.layers.GPLayer.q_sqrt`. Typically this value is chosen to be small
    (e.g., 1e-5) to reduce noise at the start of training.
    """

    likelihood_noise_variance: float
    """
    The variance of the :class:`~gpflow.likelihoods.Gaussian` likelihood that is used
    by the Deep GP.
    """

    whiten: bool = True
    """
    Determines the parameterisation of the inducing variables.
    If `True`, :math:``p(u) = N(0, I)``, otherwise :math:``p(u) = N(0, K_{uu})``.
    .. seealso:: :attr:`gpflux.layers.GPLayer.whiten`
    """


def _construct_kernel(input_dim: int, is_last_layer: bool) -> SquaredExponential:
    """
    Return a :class:`gpflow.kernels.SquaredExponential` kernel with ARD lengthscales set to
    2 and a small kernel variance of 1e-6 if the kernel is part of a hidden layer;
    otherwise, the kernel variance is set to 1.0.

    :param input_dim: The input dimensionality of the layer.
    :param is_last_layer: Whether the kernel is part of the last layer in the Deep GP.
    """
    variance = 0.351 if not is_last_layer else 0.351

    # TODO: Looking at this initializing to 2 (assuming N(0, 1) or U[0,1] normalized
    # data) seems a bit weird - that's really long lengthscales? And I remember seeing
    # something where the value scaled with the number of dimensions before
    lengthscales = [0.351] * input_dim
    return SquaredExponential(lengthscales=lengthscales, variance=variance)


def build_constant_input_dim_orth_deep_gp(
    X: np.ndarray, num_layers: int, config: Config
) -> OrthDeepGP:
    r"""
    Build a Deep GP consisting of ``num_layers`` :class:`GPLayer`\ s.
    All the hidden layers have the same input dimension as the data, that is, ``X.shape[1]``.

    The architecture is largely based on :cite:t:`salimbeni2017doubly`, with
    the most notable difference being that we keep the hidden dimension equal
    to the input dimensionality of the data.

    .. note::
        This architecture might be slow for high-dimensional data.

    .. note::
        This architecture assumes a :class:`~gpflow.likelihoods.Gaussian` likelihood
        for regression tasks. Specify a different likelihood for performing
        other tasks such as classification.

    :param X: The training input data, used to retrieve the number of datapoints and
        the input dimension and to initialise the inducing point locations using k-means. A
        tensor of rank two with the dimensions ``[num_data, input_dim]``.
    :param num_layers: The number of layers in the Deep GP.
    :param config: The configuration for (hyper)parameters. See :class:`Config` for details.
    """
    num_data, input_dim = X.shape
    X_running = X

    gp_layers = []
    centroids, _ = kmeans2(X, k=config.num_inducing_u + config.num_inducing_v, minit="points")

    centroids_u = centroids[: config.num_inducing_u, ...]
    centroids_v = centroids[config.num_inducing_u :, ...]

    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else input_dim

        # Pass in kernels, specify output dim (shared hyperparams/variables)

        inducing_var_u = construct_basic_inducing_variables(
            num_inducing=config.num_inducing_u,
            input_dim=D_in,
            share_variables=True,
            z_init=centroids_u,
        )

        inducing_var_v = construct_basic_inducing_variables(
            num_inducing=config.num_inducing_v,
            input_dim=D_in,
            share_variables=True,
            z_init=centroids_v,
        )

        kernel = construct_basic_kernel(
            kernels=_construct_kernel(D_in, is_last_layer),
            output_dim=D_out,
            share_hyperparams=True,
        )

        assert config.whiten is True, "non-whitened case not implemented yet"

        if is_last_layer:
            mean_function = gpflow.mean_functions.Zero()
            q_sqrt_scaling = 1.0
        else:
            mean_function = construct_mean_function(X_running, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = cast(tf.Tensor, X_running).numpy()
            q_sqrt_scaling = config.inner_layer_qsqrt_factor

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

    likelihood = Gaussian(config.likelihood_noise_variance)

    return OrthDeepGP(gp_layers, LikelihoodLayer(likelihood))
