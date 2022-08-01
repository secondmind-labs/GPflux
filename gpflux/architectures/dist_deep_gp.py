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
from warnings import WarningMessage

import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2

from gpflow.kernels import SquaredExponential
from gpflux.kernels import Hybrid
from gpflow.likelihoods import Gaussian, Bernoulli, MultiClass

from gpflux.helpers import (
    construct_basic_hybrid_kernel,
    construct_basic_inducing_variables,
    construct_basic_distributional_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)

from gpflux.layers import (
    DistGPLayer,
    GPLayer,
    LikelihoodLayer
)

from gpflux.models import DistDeepGP
from gpflow.mean_functions import Zero, Identity

@dataclass
class Config:
    """
    The configuration used by :func:`build_constant_input_dim_deep_gp`.
    """

    num_inducing: int
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
    
    hidden_layer_size : int
    """
    #TODO -- update this
    """

    task_type: str
    """
    Can either be 'regression' or 'classification'
    """

    dim_output: int
    """
    Mostly to be used for 'classification' option
    """

    whiten: bool = True
    """
    Determines the parameterisation of the inducing variables.
    If `True`, :math:``p(u) = N(0, I)``, otherwise :math:``p(u) = N(0, K_{uu})``.
    .. seealso:: :attr:`gpflux.layers.GPLayer.whiten`
    """



def _construct_euclidean_kernel(input_dim: int, is_last_layer: bool) -> SquaredExponential:
    """
    Return a :class:`gpflow.kernels.SquaredExponential` kernel with ARD lengthscales set to
    2 and a small kernel variance of 1e-6 if the kernel is part of a hidden layer;
    otherwise, the kernel variance is set to 1.0.

    :param input_dim: The input dimensionality of the layer.
    :param is_last_layer: Whether the kernel is part of the last layer in the Deep GP.
    """ 
    ### Custom values taken from DistDGP paper 
    variance = 0.351 if not is_last_layer else 0.351

    # TODO: Looking at this initializing to 2 (assuming N(0, 1) or U[0,1] normalized
    # data) seems a bit weird - that's really long lengthscales? And I remember seeing
    # something where the value scaled with the number of dimensions before
    lengthscales = [0.351] * input_dim
    return SquaredExponential(lengthscales=lengthscales, variance=variance)

def _construct_hybrid_kernel(input_dim: int, is_last_layer: bool, name:str) -> Hybrid:
    """
    Return a :class:`..kernels.Hybrid` kernel with ARD lengthscales set to
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

    return Hybrid("squared_exponential", lengthscales=lengthscales, variance=variance, name = name)


def build_dist_deep_gp(X: np.ndarray, num_layers: int, config: Config) -> DistDeepGP:
    r"""
    Build a Distributional Deep GP consisting of ``num_layers`` :class:`GPLayer`\ s.
    All the hidden layers have the same input dimension as the data, that is, ``X.shape[1]``.

    The architecture is largely based on :cite:t:`popescu2021hierarchical`, with
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
    centroids, _ = kmeans2(X, k=config.num_inducing, minit="points")

    ############################################
    ############# Euclidean space ##############
    ############################################

    i_layer = 0
    ### First layer is standard Euclidean-space SVGP ###
    is_last_layer = i_layer == num_layers - 1
    D_in = input_dim
    D_out = config.dim_output if is_last_layer else config.hidden_layer_size

    # Pass in kernels, specify output dim (shared hyperparams/variables)

    inducing_var = construct_basic_inducing_variables(
        num_inducing=config.num_inducing, input_dim=D_in, share_variables=True, z_init=centroids
    )

    kernel = construct_basic_kernel(
        kernels=_construct_euclidean_kernel(D_in, is_last_layer),
        output_dim=D_out,
        share_hyperparams=True,
    )

    assert config.whiten is True, "non-whitened case not implemented yet"

    if is_last_layer:
        mean_function = Zero()
        q_sqrt_scaling = 1.0
    else:
        mean_function = construct_mean_function(X_running, D_out)
        #X_running = mean_function(X_running)
        if tf.is_tensor(X_running):
            X_running = X_running.numpy()
        
        q_sqrt_scaling = config.inner_layer_qsqrt_factor

    layer = GPLayer(
        kernel,
        inducing_var,
        num_data,
        num_latent_gps=config.hidden_layer_size,
        mean_function=mean_function,
        name=f"gp_{i_layer}",
    )
    layer.q_sqrt.assign(layer.q_sqrt * q_sqrt_scaling)
    gp_layers.append(layer)

    ################################################################
    ############# Joint Euclidean & Wasserstein-2 space ############
    ################################################################

    for i_layer in range(1, num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = config.hidden_layer_size
        D_out = config.dim_output if is_last_layer else config.hidden_layer_size

        # Pass in kernels, specify output dim (shared hyperparams/variables)

        inducing_var = construct_basic_distributional_inducing_variables(
            num_inducing=config.num_inducing, input_dim=D_in, share_variables=True, name = f"dist_gp_{i_layer}"
        )

        kernel = construct_basic_hybrid_kernel  (
            kernels=_construct_hybrid_kernel(D_in, is_last_layer, name = f"dist_gp_{i_layer}"
        ),
            output_dim=D_out,
            share_hyperparams=True,
            name = f"dist_gp_{i_layer}"
        )

        assert config.whiten is True, "non-whitened case not implemented yet"

        if is_last_layer:
            mean_function = Zero()
            q_sqrt_scaling = 1.0
            
        else:
            mean_function = construct_mean_function(X_running, D_out)
            #NOTE -- I think this would only work for constant input-dim DGP architectures
            #X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = X_running.numpy()
            q_sqrt_scaling = config.inner_layer_qsqrt_factor
            
        layer = DistGPLayer(
            kernel,
            inducing_var,
            num_data,
            num_latent_gps=D_out,
            mean_function=mean_function,
            name=f"dist_gp_{i_layer}",
        )
        layer.q_sqrt.assign(layer.q_sqrt * q_sqrt_scaling)
        gp_layers.append(layer)

    if config.task_type=="regression":
        likelihood = Gaussian(config.likelihood_noise_variance)
    elif config.task_type=='classification' and config.dim_output==1:
        likelihood = Bernoulli()
    elif config.task_type=="classification" and config.dim_output>1:
        likelihood = MultiClass(config.dim_output)
    else:
        raise WarningMessage("wrong specification for likelihood")

    return DistDeepGP(gp_layers, LikelihoodLayer(likelihood), num_data = num_data)
