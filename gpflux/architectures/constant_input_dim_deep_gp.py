# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""
Deep GP where each hidden layer has the same input dimensionality as the data.
"""

from dataclasses import dataclass

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
from gpflux.layers.gp_layer import GPLayer
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.models import DeepGP


@dataclass
class Config:
    num_inducing: int
    inner_layer_qsqrt_factor: float
    between_layer_noise_variance: float
    likelihood_noise_variance: float
    whiten: bool = True


def construct_kernel(dim: int, is_last_layer: bool) -> SquaredExponential:
    """
    Returns a SquaredExponential kernel with ARD lengthscales of 2 and a small kernel variance
    (1e-6) if the kernel is part of a hidden layer, otherwise the kernel variance is set to 1.
    """
    variance = 1e-6 if not is_last_layer else 1.0
    lengthscales = [2.0] * dim
    return SquaredExponential(lengthscales=lengthscales, variance=variance)


def build_constant_input_dim_deep_gp(X: np.ndarray, num_layers: int, config: Config) -> DeepGP:
    """
    Builds a DGP consisting of *num_layers* layers,
    where the hidden layers have the same dimension as the input.

    The architecture is largely based on the model presented in
    "Doubly Stochastic Deep Gaussian processes" (Salimbeni and Deisenroth, 2017).
    The most notable difference is that here we keep the hidden dimension equal
    to the input dimensionality of the data.

    :param X: training input data, used to retrieve the number of datapoints and input dimension.
    :param num_layers: Number of layers in the returned deep GP.
    :param config: configuration for (hyper)parameters.
    """
    num_data, input_dim = X.shape
    X_running = X

    gp_layers = []
    centroids, _ = kmeans2(X, k=config.num_inducing, minit="points")

    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else input_dim

        # Pass in kernels, specify output dim (shared hyperparams/variables)

        inducing_var = construct_basic_inducing_variables(
            num_inducing=config.num_inducing, input_dim=D_in, share_variables=True, z_init=centroids
        )

        kernel = construct_basic_kernel(
            kernels=construct_kernel(D_in, is_last_layer),
            output_dim=D_out,
            share_hyperparams=True,
        )

        assert config.whiten is True, "non-whitened case not implemented yet"

        if is_last_layer:
            mean_function = gpflow.mean_functions.Zero()
            q_sqrt_scaling = 1.0
        else:
            mean_function = construct_mean_function(X_running, D_in, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = X_running.numpy()
            q_sqrt_scaling = config.inner_layer_qsqrt_factor

        layer = GPLayer(
            kernel,
            inducing_var,
            num_data,
            mean_function=mean_function,
            name=f"gp_{i_layer}",
        )
        layer.q_sqrt.assign(layer.q_sqrt * q_sqrt_scaling)
        gp_layers.append(layer)

    likelihood = Gaussian(config.likelihood_noise_variance)
    return DeepGP(gp_layers, LikelihoodLayer(likelihood))
