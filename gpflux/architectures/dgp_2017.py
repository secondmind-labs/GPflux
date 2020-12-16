# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

"""
Implements the Doubly Stochastic Deep Gaussian Process by Salimbeni & Deisenroth (2017)
http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorflow as tf

from gpflow.kernels import SquaredExponential, White
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Identity, Linear, Zero
from gpflow.utilities import set_trainable

from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.initializers import FeedForwardInitializer, KmeansInitializer
from gpflux.layers.gp_layer import GPLayer
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.models import DeepGP


@dataclass
class Config:
    num_inducing: int
    inner_layer_qsqrt_factor: float
    between_layer_noise_variance: float
    likelihood_noise_variance: float
    white: bool = True


def kernel_factory_2017(dim: int, is_last_layer: bool, config: Config):
    kernel = SquaredExponential(lengthscales=[2.0] * dim, variance=2.0)
    if not is_last_layer:
        kernel += White(config.between_layer_noise_variance)
    return kernel


def construct_mean_function(X, D_in, D_out):
    assert X.shape[-1] == D_in
    if D_in == D_out:
        mean_function = Identity()
    else:
        if D_in > D_out:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            W = V[:D_out, :].T
        else:
            W = np.concatenate([np.eye(D_in), np.zeros((D_in, D_out - D_in))], axis=1)

        assert W.shape == (D_in, D_out)
        mean_function = Linear(W)
        set_trainable(mean_function, False)

    return mean_function


def build_deep_gp_2017(X: np.ndarray, layer_dims: Sequence[int], config: Config):
    num_data, input_dim = X.shape

    assert layer_dims[0] == input_dim
    assert layer_dims[-1] == 1
    assert len(layer_dims) >= 2

    gp_layers = []

    X_running = X

    for i_layer, (D_in, D_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        is_first_layer = i_layer == 0
        is_last_layer = i_layer == len(layer_dims) - 2

        # Pass in kernels, specify output dim (shared hyperparams/variables)

        inducing_var = construct_basic_inducing_variables(
            num_inducing=config.num_inducing, input_dim=D_in, share_variables=True,
        )

        kernel = construct_basic_kernel(
            kernels=kernel_factory_2017(D_in, is_last_layer, config),
            output_dim=D_out,
            share_hyperparams=True,
        )

        assert config.white is True, "non-whitened case not implemented yet"

        if is_last_layer:
            mean_function = Zero()
            q_sqrt_scaling = 1
        else:
            mean_function = construct_mean_function(X_running, D_in, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = X_running.numpy()
            q_sqrt_scaling = config.inner_layer_qsqrt_factor

        if is_first_layer:
            initializer = KmeansInitializer(X, num_inducing=config.num_inducing)
        else:
            initializer = FeedForwardInitializer()

        layer = GPLayer(kernel, inducing_var, num_data, initializer, mean_function=mean_function)
        layer.q_sqrt.assign(layer.q_sqrt * q_sqrt_scaling)
        gp_layers.append(layer)

    likelihood = Gaussian(config.likelihood_noise_variance)
    return DeepGP(gp_layers, likelihood_layer=LikelihoodLayer(likelihood))
