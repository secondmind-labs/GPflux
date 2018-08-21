# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf
from gpflow import (Param, Parameterized, ParamList, params_as_tensors,
                    settings, transforms)

from .utils import xavier_weights


class Encoder(Parameterized):
    """
    Abstract base class for an Encoder, which produces the mean and
    variance (or [log-]standard deviation) of the latent variable
    associated to a data point.
    """

    def __init__(self,
                 latent_dim: int,
                 name: Optional[str] = None):
        """
        :param latent_dim: dimensionality of the latent variable
        """
        Parameterized.__init__(self, name=name)
        self.latent_dim = latent_dim

    def __call__(self, Z: tf.Tensor) -> None:
        """
        Given Z, returns the mean and the log of the Cholesky
        of the latent variables (only the diagonal elements)
        In other words, w_n ~ N(m_n, exp(s_n)), where m_n, s_n = f(x_n).
        For this Encoder the function f is a NN.
        :return: N x latent_dim, N x latent_dim
        """
        #TODO(vincent) doc string not appropriate for abstract base class - move to RecognitionNetwork?
        raise NotImplementedError()


class RecognitionNetwork(Encoder):
    def __init__(self,
                 latent_dim: int,
                 input_dim: int,
                 network_dims: List[int],
                 activation_func = None,
                 name: Optional[str] = None):
        """
        Encoder that uses GPflow params to encode the features.
        Creates an MLP with input dimensions `input_dim` and produces
        2 * `latent_dim` outputs.
        :param latent_dim: dimension of the latent variable
        :param input_dim: the MLP acts on data of `input_dim` dimensions
        :param network_dims: dimensions of inner MLPs, e.g. [10, 20, 10]
        :param activation_func: TensorFlow operation that can be used
            as non-linearity between the layers (default: tanh).
        """
        Encoder.__init__(self, latent_dim, name=name)

        self.input_dim = input_dim
        self.network_dims = network_dims
        self.activation_func = tf.nn.tanh if activation_func is None else activation_func
        self._build_network()

    def _build_network(self):
        Ws, bs = [], []
        dims = [self.input_dim, *self.network_dims, self.latent_dim * 2]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            Ws.append(Param(xavier_weights(dim_in, dim_out)))
            bs.append(Param(np.zeros(dim_out)))

        self.Ws, self.bs = ParamList(Ws), ParamList(bs)

    @params_as_tensors
    def __call__(self, Z: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            Z = tf.matmul(Z, W) + b
            if i < len(self.bs) - 1:
                Z = self.activation_func(Z)

        means, log_chol_diag = tf.split(Z, 2, axis=1)
        return means, log_chol_diag


class DirectlyParameterized(Encoder):
    """
    No amortization is used; each datapoint element has an
    associated mean and variance of its latent variable.

    IMPORTANT: Not compatible with minibatches
    """

    def __init__(self,
                 latent_dim: int,
                 num_data: int,
                 mean: Optional[np.array] = None,
                 name: Optional[str] = None):
        Encoder.__init__(self, latent_dim, name=name)

        self.num_data = num_data
        if mean is None:
            mean = np.random.randn(num_data, latent_dim)
        if mean.shape != (num_data, latent_dim):
            raise ValueError("mean must have shape (num_data={}, latent_dim={})"
                             .format(num_data, latent_dim))
        self.mean = Param(mean)
        self.std = Param(1e-5 * np.ones((num_data, latent_dim)),
                         transform=transforms.positive)

    @params_as_tensors
    def __call__(self, Z: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        return self.mean, self.std
