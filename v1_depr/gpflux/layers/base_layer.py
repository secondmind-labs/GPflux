# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import NamedTuple, Optional

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import Param, Parameterized, params_as_tensors, settings
from gpflow.conditionals import sample_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero

from gpflux.nonstationary import NonstationaryKernel
from gpflux.types import TensorLike


class LayerOutput(NamedTuple):
    """
    Return type of the `propagate()` function of a layer.
    Contains all the outputs of a layer's propagate.

    Note: the shape of the tensors is given in-line,
        '...' stands for arbitrary leading dimensions.
    """
    mean: TensorLike  # [..., N, D]
    covariance: TensorLike  # [..., N, D, D]
    # a sample drawn from N(`mean`, `covariance`)
    sample: TensorLike  # [..., N, D]
    # KL for the whole process
    # Typically the KL between the prior and posterior of u.
    global_kl: TensorLike  # scalar
    # per datapoint KL term
    # Typically a KL that depends on latents for each datapoint.
    local_kl: TensorLike  # [..., N]


class AbstractLayer(Parameterized):

    def propagate(self, H, **kwargs):
        """
        :param H: tf.Tensor
            N x D
        :return: a sample from, mean, and covariance of the predictive
           distribution, all of shape N x P
           where P is of size W x H x C (in the case of images)
        """
        raise NotImplementedError()  # pragma: no cover

    def describe(self):
        """ describes the key properties of a layer """
        raise NotImplementedError()  # pragma: no cover


class GPLayer(AbstractLayer):
    def __init__(self,
                 kern: gpflow.kernels.Kernel,
                 feature: gpflow.features.InducingFeature,
                 num_latent: int,
                 q_mu: Optional[np.ndarray] = None,
                 q_sqrt: Optional[np.ndarray] = None,
                 mean_function: Optional[gpflow.mean_functions.MeanFunction] = None):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = L v + mean_function(X), where v ~ N(0, I) and L Lᵀ = kern.K(X)

        The variational distribution over the whitened inducing function values is
        q(v) = N(q_mu, q_sqrt q_sqrtᵀ)

        The layer holds num_latent independent GPs, potentially with different kernels or
        different inducing inputs.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param feature: inducing features
        :param num_latent: number of latent GPs in the layer
        :param q_mu: Variational mean initialization (M x num_latent)
        :param q_sqrt: Variational Cholesky factor of variance initialization (num_latent x M x M)
        :param mean_function: The mean function that links inputs to outputs
                (e.g., linear mean function)
        """
        super().__init__()
        self.feature = feature
        self.kern = kern
        self.mean_function = Zero() if mean_function is None else mean_function

        M = len(self.feature)
        self.num_latent = num_latent
        q_mu = np.zeros((M, num_latent)) if q_mu is None else q_mu
        q_sqrt = np.tile(np.eye(M), (num_latent, 1, 1)) if q_sqrt is None else q_sqrt
        self.q_mu = Param(q_mu, dtype=settings.float_type)
        self.q_sqrt = Param(q_sqrt, dtype=settings.float_type)

    @params_as_tensors
    def propagate(self, H, *, full_output_cov=False, full_cov=False, num_samples=None, **kwargs):
        """
        :param H: input to this layer [N, P]
        """
        mean_function = self.mean_function(H)
        sample, mean, cov = sample_conditional(
            H,
            self.feature,
            self.kern,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            white=True,
            num_samples=num_samples
        )
        return LayerOutput(
            mean=mean + mean_function,
            covariance=cov,
            sample=sample + mean_function,
            global_kl=self._KL(),
            local_kl=tf.cast(0.0, settings.float_type)
        )

    @params_as_tensors
    def _KL(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        return gauss_kl(self.q_mu, self.q_sqrt)

    def describe(self):
        """ returns a string with the key properties of a GPlayer """
        return "GPLayer: kern {}, features {}, mean {}, L {}".format(
            self.kern.__class__.__name__,
            self.feature.__class__.__name__,
            self.mean_function.__class__.__name__,
            self.num_latent
        )


class NonstationaryGPLayer(GPLayer):
    def __init__(self,
                 kernel: NonstationaryKernel,
                 feature: gpflow.features.InducingFeature,
                 *args, **kwargs):
        assert isinstance(kernel, NonstationaryKernel)
        if isinstance(feature, gpflow.features.InducingPoints):
            assert feature.Z.shape[1] == kernel.input_dim
        super().__init__(kernel, feature, *args, **kwargs)

    @params_as_tensors
    def propagate(self, H, *, X=None, **kwargs):
        """
        Concatenates original input X with output H of the previous layer
        for the non-stationary kernel: the latter will be interpreted as
        lengthscales.

        :param H: input to this layer [N, P]
        :param X: original input [N, D]
        """
        XH = tf.concat([X, H], axis=1)
        return super().propagate(XH, **kwargs)
