# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import Optional

import gpflow
import numpy as np
from gpflow import Param, Parameterized, features, params_as_tensors, settings
from gpflow.conditionals import conditional, sample_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero
from ..nonstationary import NonstationaryKernel
import tensorflow as tf


class BaseLayer(Parameterized):

    def propagate(self, H, **kwargs):
        """
        :param H: tf.Tensor
            N x D
        :return: a sample from, mean, and covariance of the predictive
           distribution, all of shape N x P
           where P is of size W x H x C (in the case of images)
        """
        raise NotImplementedError()  # pragma: no cover

    def KL(self):
        """ returns KL[q(U) || p(U)] """
        raise NotImplementedError()  # pragma: no cover

    def describe(self):
        """ describes the key properties of a layer """
        raise NotImplementedError()  # pragma: no cover


class GPLayer(BaseLayer):
    def __init__(self,
                 kern: gpflow.kernels.Kernel,
                 feature: gpflow.features.InducingFeature,
                 num_latents: int,
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

        The layer holds num_latents independent GPs, potentially with different kernels or
        different inducing inputs.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param feature: inducing features
        :param num_latents: number of latent GPs in the layer
        :param q_mu: Variational mean initialization (M x num_latents)
        :param q_sqrt: Variational Cholesky factor of variance initialization (num_latents x M x M)
        :param mean_function: The mean function that links inputs to outputs
                (e.g., linear mean function)
        """
        super().__init__()
        self.feature = feature
        self.kern = kern
        self.mean_function = Zero() if mean_function is None else mean_function

        M = len(self.feature)
        self.num_latents = num_latents
        q_mu = np.zeros((M, num_latents)) if q_mu is None else q_mu
        q_sqrt = np.tile(np.eye(M), (num_latents, 1, 1)) if q_sqrt is None else q_sqrt
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
        return sample + mean_function, mean + mean_function, cov

    @params_as_tensors
    def KL(self):
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
            self.num_latents
        )


class NonstationaryGPLayer(GPLayer):
    def __init__(self,
                 kernel: NonstationaryKernel,
                 *args, **kwargs):
        assert isinstance(kernel, NonstationaryKernel)
        super().__init__(kernel, *args, **kwargs)

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
