#  Adaptation from Mark van der Wilk

import numpy as np
import tensorflow as tf

from gpflow import features, settings, params_as_tensors
from gpflow.conditionals import conditional, sample_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero
# from gpflow.params import Parameter, Parameterized
from gpflow import Param, Parameterized

jitter_level = settings.numerics.jitter_level
float_type = settings.float_type


class BaseLayer(Parameterized):

    def propagate(self, X, sampling=True):
        """
        :param X: tf.Tensor
            N x D
        :param sampling: bool
           If `True` returns a sample from the predictive distribution
           If `False` returns the mean and variance of the predictive distribution
        :return: If `sampling` is True, then the function returns a tf.Tensor
            of shape N x P, else N x P for the mean and N x P for the variance
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
    def __init__(self, kern, feature, num_latents,
                 q_mu=None, q_sqrt=None, mean_function=None):
        r"""
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = L v + mean_function(X), where v ~ N(0, I) and L Lᵀ = kern.K(X)

        The variational distribution over the whitened inducing function values is
        q(v) = N(q_mu, q_sqrt q_sqrtᵀ)

        The layer holds num_latents independent GPs, potentially with different kernels or
        different inducing inputs.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param q_mu: Variational mean initialization (M x num_latents)
        :param q_sqrt: Variational Cholesky factor of variance initialization (M x M x num_latents)
        :param Z: Inducing points (M, D_in)
        :param mean_function: The mean function that links inputs to outputs (e.g.,
                              linear mean function)
        """
        super().__init__()
        self.feature = feature
        self.kern = kern
        self.mean_function = Zero() if mean_function is None else mean_function

        M = len(self.feature)
        self.num_latents = num_latents
        q_mu = np.zeros((M, num_latents)) if q_mu is None else q_mu
        q_sqrt = np.tile(np.eye(M), (num_latents, 1, 1)) if q_sqrt is None else q_sqrt
        self.q_mu = Param(q_mu, dtype=float_type)
        self.q_sqrt = Param(q_sqrt, dtype=float_type)

    @params_as_tensors
    def propagate(self, X, *, sampling=True, full_output_cov=False, full_cov=False):
        """
        :param X: N x P
        """
        if sampling:
            sample = sample_conditional(X, self.feature, self.kern,
                                        self.q_mu, q_sqrt=self.q_sqrt,
                                        full_output_cov=full_output_cov, white=True)
            return sample + self.mean_function(X)  # N x P
        else:
            mean, var = conditional(X, self.feature, self.kern, self.q_mu,
                                    q_sqrt=self.q_sqrt, full_cov=full_cov,
                                    full_output_cov=full_output_cov, white=True)
            return mean + self.mean_function(X), var  # N x P, variance depends on args

    @params_as_tensors
    def KL(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        return gauss_kl(self.q_mu, self.q_sqrt)


    def describe(self):
        """ returns a string with the key properties of a GPlayer """

        return "GPLayer: kern {}, features {}, mean {}, L {}".\
                format(self.kern.__class__.__name__,
                       self.feature.__class__.__name__,
                       self.mean_function.__class__.__name__,
                       self.num_latents)

