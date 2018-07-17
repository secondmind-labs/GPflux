#  Adaptation from Mark van der Wilk

import numpy as np
import tensorflow as tf

from gpflow import features, settings, params_as_tensors
from gpflow.conditionals import conditional, sample_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero
from gpflow.params import Parameter, Parameterized

jitter_level = settings.numerics.jitter_level
float_type = settings.float_type


class BaseLayer(Parameterized):
    def __init__(self):
        Parameterized.__init__(self)

    def propagate(self, X, sampling=True):
        """
        :param X: tf.Tensor
            N x P
        :param sampling: bool
           If `True` returns a sample from the predictive distribution
           If `False` returns the mean and variance of the predictive distribution
        :return: If `sampling` is True, then the functions returns a tf.Tensor
            of shape N x W x H x C, else N x (W x H x C) x (W x H x C)  # TODO: think about this??
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
                 q_mu=None, q_sqrt=None, mean_function=None, Z=None):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        Note that the we don't bother subtracting the mean from the q_mu i.e. the variational
        distribution is centered.

        The layer holds D_out independent GPs with the same kernel and inducing points.

        Note that the mean function is not identical over layers, e.g. if mean function is
        the identity and D_in = D_out

        :param kern: The kernel for the layer (input_dim = D_in)
        :param q_mu: Variational mean initialization (M x D_out)
        :param q_sqrt: Variational cholesky of variance initialization (M x M x D_out)
        :param Z: Inducing points (M, D_in)
        :param mean_function: The mean function (e.g. Linear in the )
        :return:
        """
        BaseLayer.__init__(self)
        self.feature = features.inducingpoint_wrapper(feature, Z)
        self.kern = kern
        self.mean_function = mean_function or Zero()

        M, L = len(self.feature), num_latents
        self.num_latens = num_latents
        q_mu = np.zeros((M, L)) if q_mu is None else q_mu
        q_sqrt = np.tile(np.eye(M), (L, 1, 1)) if q_sqrt is None else q_sqrt
        self.q_mu = Parameter(q_mu, dtype=float_type)
        self.q_sqrt = Parameter(q_sqrt, dtype=float_type)

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
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I),
        independently for each GP
        """
        return gauss_kl(self.q_mu, self.q_sqrt)


    def describe(self):
        """ describes the key properties of a layer """

        return "GPLayer: kern {}, features {}, mean {}, L {}".\
                format(self.kern.__class__.__name__,
                       self.feature.__class__.__name__,
                       self.mean_function.__class__.__name__,
                       self.num_latens)

