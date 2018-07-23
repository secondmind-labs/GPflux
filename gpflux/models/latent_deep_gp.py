import gpflow
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag

from functools import reduce
from typing import Union, Optional, List

from gpflow import Param, ParamList
from gpflow.decors import params_as_tensors, autoflow
from gpflow.kullback_leiblers import gauss_kl
from gpflow.likelihoods import Gaussian
from gpflow.models.model import Model
from gpflow.params.dataholders import Minibatch, DataHolder
from gpflow.conditionals import _sample_mvn as sample_mvn
# from gpflow.quadrature import ndiag_mc, ndiagquad

from ..utils import lrelu, xavier_weights
from .encoders import Encoder

float_type = gpflow.settings.float_type
int_type = gpflow.settings.int_type
jitter_level = gpflow.settings.numerics.jitter_level


class LatentDeepGP(Model):

    def __init__(self,
                 X: np.ndarray,
                 encoder: Encoder,
                 layers: List,
                 likelihood: Optional[gpflow.likelihoods.Likelihood] = None,
                 batch_size: int = 32,
                 beta: float = 1.0,
                 name: Optional[str] = None) -> None:
        """
        :param X: np.ndarray
            Image dataset of size N x P, TODO: type (float32 or float64)
        :param encoder: Encoder
            Used as inference network
        :param layers: List
            List of `layers.BaseLayer` instances, e.g. PerceptronLayer,
            ConvLayer, GPLayer,...
        :param likelihood: gpflow.likelihoods.Likelihood object
            Analytic expressions exists for the Gaussian case.
        :param batch_size: int
            Size of the batch for stochastic optimization of the ELBO
        :param beta: float
            Scaler of the latent's KL.
        :param name: str (Optional)
        """
        Model.__init__(self, name=name)

        assert X.ndim == 2

        if (batch_size > 0) and (batch_size is not None):
            self.X = Minibatch(X, batch_size=batch_size, seed=0)
            self.scale = X.shape[0] / batch_size
        else:
            self.X = DataHolder(X)
            self.scale = 1.0

        self.encoder = encoder
        self.num_data = X.shape[0]
        self.X_dim = X.shape[1]
        self.layers = ParamList(layers)
        self.likelihood = likelihood or Gaussian()
        self.beta = Param(beta)
        self.beta.set_trainable(False)


    def _build_decoder(self,
                       Z: tf.Tensor,
                       full_cov_output: Optional[bool] = False) -> tf.Tensor:
        """
        Propagates a single sample Z through the layers of the model.
        :param Z: N x W
        :return:
        """
        Z = tf.cast(Z, dtype=tf.float64)
        for layer in self.layers[:-1]:
            Z = layer.propagate(Z, sampling=True, full_output_cov=False, full_cov=False)

        f_mean, f_var = self.layers[-1].propagate(Z, sampling=False, full_cov=False,
                                                  full_output_cov=full_cov_output)
        return f_mean, f_var

    @params_as_tensors
    def _build_likelihood(self) -> tf.Tensor:
        """ let 1) N: batch size, 2) W: latent dimension """

        ## Encoder
        Z_mean, Z_diag_sqrt = self.encoder(self.X)  # N x W, N x W
        nu = tf.random_normal(tf.shape(Z_mean), dtype=float_type)
        Z_sample = Z_mean + nu * tf.exp(Z_diag_sqrt)  # N x W

        ## Decoder (R stands for Reconstruced X)
        f_mean, f_var = self._build_decoder(Z_sample)  # N x P, N x P

        ## Variational expectation (error)
        E_log_prob = tf.reduce_sum(self.likelihood.variational_expectations(f_mean, f_var, self.X))

        ## KLs
        KL_Z = gauss_kl(tf.matrix_transpose(Z_mean), tf.matrix_transpose(Z_diag_sqrt))
        KL_U_layers = reduce(tf.add, (l.KL() for l in self.layers))

        ELBO = (E_log_prob - self.beta * KL_Z) * self.scale - KL_U_layers

        return tf.cast(ELBO, float_type)


    @autoflow([float_type, [None, None]])
    def decode(self, Z):
        mean, var = self._build_decoder(Z, full_cov_output=True)
        return mean, var  # N x P, N x P x P

    @autoflow([float_type, [None, None]])
    def sample(self, Z):
        mean, var = self._build_decoder(Z, full_cov_output=True)  # N x P, N x P x P
        return sample_mvn(mean, var, "full")  # N x P

    @autoflow([float_type, [None, None]])
    def decode_and_sample(self, Z):
        mean, var = self._build_decoder(Z, full_cov_output=True)  # N x P, N x P x P
        return mean, var, sample_mvn(mean, var, "full")  # N x P, N x P x P, N x P

    def describe(self):
        """ High-level description of the model """
        s = self.__class__.__name__
        s += "\nLayers"
        s += "\n------\n"
        s += "\n".join(l.describe() for l in self.layers)
        s += "\nlikelihood: " + self.likelihood.__class__.__name__
        return s


class ConditionalLatentDeepGP(LatentDeepGP):

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 encoder: Encoder,
                 layers: List,
                 likelihood: Optional[gpflow.likelihoods.Likelihood] = None,
                 batch_size: int = 32,
                 beta: float = 1.0,
                 name: Optional[str] = None) -> None:
        """
        :param X: np.ndarray
            Image dataset of size N x P, TODO: type (float32 or float64)
        :param Y: np.ndarray
            Labels, typically one-hot encoding
        :param encoder: Encoder
            Used as inference network
        :param layers: List
            List of `layers.BaseLayer` instances, e.g. PerceptronLayer,
            ConvLayer, GPLayer,...
        :param likelihood: gpflow.likelihoods.Likelihood object
            Analytic expressions exists for the Gaussian case.
        :param batch_size: int
            Size of the batch for stochastic optimization of the ELBO
        :param beta: float
            Scaler of the latent's KL.
        :param name: str (Optional)
        """
        Model.__init__(self, name=name)

        assert X.ndim == 2

        if (batch_size is not None) and (batch_size > 0):
            self.X = Minibatch(X, batch_size=batch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=batch_size, seed=0)
            self.scale = X.shape[0] / batch_size
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)
            self.scale = 1.0

        self.encoder = encoder
        self.latent_dim = self.encoder.latent_dim
        self.num_data = X.shape[0]
        self.X_dim = X.shape[1]
        self.layers = ParamList(layers)
        self.likelihood = likelihood or Gaussian()
        self.beta = Param(beta)
        self.beta.set_trainable(False)


    @params_as_tensors
    def _build_likelihood(self) -> tf.Tensor:
        """ let 1) N: batch size, 2) W: latent dimension """

        # Concatenate labels and targets
        XY = tf.concat([self.X, self.Y], axis=1)

        ## Encoder
        Z_mean, Z_log_scale_diag = self.encoder(XY)  # N x W, N x W
        Z_scale_diag = tf.exp(Z_log_scale_diag)

        qZ = MultivariateNormalDiag(loc=Z_mean, scale_diag=Z_scale_diag)
        Z_sample = qZ.sample()

        # Concatenate labels with latent variable sample
        XZ_sample = tf.concat([self.X, Z_sample], axis=1)

        ## Decoder (R stands for Reconstruced X)
        f_mean, f_var = self._build_decoder(XZ_sample)  # N x P, N x P

        ## Variational expectation (error)
        self.E_log_prob = tf.reduce_sum(self.likelihood.variational_expectations(f_mean, f_var, self.Y))

        ## KLs
        pZ = MultivariateNormalDiag(loc=tf.zeros_like(Z_mean))  # scale = 1 by default
        self.KL_Z = tf.reduce_sum(qZ.kl_divergence(pZ))
        self.KL_U_layers = reduce(tf.add, (l.KL() for l in self.layers))

        ELBO = (self.E_log_prob - self.beta * self.KL_Z) * self.scale - self.KL_U_layers
        # ELBO = tf.Print(ELBO, ["elbo:", ELBO])
        return tf.cast(ELBO, float_type)

    # @autoflow([float_type, [None, None]])
    def decode(self, Z):
        latent_sample = tf.random_normal([tf.shape(Z)[0], self.latent_dim], dtype=float_type)
        Z = tf.concat([Z, latent_sample], axis=1)
        mean, var = self._build_decoder(Z, full_cov_output=False)
        # assert isinstance(self.model.likelihood, Gaussian)
        # var = var + self.likelihood.variance
        return mean, var  # N x P, N x P

    @autoflow([float_type, [None, None]])
    def predict_f(self, X):
        return self.decode(X)  # N x P, N x P

    @autoflow([float_type, [None, None]])
    def predict_y(self, X):
        return self.likelihood.predict_mean_and_var(*self.decode(X))  # N x P, N x P

    @autoflow((float_type, [None, None]),
              (float_type, [None, None]))
    def log_pdf(self, X, Y):

        def log_pdf_func(Z, X=None, Y=None):
            XZ = tf.concat([X, Z], axis=1)  # N x (D + L)
            mean, var = self._build_decoder(XZ, full_cov_output=False)  # N x P, N x P
            logp = self.likelihood.predict_density(mean, var, Y)  # N x 1
            return logp

        N = tf.shape(X)[0]
        Z_mu = tf.zeros((N, self.latent_dim), dtype=float_type)
        Z_var = tf.ones((N, self.latent_dim), dtype=float_type)
        # if self.latent_dim > 2:
            # evaluate using Monte-Carlo
        S = 1000
        return 0
        # return ndiag_mc(log_pdf_func, S, Z_mu, Z_var, logspace=True, X=X, Y=Y)
        # else:
        #     # evaluate using Quadrature, this is possible as the latent is 1 or 2 dimenional
        #     H = 100
        #     return ndiagquad(log_pdf_func, H, Z_mu, Z_var, logspace=True, X=X, Y=Y)

    @autoflow()
    def compute_KL_Z(self):
        return self.KL_Z * self.scale

    @autoflow()
    def compute_KL_U(self):
        return self.KL_U_layers

    @autoflow()
    def compute_data_fit(self):
        return self.E_log_prob * self.scale
