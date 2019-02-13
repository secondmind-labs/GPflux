# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from enum import Enum

import tensorflow as tf

from gpflow import params_as_tensors
from gpflow.kullback_leiblers import gauss_kl
from gpflux.layers.layers import BaseLayer
from gpflux.encoders import RecognitionNetwork


class LatentVarMode(Enum):
    """
    We need to distinguish between training and test points
    when propagating with latent variables. We have
    a parameterized variational posterior for the N data,
    but at test points we might want to do one of three things:
    """
    # sample from N(0, 1)
    PRIOR = 1

    # we are dealing with the N observed data points,
    # so we use the variational posterior
    POSTERIOR = 2

    # for plotting purposes, it is useful to have a mechanism
    # for setting W to fixed values, e.g. on a grid
    GIVEN = 3


class LatentVariableLayer(BaseLayer):
    """
    A latent variable layer, with amortized mean-field VI

    The prior is N(0, 1), and the approximate posterior is factorized
    into N(a, b), where a, b come from an encoder network.

    When propagating there are two possibilities:
    1) We're doing inference, so we use the variational distribution
    2) We're looking at test points, so we use the prior
    """

    # TODO(vincent) 2) does not seem to match up with the LatentVarMode doc string

    def __init__(self, latent_dim, XY_dim=None, encoder=None):
        BaseLayer.__init__(self)
        self.latent_dim = latent_dim

        if encoder is None:
            assert XY_dim, 'must pass XY_dim or else an encoder'
            encoder = RecognitionNetwork(latent_dim, XY_dim, [10, 10])

        self.encoder = encoder
        self.is_encoded = False

    def encode_once(self):
        if not self.is_encoded:
            XY = tf.concat([self.root.X, self.root.Y], 1)
            q_mu, log_q_sqrt = self.encoder(XY)
            self.q_mu = q_mu
            self.q_sqrt = tf.nn.softplus(log_q_sqrt - 3.)  # bias it towards small vals at first
            self.is_encoded = True

    def KL(self):
        self.encode_once()
        return gauss_kl(self.q_mu, self.q_sqrt)

    def propagate(self, X, W=None, **kwargs):
        raise NotImplementedError()

    def describe(self):
        return "{} with latent dim {}".format(
            self.__class__.__name__,
            self.latent_dim
        )


class LatentVariableConcatLayer(LatentVariableLayer):
    """
    A latent variable layer where the latents are concatenated with the input
    """

    @params_as_tensors
    def propagate(self,
                  X,
                  *,
                  W=None,
                  full_cov=False,
                  full_output_cov=False,
                  latent_var_mode=LatentVarMode.POSTERIOR):

        self.encode_once()

        if latent_var_mode == LatentVarMode.POSTERIOR:
            eps = tf.random_normal(tf.shape(self.q_mu), dtype=X.dtype)  # [N, L]
            W = self.q_mu + eps * self.q_sqrt  # [N, L]

            XW_mean = tf.concat([X, self.q_mu], 1)  # [N, D + L]
            XW_var = tf.concat([tf.zeros_like(X), self.q_sqrt ** 2], 1)  # [N, D + L]

        elif latent_var_mode == LatentVarMode.PRIOR:
            W = tf.random_normal([tf.shape(X)[0], self.latent_dim], dtype=X.dtype)

            zeros = tf.zeros([tf.shape(X)[0], self.latent_dim], dtype=X.dtype)
            ones = tf.ones([tf.shape(X)[0], self.latent_dim], dtype=X.dtype)
            XW_mean = tf.concat([X, zeros], 1)  # [N, D + L]
            XW_var = tf.concat([tf.zeros_like(X), ones], 1)  # [N, D + L]

        elif latent_var_mode == LatentVarMode.GIVEN:
            assert isinstance(W, tf.Tensor)

        else:
            raise NotImplementedError

        sample = tf.concat([X, W], 1)

        if latent_var_mode == LatentVarMode.GIVEN:
            XW_mean = sample
            XW_var = None

        return sample, XW_mean, XW_var
