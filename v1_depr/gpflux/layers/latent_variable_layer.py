# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from enum import Enum

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import params_as_tensors, settings
from gpflow.kullback_leiblers import gauss_kl

from gpflux.encoders import RecognitionNetwork
from gpflux.layers import AbstractLayer, LayerOutput


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


class LatentVariableLayer(AbstractLayer):
    """
    A latent variable layer, with amortized mean-field VI

    The prior is N(0, 1), and the approximate posterior is factorized
    into N(a, b), where a, b come from an encoder network.

    When propagating there are two possibilities:
    1) We're doing inference, so we use the variational distribution
    2) We're looking at test points, so we use the prior
    """

    def __init__(self, latent_dim, XY_dim=None, encoder=None):
        AbstractLayer.__init__(self)
        self.latent_dim = latent_dim

        if encoder is None:
            assert XY_dim, 'must pass XY_dim or else an encoder'
            encoder = RecognitionNetwork(latent_dim, XY_dim, [10, 10])

        self.encoder = encoder

    def propagate(self, H, X=None, Y=None, W=None, **kwargs):
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
    def propagate(
            self,
            H,
            *,
            X=None,
            Y=None,
            W=None,
            full_cov=False,
            full_output_cov=False,
            latent_var_mode=LatentVarMode.POSTERIOR
    ) -> LayerOutput:

        local_kl = None  # will be overwritten if layer is in POSTERIOR mode

        if latent_var_mode == LatentVarMode.POSTERIOR:
            if (X is None) or (Y is None):
                raise ValueError("LatentVariableLayer in POSTERIOR mode requires "
                                 "access to X and Y for use in the recognition network.")

            XY = tf.concat([X, Y], axis=-1)  # [..., N, D+Dy]
            q_mu, q_sqrt = self.encoder(XY)

            eps = tf.random_normal(tf.shape(q_mu), dtype=H.dtype)  # [N, L]
            W = q_mu + eps * q_sqrt  # [N, L]

            HW_mean = tf.concat([H, q_mu], 1)  # [N, D + L]
            HW_var = tf.concat([tf.zeros_like(H), q_sqrt ** 2], 1)  # [N, D + L]
            local_kl = gauss_kl(q_mu, q_sqrt)

            zero, one = [tf.cast(x, dtype=tf.float32) for x in [0, 1]]
            p = tfp.distributions.Normal(zero, one)
            q = tfp.distributions.Normal(q_mu, q_sqrt)  # scale, loc
            # we can't use GPflow's gauss_kl here as we don't
            # want to sum out the minibatch dimension.
            local_kl = tf.reduce_sum(
                tfp.distributions.kl_divergence(q, p), axis=-1)  # [..., N]

        elif latent_var_mode == LatentVarMode.PRIOR:
            W = tf.random_normal([tf.shape(H)[0], self.latent_dim], dtype=H.dtype)

            zeros = tf.zeros([tf.shape(H)[0], self.latent_dim], dtype=H.dtype)
            ones = tf.ones([tf.shape(H)[0], self.latent_dim], dtype=H.dtype)
            HW_mean = tf.concat([H, zeros], 1)  # [N, D + L]
            HW_var = tf.concat([tf.zeros_like(H), ones], 1)  # [N, D + L]

        elif latent_var_mode == LatentVarMode.GIVEN:
            assert isinstance(W, tf.Tensor)

        else:
            raise NotImplementedError

        sample = tf.concat([H, W], 1)

        if latent_var_mode == LatentVarMode.GIVEN:
            HW_mean = sample
            HW_var = None

        return LayerOutput(
            mean=HW_mean,
            covariance=HW_var,
            sample=sample,
            global_kl=tf.cast(0.0, settings.float_type),
            local_kl=local_kl
        )
