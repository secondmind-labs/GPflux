# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""Latent variable layer for deep GPs"""

from typing import Optional, Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import default_float, Parameter

from gpflux.layers import TrackableLayer


class LatentVariableLayer(TrackableLayer):
    """
    A latent variable layer, with amortized mean-field VI.

    The latent variable is distribution agnostic, but assumes a variational posterior
    that is fully factorised and is of the same distribution family as the prior.
    """

    def __init__(
        self, encoder: tf.keras.Model, prior: tfp.distributions.Distribution,
    ):
        """
        :param prior: A distribution representing the prior over the latent variable
        :param encoder:  A tf.keras.Model which gets passed the arguments to the call of the
            LatentVariableLayer and returns the appropriate parameters to the approximate
            posterior distribution.
        """

        super().__init__(dtype=default_float())
        self.encoder = encoder
        self.prior = prior
        self.distribution = prior.__class__

    def build(self, input_shape):
        """Build the encoder and shapes necessary on first call"""
        super().build(input_shape)
        self.encoder.build(input_shape)

    def sample_posteriors(
        self,
        recognition_data,
        num_samples: Optional[int] = None,
        training: bool = False,
        seed: int = None,
    ):
        """
        Draw samples from the posterior disributions, given data for the encoder layer,
        and also return those distributions.

        :param recognition_data: data input for the encoder
        :param num_samples: number of samples
        :param training: training flag (for encoder)
        :param seed: random seed to sample with

        :return: (samples, posteriors)
        """
        distributions_params = self.encoder(recognition_data, training)

        posteriors = self.distribution(*distributions_params, allow_nan_stats=False)

        if num_samples is None:  # mimic numpy - take one sample and squeeze
            samples = posteriors.sample(1, seed=seed)
            samples = tf.squeeze(samples, axis=0)  # [N, D]
        else:
            samples = posteriors.sample(num_samples, seed=seed)  # [S, N, D]
        return samples, posteriors

    def sample_prior(self, sample_shape):
        """
        Draw samples from the prior.

        :param sample_shape: shape to draw from prior
        """
        return self.prior.sample(sample_shape)

    def call(
        self, recognition_data, training: bool = False, seed: int = None,
    ):
        """
        When training: draw a sample of the latent variable from the posterior,
        whose distribution is parameterized by the encoder mapping from the data.
        Add a KL divergence posterior||prior to the losses.

        When not training: draw a sample of the latent variable from the prior.

        :param recognition_data: the data inputs to the encoder network (if training).
            if not training - a single tensor, with leading dimensions indicating the
            number of samples to draw from the prior
        :param training: training mode indicator

        :return: samples of the latent variable
        """
        if training:
            samples, posteriors = self.sample_posteriors(
                recognition_data, num_samples=None, training=training, seed=seed
            )
            loss = tf.reduce_mean(self.local_kls(posteriors), name="local_kls")
        else:
            sample_shape = tf.shape(recognition_data)[:-1]
            samples = self.sample_prior(sample_shape)
            loss = tf.constant(0.0, dtype=tf.float64)

        self.add_loss(loss, inputs=True)  # for `inputs`
        # see https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss
        self.add_metric(loss, name="local_kl", aggregation="mean")
        return samples

    def local_kls(self, posteriors: tfp.distributions.Distribution):
        """
        The KL divergences [approximate posterior||prior]

        :param posteriors: a distribution representing the approximate posteriors

        :return: tensor with the KL divergences for each of the posteriors
        """
        return posteriors.kl_divergence(self.prior)
