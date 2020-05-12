# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A Keras Layer that wraps a likelihood, while containing the necessary operations
for training"""
import tensorflow as tf

from gpflow.likelihoods import Likelihood
from gpflow import default_float

from gpflux.layers import TrackableLayer


class LikelihoodLayer(TrackableLayer):
    """
    A layer which wraps a GPflow likelihood, while providing a clear interface to
    help training.
    """

    def __init__(self, likelihood: Likelihood):
        super().__init__(dtype=default_float())
        self.likelihood = likelihood

    def call(self, inputs, training=False, targets=None):
        """Note that this function can operate both on tuples of (mean, variance), and also simply
        samples.
        """

        use_mean_and_cov = isinstance(inputs, tuple)
        if use_mean_and_cov:
            F_mu, F_var = inputs

            # TF quirk: add_loss must add a tensor to compile
            if training:
                losses = -self.variational_expectations(F_mu, F_var, targets)
            else:
                losses = tf.zeros((1, 1), dtype=default_float())

            # Scale by batch size, and sum over output dimensions
            loss_per_datapoint = tf.reduce_sum(tf.reduce_mean(losses, axis=0))
            self.add_loss(loss_per_datapoint)
            self.add_metric(loss_per_datapoint, name="elbo_datafit", aggregation="mean")

            return self.predict_mean_and_var(F_mu, F_var)
        else:
            samples = inputs

            # TF quirk: add_loss must add a tensor to compile
            if training:
                losses = -self.log_prob(samples, targets)
            else:
                losses = tf.zeros((1, 1), dtype=default_float())

            # Scale by batch size, and sum over output dimensions
            loss_per_datapoint = tf.reduce_sum(tf.reduce_mean(losses, axis=0))
            self.add_loss(loss_per_datapoint)
            self.add_metric(loss_per_datapoint, name="elbo_datafit", aggregation="mean")

            # TO DO: should not be identity - we should sample through the likelihood
            return tf.identity(inputs)

    def log_prob(self, samples, targets):
        """A wrapper around the gpflow.Likelihood method"""
        return self.likelihood.log_prob(samples, targets)

    def variational_expectations(self, F_mu, F_var, targets):
        """A wrapper around the gpflow.Likelihood method"""
        return self.likelihood.variational_expectations(F_mu, F_var, targets)

    def predict_mean_and_var(self, F_mu, F_var):
        """A wrapper around the gpflow.Likelihood method"""
        return self.likelihood.predict_mean_and_var(F_mu, F_var)
