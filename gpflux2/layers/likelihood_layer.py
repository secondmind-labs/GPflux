# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A Keras Layer that wraps a likelihood, while containing the necessary operations
for training"""
import tensorflow as tf
from tensorflow.keras.backend import learning_phase

from gpflow.likelihoods import Likelihood
from gpflow import default_float

from gpflux2.layers import TrackableLayer


class LikelihoodLayer(TrackableLayer):
    """
    A layer which wraps a GPflow likelihood, while providing a clear interface to
    help training.
    """

    def __init__(self, likelihood: Likelihood):
        super().__init__(dtype=default_float())
        self.likelihood = likelihood

    def call(self, inputs, training=None, targets=None, **kwargs):
        """TO DO: Note that this function returns a lot of different things. This func
        returns at different times, an expectation, a log probability, a mean and
        variance, and just the input!  This function needs some thought.
        """
        if training is None:
            training = learning_phase()

        if training:
            if isinstance(inputs, tuple):
                F_mu, F_var = inputs
                return self.likelihood.variational_expectations(F_mu, F_var, targets)
            else:
                return self.likelihood.log_prob(inputs, targets)
        else:
            if isinstance(inputs, tuple):
                F_mu, F_var = inputs
                return self.likelihood.predict_mean_and_var(F_mu, F_var)
            else:
                return tf.identity(inputs) # TO DO: should not be identity!
