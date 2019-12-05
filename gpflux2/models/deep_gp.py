# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import tensorflow as tf
import numpy as np

from gpflux2.layers import GPLayer

class DeepGP(tf.keras.Model):
    def __init__(self, gp_layers, likelihood_layer):
        super().__init__()
        for gp_layer in gp_layers:
            assert isinstance(gp_layer, GPLayer)

        self.gp_layers = gp_layers
        self.likelihood_layer = likelihood_layer

    def elbo(self, X, Y):
        variational_expectations = self.call(X, targets=Y, training=True)
        variational_expectation = tf.reduce_sum(variational_expectations)

        kl_divergence = np.sum(self.losses)
        elbo = variational_expectation - kl_divergence
        return elbo

    def call(self, inputs, training=None, targets=None):
        if training:
            assert targets is not None

        intermediate_output = inputs
        for layer in self.gp_layers:
            intermediate_output = layer(intermediate_output, training=training)

        final_output = self.likelihood_layer(
            intermediate_output, features=inputs, targets=targets, training=training
        )

        return final_output
