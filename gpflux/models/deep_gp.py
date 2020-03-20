# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import tensorflow as tf

from gpflux.layers import GPLayer


class DeepGP(tf.keras.Model):
    def __init__(self, gp_layers, likelihood_layer):
        super().__init__()
        for gp_layer in gp_layers:
            assert isinstance(gp_layer, GPLayer)

        self.gp_layers = gp_layers
        self.likelihood_layer = likelihood_layer

    def elbo(self, data):
        """This is just a wrapper for explanatory value. Calling the model with training=True
        returns the predictive means and variances, but calculates the ELBO in the losses
        of the model"""

        # TF quirk: It looks like "__call__" does extra work before calling "call". We want this.
        _ = self.__call__(data, training=True)
        return -tf.reduce_sum(self.losses * self.gp_layers[0].num_data)

    def predict_f(self, inputs, training=None):
        intermediate_output = inputs
        for layer in self.gp_layers:
            intermediate_output = layer(intermediate_output, training=training)
        return intermediate_output

    def predict_y(self, inputs, training=None):
        if training:
            raise NotImplementedError("use __call__ for training")
        final_gp_output = self.predict_f(inputs)
        final_output = self.likelihood_layer(final_gp_output)
        return final_output

    def call(self, data, training=None):
        inputs, targets = data

        final_gp_output = self.predict_f(inputs, training)
        final_output = self.likelihood_layer(
            final_gp_output, targets=targets, training=training
        )

        return final_output
