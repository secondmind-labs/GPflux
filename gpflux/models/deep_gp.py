# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import tensorflow as tf

from gpflux.layers import GPLayer
from gpflux.models.bayesian_model import BayesianModel


class DeepGP(BayesianModel):
    def __init__(self, gp_layers, likelihood_layer, input_dim=None, output_dim=None):
        X_input = tf.keras.Input((input_dim,))
        Y_input = tf.keras.Input((output_dim,))

        F_output = X_input
        for gp_layer in gp_layers:
            assert isinstance(gp_layer, GPLayer)
            F_output = gp_layer(F_output)
        num_data = gp_layers[0].num_data

        super().__init__(X_input, Y_input, F_output, likelihood_layer, num_data)

        self.gp_layers = gp_layers
