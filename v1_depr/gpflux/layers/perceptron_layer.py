# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import tensorflow as tf

from gpflow import params_as_tensors, settings

from gpflux.layers import LayerOutput, LinearLayer


class PerceptronLayer(LinearLayer):
    """
    Performs a linear transformation and can potentially
    pass the output through a non-linear activation function.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation=None):
        super().__init__(input_dim, output_dim)
        self.activation = activation

    @params_as_tensors
    def propagate(self, H, **kwargs):
        mean = super().propagate(H).mean  # linear_transformation

        if self.activation is not None:
            mean = self.activation(mean)

        return LayerOutput(
            mean=mean,
            sample=mean,
            covariance=None,
            global_kl=tf.cast(0.0, settings.float_type),
            local_kl=tf.cast(0.0, settings.float_type),

        )
