# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import tensorflow as tf
import numpy as np

from typing import Optional

from gpflow import params_as_tensors, Param, settings

from ..utils import xavier_weights
from .layers import BaseLayer


class LinearLayer(BaseLayer):
    """
    Performs a deterministic linear transformation.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 weight: Optional[np.ndarray] = None,
                 bias: Optional[np.ndarray] = None):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if weight is None:
            weight = xavier_weights(input_dim, output_dim)
        if weight.shape != (input_dim, output_dim):
            raise ValueError("weight must have shape (input_dim={}, output_dim={})"
                             .format(input_dim, output_dim))

        if bias is None:
            bias = np.zeros((output_dim, ))
        if bias.shape != (output_dim,):
            raise ValueError("bias must have length equal to output_dim={}".format(output_dim))

        self.weight = Param(weight)
        self.bias = Param(bias)

    @params_as_tensors
    def propagate(self, H, **kwargs):
        mean = tf.matmul(H, self.weight) + self.bias
        return mean, mean, None

    def KL(self):
        return tf.cast(0.0, settings.float_type)

    def describe(self):
        return "LinearLayer: input_dim {}, output_dim {}".format(
            self.input_dim,
            self.output_dim
        )
