# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidentialimport numpy as np


import tensorflow as tf
import numpy as np
import gpflow

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
        self.input_dim, self.output_dim = input_dim, output_dim

        if weight is None:
            self.weight = Param(xavier_weights(input_dim, output_dim))
        else:
            self.weight = weight
        self.bias = Param(np.zeros((output_dim))) if bias is None else bias


    @params_as_tensors
    def propagate(self, X, sampling=True, **kwargs):
        if not sampling:
            raise ValueError("We can only sample from a single "
                             "layer multi-perceptron.")

        return tf.matmul(X, self.weight) + self.bias

    def KL(self):
        return tf.cast(0.0, settings.float_type)

    def describe(self):
        return "LinearLayer: input_dim {}, output_dim {}".\
                format(self.input_dim, self.output_dim)
