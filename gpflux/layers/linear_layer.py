import tensorflow as tf
import numpy as np
import gpflow

from gpflow import params_as_tensors
from gpflow.params import Parameter

from ..utils import xavier_weights
from .layers import BaseLayer

class LinearLayer(BaseLayer):
    """
    Performs a deterministic linear transformation.
    """
    # TODO: probabilistic linear transformation

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.weight = Parameter(xavier_weights(input_dim, output_dim))
        self.bias = Parameter(np.zeros((output_dim)))

    @params_as_tensors
    def propagate(self, X, sampling=True, **kwargs):
        if not sampling:
            raise ValueError("We can only sample from a single "
                             "layer multi-perceptron.")

        return tf.matmul(X, self.weight) + self.bias

    def KL(self):
        return tf.cast(0.0, gpflow.settings.float_type)

    def describe(self):
        return "LinearLayer: input_dim {}, output_dim {}".\
                format(self.input_dim, self.output_dim)
