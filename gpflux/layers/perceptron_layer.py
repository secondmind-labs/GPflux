# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from gpflow import params_as_tensors

from gpflux.layers.linear_layer import LinearLayer


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
    def propagate(self, X, **kwargs):
        sample, mean, cov = super().propagate(X)  # linear_transformation

        if self.activation is not None:
            sample = self.activation(sample)
        return sample, sample, None
