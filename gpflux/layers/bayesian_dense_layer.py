# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A Bayesian Dense Keras Layer"""

from typing import Optional, Callable

import numpy as np
import tensorflow as tf

from gpflow.kullback_leiblers import gauss_kl
from gpflow.utilities.bijectors import triangular, positive
from gpflow import default_float, Parameter

from gpflux.layers import TrackableLayer


class BayesianDenseLayer(TrackableLayer):
    """A Bayesian dense layer for variational Bayesian neural networks"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_data: int,
        w_mu: Optional[np.ndarray] = None,
        w_sqrt: Optional[np.ndarray] = None,
        activity_function: Optional[Callable] = None,
        is_mean_field: bool = True,
        temperature: float = 1e-4,
        returns_samples: bool = True
    ):
        """
        A Bayesian dense layer for variational Bayesian neural nets. This layer holds the
        weight mean and sqrt as well as the temperature for cooling (or heating) the posterior.

        :param input_dim: The layer's input dimension (excluding bias)
        :param output_dim: The layer's output dimension
        :param num_data: number of data points
        :param w_mu: Initial value of the variational mean (weights + bias)
        :param w_sqrt: Initial value of the variational Cholesky (covering weights + bias)
        :param activity_function: The type of activity function (None is linear)
        :param is_mean_field: Determines mean field approximation of the weight posterior
        :param temperature: For cooling or heating the posterior
        :param returns_samples: If True, return samples on calling the layer,
             Else return mean and variance
        """

        super().__init__(dtype=default_float())

        assert input_dim >= 1
        assert output_dim >= 1
        assert num_data >= 1
        if w_mu is not None:  # add + 1 for the bias
            assert w_mu.shape == ((input_dim + 1) * output_dim,)
        if w_sqrt is not None:
            if not is_mean_field:
                assert w_sqrt.shape == ((input_dim + 1) * output_dim, (input_dim + 1) * output_dim)
            else:
                assert w_sqrt.shape == ((input_dim + 1) * output_dim,)
        assert temperature > 0.0

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_data = num_data

        self.activity_function = activity_function
        self.is_mean_field = is_mean_field
        self.temperature = temperature
        self.returns_samples = returns_samples

        self.dim = (input_dim + 1) * output_dim
        self.full_output_cov = False
        self.full_cov = False

        if w_mu is None:
            w = np.random.randn(input_dim, output_dim) * (2. / (input_dim + output_dim)) ** 0.5
            b = np.zeros((1, output_dim))
            w_mu = np.concatenate((w, b), axis=0).reshape((self.dim,))
        self.w_mu = Parameter(
            w_mu[:, None],
            dtype=default_float(),
            name="w_mu"
        )  # [dim, 1]

        if w_sqrt is None:
            if not self.is_mean_field:
                w_sqrt = 1e-5 * np.eye(self.dim)
            else:
                w_sqrt = 1e-5 * np.ones((self.dim,))
        self.w_sqrt = Parameter(
            w_sqrt[None] if not self.is_mean_field else w_sqrt[:, None],
            transform=triangular() if not self.is_mean_field else positive(),
            dtype=default_float(),
            name="w_sqrt"
        )  # [1, dim, dim] or [dim, 1]

        self._initialized = False

    def build(self, input_shape):
        """Build the variables necessary on first call"""
        super().build(input_shape)
        self._initialized = True

    def predict(
        self,
        inputs,
        *,
        num_samples: Optional[int] = None,
        full_output_cov: bool = False,
        full_cov: bool = False,
        white: bool = False,
    ):
        """
        Make a sample predictions at N test inputs, with input_dim = D, output_dim = Q. Return a
        sample, and the conditional mean and covariance at these points.

        :param inputs: the inputs to predict at. shape [N, D]
        :param num_samples: the number of samples S, to draw.
            shape [S, N, Q] if S is not None else [N, Q].
        :param full_output_cov: assert to False since not supported for now
        :param full_cov: assert to False since not supported for now
        :param white: assert to False since not sensible in Bayesian neural nets
        """
        assert full_output_cov is False
        assert full_cov is False
        assert white is False

        _num_samples = num_samples or 1
        z = tf.random.normal((self.dim, _num_samples), dtype=default_float())  # [dim, S]
        if not self.is_mean_field:
            w = self.w_mu + tf.matmul(self.w_sqrt[0], z)  # [dim, S]
        else:
            w = self.w_mu + self.w_sqrt * z  # [dim, S]

        N = tf.shape(inputs)[0]
        inputs_concat_1 = tf.concat(
            (inputs, tf.ones((N, 1), dtype=default_float())), axis=1
        )  # [N, D+1]
        samples = tf.tensordot(
            inputs_concat_1,
            tf.reshape(tf.transpose(w), (_num_samples, self.input_dim + 1, self.output_dim)),
            [[-1], [1]]
        )  # [N, S, Q]
        if num_samples is None:
            samples = samples[:, 0, :]  # [N, Q]
        else:
            samples = tf.transpose(samples, perm=[1, 0, 2])  # [S, N, Q]

        if self.activity_function is not None:
            samples = self.activity_function(samples)

        # treat samples as mean with zero cov (mean and cov are not required by this use case)
        return samples, samples, tf.ones_like(samples) * 1e-10

    def call(self, inputs, training=False):
        """The default behaviour upon calling the BayesianDenseLayer()(X)"""
        samples, mean, cov = self.predict(
            inputs,
            num_samples=None,
            full_output_cov=self.full_output_cov,
            full_cov=self.full_cov,
        )

        # TF quirk: add_loss must add a tensor to compile, multiply with temperature
        loss = self.temperature * self.prior_kl() if training \
            else tf.constant(0.0, dtype=default_float())
        loss_per_datapoint = loss / self.num_data

        self.add_loss(loss_per_datapoint)

        if self.returns_samples:
            return samples  # for latent layers, return samples
        return mean, cov  # for output layers, return samples as mean with 0 cov

    def prior_kl(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(w_mu, w_sqrt) to N(0, I)
        """
        return gauss_kl(self.w_mu, self.w_sqrt)
