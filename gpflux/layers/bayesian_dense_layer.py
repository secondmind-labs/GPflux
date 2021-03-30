#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""A Bayesian Dense Keras Layer"""

from typing import Callable, Optional, Union

import numpy as np
import tensorflow as tf

from gpflow import Parameter, default_float
from gpflow.base import TensorType
from gpflow.kullback_leiblers import gauss_kl
from gpflow.models.model import MeanAndVariance
from gpflow.utilities.bijectors import positive, triangular

from gpflux.helpers import xavier_initialization_numpy
from gpflux.layers.trackable_layer import TrackableLayer
from gpflux.types import ShapeType


class BayesianDenseLayer(TrackableLayer):
    """A Bayesian dense layer for variational Bayesian neural networks"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_data: int,
        w_mu: Optional[np.ndarray] = None,
        w_sqrt: Optional[np.ndarray] = None,
        activation: Optional[Callable] = None,
        is_mean_field: bool = True,
        temperature: float = 1e-4,
        returns_samples: bool = True,
    ):
        """
        A Bayesian dense layer for variational Bayesian neural nets. This layer holds the
        weight mean and sqrt as well as the temperature for cooling (or heating) the posterior.

        :param input_dim: The layer's input dimension (excluding bias)
        :param output_dim: The layer's output dimension
        :param num_data: number of data points
        :param w_mu: Initial value of the variational mean (weights + bias)
        :param w_sqrt: Initial value of the variational Cholesky (covering weights + bias)
        :param activation: The type of activation function (None is linear)
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
                assert w_sqrt.shape == (
                    (input_dim + 1) * output_dim,
                    (input_dim + 1) * output_dim,
                )
            else:
                assert w_sqrt.shape == ((input_dim + 1) * output_dim,)
        assert temperature > 0.0

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_data = num_data

        self.w_mu_ini = w_mu
        self.w_sqrt_ini = w_sqrt

        self.activation = activation
        self.is_mean_field = is_mean_field
        self.temperature = temperature
        self.returns_samples = returns_samples

        self.dim = (input_dim + 1) * output_dim
        self.full_output_cov = False
        self.full_cov = False

        self.w_mu = Parameter(np.zeros((self.dim,)), dtype=default_float(), name="w_mu")  # [dim]

        self.w_sqrt = Parameter(
            np.zeros((self.dim, self.dim)) if not self.is_mean_field else np.ones((self.dim,)),
            transform=triangular() if not self.is_mean_field else positive(),
            dtype=default_float(),
            name="w_sqrt",
        )  # [dim, dim] or [dim]

    def initialize_variational_distribution(self) -> None:
        if self.w_mu_ini is None:
            w = xavier_initialization_numpy(self.input_dim, self.output_dim)
            b = np.zeros((1, self.output_dim))
            self.w_mu_ini = np.concatenate((w, b), axis=0).reshape((self.dim,))
        self.w_mu.assign(self.w_mu_ini)

        if self.w_sqrt_ini is None:
            if not self.is_mean_field:
                self.w_sqrt_ini = 1e-5 * np.eye(self.dim)
            else:
                self.w_sqrt_ini = 1e-5 * np.ones((self.dim,))
        self.w_sqrt.assign(self.w_sqrt_ini)

    def build(self, input_shape: ShapeType) -> None:
        """Build the variables necessary on first call"""
        super().build(input_shape)
        self.initialize_variational_distribution()

    def predict_samples(
        self,
        inputs: TensorType,
        *,
        num_samples: Optional[int] = None,
        full_output_cov: bool = False,
        full_cov: bool = False,
        whiten: bool = False,
    ) -> tf.Tensor:
        """
        Make a sample predictions at N test inputs, with input_dim = D, output_dim = Q. Return a
        sample, and the conditional mean and covariance at these points.

        :param inputs: the inputs to predict at. shape [N, D]
        :param num_samples: the number of samples S, to draw.
            shape [S, N, Q] if S is not None else [N, Q].
        :param full_output_cov: assert to False since not supported for now
        :param full_cov: assert to False since not supported for now
        :param whiten: assert to False since not sensible in Bayesian neural nets
        """
        assert full_output_cov is False
        assert full_cov is False
        assert whiten is False

        _num_samples = num_samples or 1
        z = tf.random.normal((self.dim, _num_samples), dtype=default_float())  # [dim, S]
        if not self.is_mean_field:
            w = self.w_mu[:, None] + tf.matmul(self.w_sqrt, z)  # [dim, S]
        else:
            w = self.w_mu[:, None] + self.w_sqrt[:, None] * z  # [dim, S]

        N = tf.shape(inputs)[0]
        inputs_concat_1 = tf.concat(
            (inputs, tf.ones((N, 1), dtype=default_float())), axis=-1
        )  # [N, D+1]
        samples = tf.tensordot(
            inputs_concat_1,
            tf.reshape(tf.transpose(w), (_num_samples, self.input_dim + 1, self.output_dim)),
            [[-1], [1]],
        )  # [N, S, Q]
        if num_samples is None:
            samples = tf.squeeze(samples, axis=-2)  # [N, Q]
        else:
            samples = tf.transpose(samples, perm=[1, 0, 2])  # [S, N, Q]

        if self.activation is not None:
            samples = self.activation(samples)

        return samples

    def call(
        self, inputs: TensorType, training: Optional[bool] = False
    ) -> Union[tf.Tensor, MeanAndVariance]:
        """The default behaviour upon calling the BayesianDenseLayer()(X)"""
        sample = self.predict_samples(
            inputs,
            num_samples=None,
            full_output_cov=self.full_output_cov,
            full_cov=self.full_cov,
        )

        # TF quirk: add_loss must add a tensor to compile
        if training:
            # Wenzel et al. 2020: How good is the Bayes posterior in DNNs really?
            loss = self.temperature * self.prior_kl()
        else:
            loss = tf.constant(0.0, dtype=default_float())
        loss_per_datapoint = loss / self.num_data

        self.add_loss(loss_per_datapoint)

        # for latent layers, return samples
        if self.returns_samples:
            return sample  # [N, Q]
        # for output layers, return samples as mean with 0 cov
        return sample, tf.ones_like(sample) * 1e-10  # [N, Q], [N, Q]

    def prior_kl(self) -> tf.Tensor:
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(w_mu, w_sqrt) to N(0, I)
        """
        return gauss_kl(
            self.w_mu[:, None],
            self.w_sqrt[None] if not self.is_mean_field else self.w_sqrt[:, None],
        )
