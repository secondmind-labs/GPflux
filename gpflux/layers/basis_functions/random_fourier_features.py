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
""" A kernel's features and coefficients using Random Fourier Features (RFF). """

from typing import Mapping, Tuple, Type

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import TensorType

from gpflux.types import ShapeType

RFF_SUPPORTED_KERNELS: Tuple[Type[gpflow.kernels.Stationary], ...] = (
    gpflow.kernels.SquaredExponential,
    gpflow.kernels.Matern12,
    gpflow.kernels.Matern32,
    gpflow.kernels.Matern52,
)
"""
Kernels supported by :class:`RandomFourierFeatures`.

You can build RFF for shift-invariant stationary kernels from which you can
sample frequencies from their power spectrum, following Bochner's theorem.
"""


class RandomFourierFeatures(tf.keras.layers.Layer):
    r"""
    Random Fourier Features (RFF) is a method for approximating kernels. The essential
    element of the RFF approach :cite:p:`rahimi2007random` is the realization that Bochner's theorem
    for stationary kernels can be approximated by a Monte Carlo sum.

    We will approximate the kernel :math:`k(x, x')` by :math:`\Phi(x)^\top \Phi(x')`
    where :math:`Phi: x \to \mathbb{R}` is a finite-dimensional feature map.
    Each feature is defined as:

    .. math:: \Phi(x) = \sqrt{2 \sigma^2 / \ell) \cos(\theta^\top x + \tau)

    where :math:`\sigma^2` is the kernel variance.

    The features are parameterised by random weights:
    * :math:`\theta`, sampled proportional to the kernel's spectral density
    * :math:`\tau \sim \mathcal{U}(0, 2\pi)`
    """

    def __init__(self, kernel: gpflow.kernels.Kernel, output_dim: int, **kwargs: Mapping):
        """
        :param kernel: kernel to approximate using a set of random features.
        :param output_dim: total number of basis functions used to approximate
            the kernel.
        """
        super().__init__(**kwargs)

        self.kernel = kernel
        assert isinstance(self.kernel, RFF_SUPPORTED_KERNELS), "Unsupported Kernel"
        self.output_dim = output_dim  # M

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        input_dim = input_shape[-1]

        shape_bias = [1, self.output_dim]
        self.b = self._sample_bias(shape_bias, dtype=self.dtype)
        shape_weights = [self.output_dim, input_dim]
        self.W = self._sample_weights(shape_weights, dtype=self.dtype)
        super().build(input_shape)

    def _sample_bias(self, shape: ShapeType, **kwargs: Mapping) -> tf.Tensor:
        return tf.random.uniform(shape=shape, maxval=2 * np.pi, **kwargs)

    def _sample_weights(self, shape: ShapeType, **kwargs: Mapping) -> tf.Tensor:
        if isinstance(self.kernel, gpflow.kernels.SquaredExponential):
            return tf.random.normal(shape, **kwargs)
        else:
            if isinstance(self.kernel, gpflow.kernels.Matern52):
                nu = 2.5
            elif isinstance(self.kernel, gpflow.kernels.Matern32):
                nu = 1.5
            elif isinstance(self.kernel, gpflow.kernels.Matern12):
                nu = 0.5
            else:
                raise NotImplementedError("Unsupported Kernel")

            # Sample student-t using "Implicit Reparameterization Gradients",
            # Figurnov et al.
            normal_rvs = tf.random.normal(shape=shape, **kwargs)
            shape = tf.concat([shape[:-1], [1]], axis=0)
            gamma_rvs = tf.tile(tf.random.gamma(shape, alpha=nu, beta=nu, **kwargs), [1, shape[-1]])
            return tf.math.rsqrt(gamma_rvs) * normal_rvs

    def call(self, inputs: TensorType) -> tf.Tensor:
        """
        Evaluate the basis functions at ``inputs``.

        :param inputs: The evaluation points, a tensor with the shape ``[N, D]``.

        :return: A tensor with the shape ``[N, M]``.
        """
        c = tf.sqrt(2 * self.kernel.variance / self.output_dim)
        inputs = tf.divide(inputs, self.kernel.lengthscales)  # [N, D]
        basis_functions = tf.cos(tf.matmul(inputs, self.W, transpose_b=True) + self.b)  # [N, M]
        output = c * basis_functions  # [N, M]
        tf.ensure_shape(output, self.compute_output_shape(inputs.shape))
        return output

    def compute_output_shape(self, input_shape: ShapeType) -> tf.TensorShape:
        """
        Computes the output shape of the layer.
        See `tf.keras.layers.Layer.compute_output_shape()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#compute_output_shape>`_.
        """
        # TODO: Keras docs say "If the layer has not been built, this method
        # will call `build` on the layer." -- do we need to do so?
        tensor_shape = tf.TensorShape(input_shape).with_rank(2)
        return tensor_shape[:-1].concatenate(self.output_dim)

    def get_config(self) -> Mapping:
        """
        Returns the config of the layer.
        See `tf.keras.layers.Layer.get_config()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config>`_.
        """
        config = super().get_config()
        config.update({"kernel": self.kernel, "output_dim": self.output_dim})

        return config
