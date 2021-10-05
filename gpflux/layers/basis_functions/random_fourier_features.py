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

from typing import Mapping, Optional

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import DType, TensorType

from gpflux.layers.basis_functions.utils import (
    RFF_SUPPORTED_KERNELS,
    _mapping,
    _matern_number,
    _sample_students_t,
)
from gpflux.types import ShapeType

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
        if kwargs.get("input_dim", None):
            self._input_dim = kwargs["input_dim"]
            self.build(tf.TensorShape([self._input_dim]))
        else:
            self._input_dim = None

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        input_dim = input_shape[-1]

        shape_bias = [1, self.output_dim]
        self.b = self.add_weight(
            name="bias",
            trainable=False,
            shape=shape_bias,
            dtype=self.dtype,
            initializer=self.bias_init,
        )

        shape_weights = [self.output_dim, input_dim]
        self.W = self.add_weight(
            name="weights",
            trainable=False,
            shape=shape_weights,
            dtype=self.dtype,
            initializer=self.weights_init,
        )

        super().build(input_shape)

    def bias_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        return tf.random.uniform(shape=shape, maxval=2.0 * np.pi, dtype=dtype)

    def weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        if isinstance(self.kernel, gpflow.kernels.SquaredExponential):
            return tf.random.normal(shape, dtype=dtype)
        else:
            p = _matern_number(self.kernel)
            nu = 2.0 * p + 1.0  # degrees of freedom
            return _sample_students_t(nu, shape, dtype)

    def call(self, inputs: TensorType) -> tf.Tensor:
        """
        Evaluate the basis functions at ``inputs``.

        :param inputs: The evaluation points, a tensor with the shape ``[N, D]``.

        :return: A tensor with the shape ``[N, M]``.
        """
        output = _mapping(
            inputs,
            self.W,
            self.b,
            variance=self.kernel.variance,
            lengthscales=self.kernel.lengthscales,
            n_components=self.output_dim,
        )
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
        config.update(
            {"kernel": self.kernel, "output_dim": self.output_dim, "input_dim": self._input_dim}
        )

        return config
