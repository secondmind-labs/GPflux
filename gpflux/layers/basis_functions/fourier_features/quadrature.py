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
""" A kernel's features and coefficients using quadrature Fourier features (QFF). """

from typing import Mapping

import tensorflow as tf

import gpflow
from gpflow.base import TensorType
from gpflow.quadrature.gauss_hermite import ndgh_points_and_weights

from gpflux.layers.basis_functions.fourier_features.utils import (
    QFF_SUPPORTED_KERNELS,
    _mapping_concat,
)
from gpflux.types import ShapeType


class QuadratureFourierFeatures(tf.keras.layers.Layer):
    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        """
        :param kernel: kernel to approximate using a set of random features.
        :param output_dim: total number of basis functions used to approximate
            the kernel.
        """
        super().__init__(**kwargs)

        assert isinstance(kernel, QFF_SUPPORTED_KERNELS), "Unsupported Kernel"
        self.kernel = kernel
        self.n_components = n_components  # M: number of quadrature points
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
        abscissa_value, weights_value = ndgh_points_and_weights(
            dim=input_dim, n_gh=self.n_components
        )

        # Quadrature node points
        self.abscissa = tf.Variable(initial_value=abscissa_value, trainable=False)
        # Gauss-Hermite weights (not to be confused with random Fourier feature
        # weights or NN weights)
        self.weights = tf.Variable(initiial_value=weights_value, trainable=False)
        super().build(input_shape)

    def compute_output_shape(self, input_shape: ShapeType) -> tf.TensorShape:
        """
        Computes the output shape of the layer.
        See `tf.keras.layers.Layer.compute_output_shape()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#compute_output_shape>`_.
        """
        # TODO: Keras docs say "If the layer has not been built, this method
        # will call `build` on the layer." -- do we need to do so?
        input_dim = input_shape[-1]
        output_dim = 2 * self.n_components ** input_dim
        tensor_shape = tf.TensorShape(input_shape).with_rank(2)
        return tensor_shape[:-1].concatenate(output_dim)

    def get_config(self) -> Mapping:
        """
        Returns the config of the layer.
        See `tf.keras.layers.Layer.get_config()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config>`_.
        """
        config = super().get_config()
        config.update(
            {"kernel": self.kernel, "n_components": self.n_components, "input_dim": self._input_dim}
        )

        return config

    def call(self, inputs: TensorType) -> tf.Tensor:
        """
        Evaluate the basis functions at ``inputs``.

        :param inputs: The evaluation points, a tensor with the shape ``[N, D]``.

        :return: A tensor with the shape ``[N, 2M^D]``.
        """
        const = tf.tile(tf.sqrt(self.kernel.variance * self.weights), multiples=[2])
        bases = _mapping_concat(inputs, self.abscissa, lengthscales=self.kernel.lengthscales)
        output = const * bases
        tf.ensure_shape(output, self.compute_output_shape(inputs.shape))
        return output
