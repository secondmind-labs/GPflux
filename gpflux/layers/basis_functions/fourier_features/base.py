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
""" Shared functionality for stationary kernel basis functions. """

from abc import ABC, abstractmethod
from itertools import cycle
from typing import Mapping, Optional

import tensorflow as tf

import gpflow
from gpflow.base import TensorType
from gpflow.keras import tf_keras

from gpflux.types import ShapeType


class FourierFeaturesBase(ABC, tf_keras.layers.Layer):
    r"""
    The base class for all Fourier feature layers, used for both random Fourier feature layers and
    quadrature layers. We subclass :class:`tf.keras.layers.Layer`, so we must provide
    :method:`build` and :method:`call` methods.
    """

    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        """
        :param kernel: kernel to approximate using a set of Fourier bases.
        :param n_components: number of components (e.g. Monte Carlo samples,
            quadrature nodes, etc.) used to numerically approximate the kernel.
        """
        super(FourierFeaturesBase, self).__init__(**kwargs)
        self.kernel = kernel
        self.n_components = n_components
        if isinstance(kernel, gpflow.kernels.MultioutputKernel):
            self.is_batched = True
            self.is_multioutput = True
            self.batch_size = kernel.num_latent_gps
            self.sub_kernels = kernel.latent_kernels
        elif isinstance(kernel, gpflow.kernels.Combination):
            self.is_batched = True
            self.is_multioutput = False
            self.batch_size = len(kernel.kernels)
            self.sub_kernels = kernel.kernels
        else:
            self.is_batched = False
            self.is_multioutput = False
            self.batch_size = 1
            self.sub_kernels = []

        if kwargs.get("input_dim", None):
            self._input_dim = kwargs["input_dim"]
            self.build(tf.TensorShape([self._input_dim]))
        else:
            self._input_dim = None

    def call(self, inputs: TensorType) -> tf.Tensor:
        """
        Evaluate the basis functions at ``inputs``.

        :param inputs: The evaluation points, a tensor with the shape ``[N, D]``.

        :return: A tensor with the shape ``[N, M]``, or shape ``[P, N, M]'' in the multioutput case.
        """
        if self.is_batched:
            bases = [
                # restrict inputs to the appropriate active_dims for each sub_kernel
                self._compute_bases(tf.divide(k.slice(inputs, None)[0], k.lengthscales), i)
                # SharedIndependent repeatedly uses the same sub_kernel
                for i, k in zip(range(self.batch_size), cycle(self.sub_kernels))
            ]
            bases = tf.stack(bases, axis=0)  # [P, N, M]
        else:
            # restrict inputs to the kernel's active_dims
            X = tf.divide(self.kernel.slice(inputs, None)[0], self.kernel.lengthscales)  # [N, D]
            bases = self._compute_bases(X, None)  # [N, M]

        const = self._compute_constant()  # [] or [P, 1, 1]
        output = const * bases

        if self.is_batched and not self.is_multioutput:
            # For combination kernels, remove batch dimension and instead concatenate into the
            # feature dimension.
            output = tf.transpose(output, perm=[1, 2, 0])  # [N, M, P]
            output = tf.reshape(output, [tf.shape(output)[0], -1])  # [N, M*P]

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
        output_dim = self.compute_output_dim(input_shape)
        trailing_shape = tensor_shape[:-1].concatenate(output_dim)
        if self.is_multioutput:
            return tf.TensorShape([self.batch_size]).concatenate(trailing_shape)  # [P, N, M]
        else:
            return trailing_shape  # [N, M] or [N, M*P]

    def get_config(self) -> Mapping:
        """
        Returns the config of the layer.
        See `tf.keras.layers.Layer.get_config()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config>`_.
        """
        config = super(FourierFeaturesBase, self).get_config()
        config.update(
            {
                "kernel": self.kernel,
                "n_components": self.n_components,
                "input_dim": self._input_dim,
            }
        )

        return config

    @abstractmethod
    def compute_output_dim(self, input_shape: ShapeType) -> int:
        """
        Compute the output dimension of the layer.
        """
        pass

    @abstractmethod
    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.
        """
        pass

    @abstractmethod
    def _compute_bases(self, inputs: TensorType, batch: Optional[int]) -> tf.Tensor:
        """
        Compute basis functions.

        For batched layers (self.is_batched), batch indicates which sub-kernel to target.
        """
        pass
