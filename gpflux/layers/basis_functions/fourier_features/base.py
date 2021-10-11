from abc import ABC, abstractmethod
from typing import Mapping

import tensorflow as tf

import gpflow
from gpflow.base import TensorType

from gpflux.types import ShapeType


class FourierFeaturesBase(ABC, tf.keras.layers.Layer):
    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        """
        :param kernel: kernel to approximate using a set of random features.
        :param output_dim: total number of basis functions used to approximate
            the kernel.
        """
        super(FourierFeaturesBase, self).__init__(**kwargs)
        self.kernel = kernel
        self.n_components = n_components  # M: number of Monte Carlo samples
        if kwargs.get("input_dim", None):
            self._input_dim = kwargs["input_dim"]
            self.build(tf.TensorShape([self._input_dim]))
        else:
            self._input_dim = None

    def call(self, inputs: TensorType) -> tf.Tensor:
        """
        Evaluate the basis functions at ``inputs``.

        :param inputs: The evaluation points, a tensor with the shape ``[N, D]``.

        :return: A tensor with the shape ``[N, M]``.
        """
        const = self._compute_constant()
        bases = self._compute_bases(inputs)
        output = const * bases
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
        output_dim = self._compute_output_dim(input_shape)
        return tensor_shape[:-1].concatenate(output_dim)

    def get_config(self) -> Mapping:
        """
        Returns the config of the layer.
        See `tf.keras.layers.Layer.get_config()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config>`_.
        """
        config = super(FourierFeaturesBase, self).get_config()
        config.update(
            {"kernel": self.kernel, "n_components": self.n_components, "input_dim": self._input_dim}
        )

        return config

    @abstractmethod
    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        pass

    @abstractmethod
    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.
        """
        pass

    @abstractmethod
    def _compute_bases(self, inputs: TensorType) -> tf.Tensor:
        """
        Compute basis functions.
        """
        pass
