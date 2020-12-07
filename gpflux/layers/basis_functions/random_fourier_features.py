#  Copyright (C) PROWLER.io 2020 - All Rights Reserved
#  Unauthorised copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
from typing import Mapping, Tuple, Type

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import TensorType

from gpflux.types import ShapeType

# Supported kernels for Random Fourier Features (RFF).
# RFF can be built for stationary kernels (shift invariant) for which we can
# sample frequencies from their power spectrum.
RFF_SUPPORTED_KERNELS: Tuple[Type[gpflow.kernels.Stationary], ...] = (
    gpflow.kernels.SquaredExponential,
    gpflow.kernels.Matern12,
    gpflow.kernels.Matern32,
    gpflow.kernels.Matern52,
)


class RandomFourierFeatures(tf.keras.layers.Layer):
    """
    Random Fourier Features (RFF) is a method for approximating kernels. The essential
    element of the RFF approach [1] is the realization that Bochner's theorem
    for stationary kernels can be approximated by a Monte Carlo sum.

    We will approximate the kernel k(x, x′) by ϕ(x)ᵀϕ(x′) where ϕ: χ → ℝˡ is a finite dimensional
    feature map. Each feature is defined as:
        ϕ(x) = √(2σ² / ℓ) · cos(θᵀx + 𝜏)
    where σ² is the kernel variance. The features are parameterised by random weights:
    * θ - sampled proportional to the kernel's spectral density
    * 𝜏 ∼ 𝒰(0, 2π)

    [1] Random Features for Large-Scale Kernel Machines, Ali Rahimi and Ben Recht
        https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
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
        input_dim = input_shape[-1]

        if not isinstance(self.kernel, gpflow.kernels.SquaredExponential):
            tf.assert_equal(input_dim, 1, "Matern kernels only support 1 dimensional inputs")

        shape_bias = [1, self.output_dim]
        self.b = self._sample_bias(shape_bias, dtype=self.dtype)
        shape_weights = [self.output_dim, input_dim]
        self.W = self._sample_weights(shape_weights, dtype=self.dtype)
        super().build(input_shape)

    def _sample_bias(self, shape: ShapeType, **kwargs: Mapping) -> TensorType:
        return tf.random.uniform(shape=shape, maxval=2 * np.pi, **kwargs)

    def _sample_weights(self, shape: ShapeType, **kwargs: Mapping) -> TensorType:
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
            # TODO(VD): sample directly from Student-t using TFP.
            normal_rvs = tf.random.normal(shape=shape, **kwargs)
            gamma_rvs = tf.random.gamma(shape=shape, alpha=nu, beta=nu, **kwargs)
            return tf.math.rsqrt(gamma_rvs) * normal_rvs

    def call(self, inputs: TensorType) -> TensorType:
        """
        Evaluates the basis functions at `inputs`

        :param inputs: evaluation points [N, D]

        :return: [N, M]
        """
        c = tf.sqrt(2 * self.kernel.variance / self.output_dim)
        inputs = tf.divide(inputs, self.kernel.lengthscales)  # [N, D]
        basis_functions = tf.cos(tf.matmul(inputs, self.W, transpose_b=True) + self.b)  # [N, M]
        output = c * basis_functions  # [N, M]
        tf.ensure_shape(output, self.compute_output_shape(inputs.shape))
        return output

    def compute_output_shape(self, input_shape: ShapeType) -> tf.TensorShape:
        tensor_shape = tf.TensorShape(input_shape).with_rank(2)
        return tensor_shape[:-1].concatenate(self.output_dim)

    def get_config(self) -> Mapping:
        config = super().get_config()
        config.update({"kernel": self.kernel, "output_dim": self.output_dim})

        return config
