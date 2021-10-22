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
"""
This module provides a set of common utilities for kernel feature decompositions.
"""
from typing import Tuple, Type

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import DType, TensorType

from gpflux.types import ShapeType

"""
Kernels supported by :class:`QuadratureFourierFeatures`.

Currently we only support the :class:`gpflow.kernels.SquaredExponential` kernel.
For Matern kernels please use :class:`RandomFourierFeatures`
or :class:`RandomFourierFeaturesCosine`.
"""
QFF_SUPPORTED_KERNELS: Tuple[Type[gpflow.kernels.Stationary], ...] = (
    gpflow.kernels.SquaredExponential,
)

"""
Kernels supported by :class:`OrthogonalRandomFeatures`.

This random matrix sampling scheme only applies to the :class:`gpflow.kernels.SquaredExponential`
kernel.
For Matern kernels please use :class:`RandomFourierFeatures`
or :class:`RandomFourierFeaturesCosine`.
"""
ORF_SUPPORTED_KERNELS: Tuple[Type[gpflow.kernels.Stationary], ...] = (
    gpflow.kernels.SquaredExponential,
)

"""
Kernels supported by :class:`RandomFourierFeatures`.

You can build RFF for shift-invariant stationary kernels from which you can
sample frequencies from their power spectrum, following Bochner's theorem.
"""
RFF_SUPPORTED_KERNELS: Tuple[Type[gpflow.kernels.Stationary], ...] = (
    gpflow.kernels.SquaredExponential,
    gpflow.kernels.Matern12,
    gpflow.kernels.Matern32,
    gpflow.kernels.Matern52,
)


def _matern_number(kernel: gpflow.kernels.Kernel) -> int:
    if isinstance(kernel, gpflow.kernels.Matern52):
        p = 2
    elif isinstance(kernel, gpflow.kernels.Matern32):
        p = 1
    elif isinstance(kernel, gpflow.kernels.Matern12):
        p = 0
    else:
        raise NotImplementedError("Not a recognized Matern kernel")
    return p


def _sample_chi_squared(nu: float, shape: ShapeType, dtype: DType) -> TensorType:
    """
    Draw samples from Chi-squared distribution with `nu` degrees of freedom.

    See https://mathworld.wolfram.com/Chi-SquaredDistribution.html for further
    details regarding relationship to Gamma distribution.
    """
    return tf.random.gamma(shape=shape, alpha=0.5 * nu, beta=0.5, dtype=dtype)


def _sample_chi(nu: float, shape: ShapeType, dtype: DType) -> TensorType:
    """
    Draw samples from Chi-distribution with `nu` degrees of freedom.
    """
    s = _sample_chi_squared(nu, shape, dtype)
    return tf.sqrt(s)


def _sample_students_t(nu: float, shape: ShapeType, dtype: DType) -> TensorType:
    """
    Draw samples from a (central) Student's t-distribution using the following:
      BETA ~ Gamma(nu/2, nu/2) (shape-rate parameterization)
      X ~ Normal(0, 1/BETA)
    then:
      X ~ StudentsT(nu)

    Note this is equivalent to the more commonly used parameterization
      Z ~ Chi2(nu) = Gamma(nu/2, 1/2)
      EPSILON ~ Normal(0, 1)
      X = EPSILON * sqrt(nu/Z)

    To see this, note
      Z/nu ~ Gamma(nu/2, nu/2)
    and
      X ~ Normal(0, nu/Z)
    The equivalence becomes obvious when we set BETA = Z/nu
    """
    # Normal(0, 1)
    normal_rvs = tf.random.normal(shape=shape, dtype=dtype)
    shape = tf.concat([shape[:-1], [1]], axis=0)
    # Gamma(nu/2, nu/2)
    gamma_rvs = tf.random.gamma(shape, alpha=0.5 * nu, beta=0.5 * nu, dtype=dtype)
    # StudentsT(nu)
    students_t_rvs = tf.math.rsqrt(gamma_rvs) * normal_rvs
    return students_t_rvs


def _bases_cosine(X: TensorType, W: TensorType, b: TensorType) -> TensorType:
    """
    Feature map for random Fourier features (RFF) as originally prescribed
    by Rahimi & Recht, 2007 :cite:p:`rahimi2007random`.
    See also :cite:p:`sutherland2015error` for additional details.
    """
    proj = tf.matmul(X, W, transpose_b=True) + b  # [N, M]
    return tf.cos(proj)  # [N, M]


def _bases_concat(X: TensorType, W: TensorType) -> TensorType:
    """
    Feature map for random Fourier features (RFF) as originally prescribed
    by Rahimi & Recht, 2007 :cite:p:`rahimi2007random`.
    See also :cite:p:`sutherland2015error` for additional details.
    """
    proj = tf.matmul(X, W, transpose_b=True)  # [N, M]
    return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)  # [N, 2M]


def _ceil_divide(a: float, b: float) -> int:
    """
    Ceiling division. Returns the smallest integer `m` s.t. `m*b >= a`.
    """
    return -np.floor_divide(-a, b)
