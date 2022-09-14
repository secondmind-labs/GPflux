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
Math utilities
"""
import tensorflow as tf

from gpflow import default_jitter
from gpflow.base import TensorType


def _cholesky_with_jitter(cov: TensorType) -> tf.Tensor:
    """
    Compute the Cholesky of the covariance, adding jitter (determined by
    :func:`gpflow.default_jitter`) to the diagonal to improve stability.

    :param cov: full covariance with shape ``[..., N, D, D]``.
    """
    # cov [..., N, D, D]
    cov_shape = tf.shape(cov)
    batch_shape = cov_shape[:-2]
    D = cov_shape[-2]
    jittermat = default_jitter() * tf.eye(
        D, batch_shape=batch_shape, dtype=cov.dtype
    )  # [..., N, D, D]
    return tf.linalg.cholesky(cov + jittermat)  # [..., N, D, D]


def compute_A_inv_b(A: TensorType, b: TensorType) -> tf.Tensor:
    r"""
    Computes :math:`A^{-1} b` using the Cholesky of ``A`` instead of the explicit inverse,
    as this is often numerically more stable.

    :param A: A positive-definite matrix with shape ``[..., M, M]``.
        Can contain any leading dimensions (``...``) as long as they correspond
        to the leading dimensions in ``b``.
    :param b: Tensor with shape ``[..., M, D]``.
        Can contain any leading dimensions (``...``) as long as they correspond
        to the leading dimensions in ``A``.

    :returns: Tensor with shape ``[..., M, D]``.
        Leading dimensions originate from ``A`` and ``b``.
    """
    # A = L Lᵀ
    L = tf.linalg.cholesky(A)
    # A⁻¹ = L⁻ᵀ L⁻¹
    L_inv_b = tf.linalg.triangular_solve(L, b)
    A_inv_b = tf.linalg.triangular_solve(L, L_inv_b, adjoint=True)  # adjoint = transpose
    return A_inv_b
