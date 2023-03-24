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
import tensorflow as tf

import gpflow
from gpflow.base import TensorType


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


def _bases_cosine(X: TensorType, W: TensorType, b: TensorType) -> TensorType:
    """
    Feature map for random Fourier features (RFF) as originally prescribed
    by Rahimi & Recht, 2007 :cite:p:`rahimi2007random`.
    See also :cite:p:`sutherland2015error` for additional details.
    """
    proj = tf.matmul(X, W, transpose_b=True) + b  # [N, M] or [P, N, M]
    return tf.cos(proj)  # [N, M] or [P, N, M]


def _bases_concat(X: TensorType, W: TensorType) -> TensorType:
    """
    Feature map for random Fourier features (RFF) as originally prescribed
    by Rahimi & Recht, 2007 :cite:p:`rahimi2007random`.
    See also :cite:p:`sutherland2015error` for additional details.
    """
    proj = tf.matmul(X, W, transpose_b=True)  # [N, M] or [P, N, M]
    return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)  # [N, 2M] or [P, N, M]
