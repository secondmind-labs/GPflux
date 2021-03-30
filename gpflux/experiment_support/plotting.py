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
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from gpflow.base import TensorType
from gpflow.conditionals.util import sample_mvn

from gpflux.layers import GPLayer


def all_layer_mean_var_samples(
    gp_layers: Sequence[GPLayer], X: TensorType
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    S = 5
    sample = X
    means, covs, samples = [], [], []
    for layer in gp_layers:
        mean, cov = layer.predict(sample, full_output_cov=False, full_cov=True)  # [N, D], [D, N, N]
        all_samples = sample_mvn(tf.linalg.adjoint(mean), cov, full_cov=True, num_samples=S)
        all_samples = tf.linalg.adjoint(all_samples)
        sample = all_samples[0]

        means.append(mean.numpy())
        covs.append(cov.numpy())
        samples.append(all_samples.numpy())

    return means, covs, samples


def plot_layer(
    X: TensorType,
    m: List[TensorType],
    v: List[TensorType],
    s: List[TensorType],
    idx: int,
    axes: Optional[plt.Axes] = None,
) -> None:  # pragma: no cover
    """
    :param X: inputs of the DGP: N x 1
    :param means: array of num_layer elements of shape N x D
    :param variances: array of num_layer elements of shape D x N x N
    :param samples: array of num_layer elements of shape N x D x S
    """
    if axes is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3, 10))
    else:
        assert len(axes) == 3
        ax1, ax2, ax3 = axes
    # Input
    ax1.set_title("Layer {}\nInput".format(idx + 1))
    layer_input = X if (idx == 0) else s[idx - 1][0, :, 0]
    ax1.plot(X, layer_input)
    # covariance
    ax2.matshow(v[idx][0, ...], aspect="auto")
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    # samples
    ax3.set_title("Samples")
    ax3.plot(X, s[idx][:, :, 0].T)


def plot_layers(X: TensorType, gp_layers: Sequence[GPLayer]) -> None:  # pragma: no cover
    L = len(gp_layers)
    m, v, s = all_layer_mean_var_samples(gp_layers, X)
    fig, axes = plt.subplots(3, L, figsize=(L * 3.33, 10))
    for i in range(L):
        plot_layer(X, m, v, s, i, axes[:, i])
