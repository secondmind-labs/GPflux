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
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from gpflow.base import TensorType


def plot_layer(
    X: TensorType,
    layer_input: TensorType,
    mean: List[TensorType],
    cov: List[TensorType],
    sample: List[TensorType],
    idx: Optional[int],
    axes: Optional[Sequence[plt.Axes]] = None,
) -> None:
    """
    :param X: original inputs to the DGP, shape [N, 1]
    :param layer_input: inputs to this layer, shape [N, 1]
    :param mean: mean of this layer's output, shape [N, 1]
    :param cov: covariance of this layer's output, shape [1, N, N]
    :param sample: samples from this layer's output, shape [S, N, 1]
    :param idx: the index of this layer (for labels)
    :param axes: the sequence of 3 axes on which to plot
    """
    if axes is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3, 10))
    else:
        assert len(axes) == 3
        ax1, ax2, ax3 = axes

    # Input
    title = "Input"
    if idx is not None:
        title = f"Layer {idx + 1}\n{title}"
    ax1.set_title(title)
    ax1.plot(X, layer_input)

    # covariance
    ax2.matshow(np.squeeze(cov, axis=0), aspect="auto")
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    # samples
    ax3.set_title("Samples")
    ax3.plot(X, np.squeeze(sample, axis=-1).T)


def plot_layers(
    X: TensorType, means: List[TensorType], covs: List[TensorType], samples: List[TensorType]
) -> None:  # pragma: no cover
    L = len(means)
    fig, axes = plt.subplots(3, L, figsize=(L * 3.33, 10))
    for i in range(L):
        layer_input = X if i == 0 else samples[i - 1][0]
        plot_layer(X, layer_input, means[i], covs[i], samples[i], i, axes[:, i])
