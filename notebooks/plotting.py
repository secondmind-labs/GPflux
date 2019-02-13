# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import matplotlib.pyplot as plt

import gpflow
from gpflux.models.deep_gp import DeepGP


class PlotDeepGP(DeepGP):
    """ Adds plotting functionality to the model """

    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]))
    def all_layer_mean_var_samples(self, X):
        S = 5
        sample = X
        means, variances, samples = [], [], []
        for l in self.layers:
            all_samples, m, v = l.propagate(sample, X=X,
                                            full_output_cov=False, full_cov=True,
                                            num_samples=S)  # S x N x D, N x D, D x N x N
            sample = all_samples[0]

            means.append(m)
            variances.append(v)
            samples.append(all_samples)

        return means, variances, samples


def plot_layer(X, m, v, s, idx, axes=None):
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
    ax3.plot(X, s[idx][:, :, 0].T);


def plot_layers(X, model):
    L = len(model.layers)
    m, v, s = model.all_layer_mean_var_samples(X)
    fig, axes = plt.subplots(3, L, figsize=(L * 3.33, 10))
    for i in range(L):
        plot_layer(X, m, v, s, i, axes[:, i])
