# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import matplotlib.pyplot as plt
import tensorflow as tf

from gpflow.conditionals.util import sample_mvn


def all_layer_mean_var_samples(gp_layers, X):
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
    ax3.plot(X, s[idx][:, :, 0].T)


def plot_layers(X, gp_layers):
    L = len(gp_layers)
    m, v, s = all_layer_mean_var_samples(gp_layers, X)
    fig, axes = plt.subplots(3, L, figsize=(L * 3.33, 10))
    for i in range(L):
        plot_layer(X, m, v, s, i, axes[:, i])
