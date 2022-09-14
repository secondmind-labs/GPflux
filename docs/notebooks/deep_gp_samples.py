# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Deep GP samples

To help develop a more intuitive understanding of deep Gaussian processes, in this notebook we show how to generate a sample from the full deep GP, by propagating a sample through the layers.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.layers import GPLayer
from gpflux.experiment_support.plotting import plot_layer

tf.random.set_seed(42)

# %%
num_data = 200
D = 1
a, b = 0, 1
X = np.linspace(a, b, num_data).reshape(-1, 1)

# %% [markdown]
"""
## Constructing the layers

Note that we give the `full_cov=True` argument to `GPLayer` so that we obtain correlated samples.
We give the last layer a `gpflow.mean_functions.Zero` mean function (the GPflux default is an Identity mean function).
"""

# %%
num_samples = 5

# %%
Z = X.copy()
M = Z.shape[0]

# Layer 1
inducing_var1 = construct_basic_inducing_variables(
    M, D, D, share_variables=True, z_init=Z.copy()
)
kernel1 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=0.15),
    output_dim=D,
    share_hyperparams=True,
)
layer1 = GPLayer(
    kernel1, inducing_var1, num_data, full_cov=True, num_samples=num_samples
)

# Layer 2
inducing_var2 = construct_basic_inducing_variables(
    M, D, D, share_variables=True, z_init=Z.copy()
)
kernel2 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=0.8, variance=0.1),
    output_dim=D,
    share_hyperparams=True,
)
layer2 = GPLayer(
    kernel2, inducing_var2, num_data, full_cov=True, num_samples=num_samples
)

# Layer 3
inducing_var3 = construct_basic_inducing_variables(
    M, D, D, share_variables=True, z_init=Z.copy()
)
kernel3 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=0.3, variance=0.1),
    output_dim=D,
    share_hyperparams=True,
)
layer3 = GPLayer(
    kernel3,
    inducing_var3,
    num_data,
    full_cov=True,
    num_samples=num_samples,
    mean_function=gpflow.mean_functions.Zero(),
)

gp_layers = [layer1, layer2, layer3]

# %% [markdown]
"""
## Propagating samples through the layers
"""

# %%
layer_input = X

# %%
means, covs, samples = [], [], []

for layer in gp_layers:
    layer_output = layer(layer_input)

    mean = layer_output.mean()
    cov = layer_output.covariance()
    sample = tf.convert_to_tensor(layer_output)  # generates num_samples samples...

    layer_input = sample[0]  # for the next layer

    means.append(mean.numpy().T)  # transpose to go from [1, N] to [N, 1]
    covs.append(cov.numpy())
    samples.append(sample.numpy())

# %% [markdown]
"""
## Visualising samples

From top to bottom we plot the input to a layer, the covariance of outputs of that layer, and samples from the layer's output.
"""

# %%
num_layers = len(gp_layers)
fig, axes = plt.subplots(3, num_layers, figsize=(num_layers * 3.33, 10))

for i in range(num_layers):
    layer_input = X if i == 0 else samples[i - 1][0]
    plot_layer(X, layer_input, means[i], covs[i], samples[i], i, axes[:, i])
