# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Deep GP samples

TODO: Some explanation...
"""
# %%
import gpflow
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.layers import GPLayer
import numpy as np
import tensorflow as tf
from gpflux.experiment_support.plotting import plot_layers

tf.random.set_seed(42)

# %%
Ns = 1000
D = 1
a, b = 0, 1
X = np.linspace(a, b, 1000).reshape(-1, 1)

# %%
Z = X.copy()
M = Z.shape[0]

# Layer 1
inducing_var1 = construct_basic_inducing_variables(M, D, D, share_variables=True, z_init=Z.copy())
kernel1 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=0.15), output_dim=D, share_hyperparams=True,
)
layer1 = GPLayer(kernel1, inducing_var1, Ns)

# Layer 2
inducing_var2 = construct_basic_inducing_variables(M, D, D, share_variables=True, z_init=Z.copy())
kernel2 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=0.8, variance=0.1),
    output_dim=D,
    share_hyperparams=True,
)
layer2 = GPLayer(kernel2, inducing_var2, Ns)

# Layer 3
inducing_var3 = construct_basic_inducing_variables(M, D, D, share_variables=True, z_init=Z.copy())
kernel3 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=0.3, variance=0.1),
    output_dim=D,
    share_hyperparams=True,
)
layer3 = GPLayer(kernel3, inducing_var3, Ns)

gp_layers = [layer1, layer2, layer3]
for layer in gp_layers:
    layer.build([None, D])

# %%
# model = gpflux.DeepGP(np.empty((1, 1)), np.empty((1, 1)), [layer1, layer2, layer3])

# %%
plot_layers(X, gp_layers)
