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
# Efficient sampling using Random Fourier Features

In this notebook we showcase how to efficiently draw samples from a GP using Random Fourier Features <cite data-cite="rahimi2007random"/>. The main idea is to group a kernel (e.g., `gpflow.kernels.Matern52`) with its RFF decomposition using `gpflux.sampling.KernelWithFeatureDecomposition`)

TODO: link to other notebooks
"""
# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gpflow
import gpflux

from gpflow.config import default_float

from gpflux.layers.basis_functions.random_fourier_features import RandomFourierFeatures
from gpflux.sampling import KernelWithFeatureDecomposition
from gpflux.models.deep_gp import sample_dgp

tf.keras.backend.set_floatx("float64")

# %%
d = np.load("../../tests/snelson1d.npz")
X, Y = data = d["X"], d["Y"]
num_data, input_dim = X.shape

# %% [markdown]
r"""
## Setting up the kernel and its feature decomposition

The `KernelWithFeatureDecomposition` instance represents a kernel together with its finite feature decomposition. Such that:
$$
k(x, x') = \sum_{i=0}^L \lambda_i \phi_i(x) \phi_i(x'),
$$
where $\lambda_i$ and $\phi_i(\cdot)$ are the coefficients and features, respectively.
"""

# %%
kernel = gpflow.kernels.Matern52()
Z = np.linspace(X.min(), X.max(), 10).reshape(-1, 1).astype(np.float64)

inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
gpflow.utilities.set_trainable(inducing_variable, False)

num_rff = 1000
eigenfunctions = RandomFourierFeatures(kernel, num_rff, dtype=default_float())
eigenvalues = np.ones((num_rff, 1), dtype=default_float())
kernel_with_features = KernelWithFeatureDecomposition(kernel, eigenfunctions, eigenvalues)

# %% [markdown]
r"""
## Building and training the single-layer GP
"""
# %%
layer = gpflux.layers.GPLayer(
    kernel_with_features,
    inducing_variable,
    num_data,
    whiten=False,
    num_latent_gps=1,
    mean_function=gpflow.mean_functions.Zero(),
)
likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian())  # noqa: E231
dgp = gpflux.models.DeepGP([layer], likelihood_layer)
model = dgp.as_training_model()
# %%

model.compile(tf.optimizers.Adam(learning_rate=0.1))

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        patience=5,
        factor=0.95,
        verbose=0,
        min_lr=1e-6,
    )
]

history = model.fit(
    {"inputs": X, "targets": Y},
    batch_size=num_data,
    epochs=100,
    callbacks=callbacks,
    verbose=0,
)
# %% [markdown]
r"""
## Drawing samples

Now that the model is trained we can draw consistent samples from the posterior GP.
"""

# %%
a, b = 5, 5
n_x = 1000
spread = X.max() + b - (X.min() - a)
X_test = np.linspace(X.min() - a, X.max() + b, n_x).reshape(-1, 1)

f_mean, f_var = dgp.predict_f(X_test)
f_scale = np.sqrt(f_var)

n_sim = 10
for _ in range(n_sim):
    f_sample = sample_dgp(dgp)
    plt.plot(X_test, f_sample(X_test).numpy())

plt.plot(X_test, f_mean, "C0")
plt.plot(X_test, f_mean + f_scale, "C0--")
plt.plot(X_test, f_mean - f_scale, "C0--")
plt.plot(X, Y, "kx")
plt.xlim(X.min() - a, X.max() + b)
plt.ylim(Y.min() - a, Y.max() + b)
plt.show()
