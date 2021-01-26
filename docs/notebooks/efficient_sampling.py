# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Efficient sampling

TODO: Some explanation...
"""
# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gpflow
import gpflux

from gpflow.config import default_float

from gpflux.layers.basis_functions.random_fourier_features import RandomFourierFeatures
from gpflux.sampling.kernel_with_mercer_decomposition import KernelWithMercerDecomposition

tf.keras.backend.set_floatx("float64")

# %%
d = np.load("../../tests/snelson1d.npz")
X, Y = data = d["X"], d["Y"]
num_data, input_dim = X.shape

# %%
kernel = gpflow.kernels.Matern52()
Z = np.linspace(X.min(), X.max(), 10).reshape(-1, 1).astype(np.float64)

num_rff = 1000
inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
gpflow.utilities.set_trainable(inducing_variable, False)

layer = gpflux.layers.GPLayer(
    kernel,
    inducing_variable,
    num_data,
    white=False,
    verify=False,
    num_latent_gps=1,
    mean_function=gpflow.mean_functions.Zero(),
)
layer._initialized = True
likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian())  # noqa: E231
model = gpflux.models.DeepGP([layer], likelihood_layer)
# %%

model.compile(tf.optimizers.Adam(learning_rate=0.1))

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=5, factor=0.95, verbose=1, min_lr=1e-6,
    )
]

history = model.fit(x=X, y=Y, batch_size=num_data, epochs=100, callbacks=callbacks)


# %%
eigenfunctions = RandomFourierFeatures(kernel, num_rff, dtype=default_float())
eigenvalues = np.ones((num_rff, 1), dtype=default_float())
kernel2 = KernelWithMercerDecomposition(None, eigenfunctions, eigenvalues)


a, b = 5, 5
n_x = 1000
spread = X.max() + b - (X.min() - a)
X_test_1 = np.sort(np.random.rand(50, 1) * spread + (X.min() - a), axis=0)
X_test = np.linspace(X.min() - a, X.max() + b, n_x).reshape(-1, 1)

f_dist = model.predict_f(X_test)
m = f_dist.mean().numpy()
v = f_dist.scale.diag.numpy()


n_sim = 10
for i in range(n_sim):
    f_sample_post = model.sample()
    plt.plot(X_test, f_sample_post(X_test).numpy())

plt.plot(X_test, m, "C0")
plt.plot(X_test, m + v ** 0.5, "C0--")
plt.plot(X_test, m - v ** 0.5, "C0--")
plt.plot(X, Y, "kx")
plt.xlim(X.min() - a, X.max() + b)
plt.ylim(Y.min() - a, Y.max() + b)
plt.show()
