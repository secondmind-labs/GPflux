# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
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
# Hybrid Deep GP models: combining GP and Neural net Layers

In this notebook we show how to combine `gpflux.layers.GPLayer` layers with plain keras neural network layers. This allows one to build hybrid deep GP models. Compared to the other tutorials, we are also going to use keras' `Sequential` model to build our hierarchical model and use a `gpflux.losses.LikelihoodLoss` instead of a `gpflux.layers.LikelihoodLayer`.
"""

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gpflow
import gpflux

from gpflow.config import default_float

tf.keras.backend.set_floatx("float64")

# %% [markdown]
"""
## Load Snelson dataset

We use a simple one-dimensional dataset to allow for easy plotting. To help training we normalize the input features.
"""

# %%
d = np.load("../../tests/snelson1d.npz")
X, Y = data = d["X"], d["Y"]
X = (X - X.mean()) / X.std()
num_data, input_dim = X.shape

# %% [markdown]
"""
## Initialize the GP Layer

As per usual we create a one-dimensional `gpflux.layers.GPLayer` with a simple `SquaredExponential` kernel and `InducingPoints` inducing variable:
"""

# %%
num_data = len(X)
num_inducing = 10
output_dim = Y.shape[1]

kernel = gpflow.kernels.SquaredExponential()
inducing_variable = gpflow.inducing_variables.InducingPoints(
    np.linspace(X.min(), X.max(), num_inducing).reshape(-1, 1)
)
gp_layer = gpflux.layers.GPLayer(
    kernel, inducing_variable, num_data=num_data, num_latent_gps=output_dim
)

# %% [markdown]
"""
## Sequential Keras model with GP and Neural net layers

We construct a model that consists of three `tf.keras.layers.Dense` layers and a GP. The first two Dense layers are configured to have 100 units and use a ReLU non-linearity. The last neural network layers reduces the dimension to one and does not utilise a non-linearity. We can interpret these three neural network layers as performing non-linear feature warping. The final layer in the model is the GP we defined above.
"""

# %%
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
        gp_layer,
    ]
)
loss = gpflux.losses.LikelihoodLoss(gpflow.likelihoods.Gaussian(0.001))

# TODO: How do we track the variables in the likelihoodloss?
# model = gpflux.layers.TrackableLayer(loss)

# %% [markdown]
"""
We compile our model by specifying the loss and the optimizer to use. Once this is done, we fit the data and plot the trajectory of the loss:
"""

# %%
model.compile(loss=loss, optimizer="adam")
hist = model.fit(X, Y, epochs=500, verbose=0)
plt.plot(hist.history["loss"])

# %% [markdown]
"""
We can now inspect the final model by plotting its predictions. Note that `model(X_test)` now returns the output of the final `GPLayer` and *not* a `LikelihoodLayer`. The output of a `GPLayer` is a TFP distribution with a `mean()` and `variance()`.
"""


# %%
def plot(model, X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    a = 1.0
    N_test = 100
    X_test = np.linspace(X.min() - a, X.max() + a, N_test).reshape(-1, 1)
    out = model(X_test)

    mu = out.mean().numpy().squeeze()
    var = out.variance().numpy().squeeze()
    X_test = X_test.squeeze()
    lower = mu - 2 * np.sqrt(var)
    upper = mu + 2 * np.sqrt(var)

    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)
    ax.plot(X, Y, "kx", alpha=0.5)
    ax.plot(X_test, mu, "C1")

    ax.fill_between(X_test, lower, upper, color="C1", alpha=0.3)
    return out


o = plot(model, X, Y)
