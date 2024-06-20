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
# Hybrid Deep GP models: combining GP and Neural Network layers

In this notebook we show how to combine `gpflux.layers.GPLayer` layers with plain Keras neural network layers. This allows one to build hybrid deep GP models. Compared to the other tutorials, we are also going to use Keras's `Sequential` model to build our hierarchical model and use a `gpflux.losses.LikelihoodLoss` instead of a `gpflux.layers.LikelihoodLayer`.
"""

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gpflow
import gpflux

from gpflow.config import default_float
from gpflow.keras import tf_keras


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
likelihood = gpflow.likelihoods.Gaussian(0.1)

# So that Keras can track the likelihood variance, we need to provide the likelihood as part of a "dummy" layer:
likelihood_container = gpflux.layers.TrackableLayer()
likelihood_container.likelihood = likelihood

model = tf_keras.Sequential(
    [
        tf_keras.layers.Dense(100, activation="relu"),
        tf_keras.layers.Dense(100, activation="relu"),
        tf_keras.layers.Dense(1, activation="linear"),
        gp_layer,
        likelihood_container,  # no-op, for discovering trainable likelihood parameters
    ]
)
loss = gpflux.losses.LikelihoodLoss(likelihood)

# %% [markdown]
"""
We compile our model by specifying the loss and the optimizer to use. After this is done, we fit the data and plot the trajectory of the loss:
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

    x_margin = 2.0
    N_test = 100
    X_test = np.linspace(X.min() - x_margin, X.max() + x_margin, N_test).reshape(-1, 1)
    f_distribution = model(X_test)

    mean = f_distribution.mean().numpy().squeeze()
    var = f_distribution.variance().numpy().squeeze() + model.layers[-1].likelihood.variance.numpy()
    X_test = X_test.squeeze()
    lower = mean - 2 * np.sqrt(var)
    upper = mean + 2 * np.sqrt(var)

    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)
    ax.plot(X, Y, "kx", alpha=0.5)
    ax.plot(X_test, mean, "C1")

    ax.fill_between(X_test, lower, upper, color="C1", alpha=0.3)


plot(model, X, Y)

# %%
gpflow.utilities.print_summary(model, fmt="notebook")
