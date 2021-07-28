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
# Introduction to GPflux

In this notebook we cover the basics of Deep Gaussian processes (DGPs) <cite data-cite="damianou2013deep"/> with GPflux. We assume that the reader is familiar with the concepts of Gaussian processes and Deep GPs (see <cite data-cite="rasmussen,gpflow2020"/> for an in-depth overview).
"""

# %%
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
import tensorflow as tf

tf.keras.backend.set_floatx("float64")
tf.get_logger().setLevel("INFO")

# %% [markdown]
"""
## The motorcycle dataset

We are going to model a one-dimensional dataset containing observations from a simulated motorcycle accident, used to test crash helmets \[1\]. The dataset has many interesting properties that we can use to showcase the power of DGPs as opposed to shallow (that is, single layer) GP models.
"""


# %%
def motorcycle_data():
    """ Return inputs and outputs for the motorcycle dataset. We normalise the outputs. """
    df = pd.read_csv("./data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    X = X / X.max()
    return X, Y


X, Y = motorcycle_data()
# plt.plot(X, Y, "kx")
# plt.xlabel("time")
# plt.ylabel("Acceleration")

# %% [markdown]
"""
## Single-layer GP

We start this introduction by building a single-layer GP using GPflux. However, you'll notice that we rely a lot on [GPflow](www.github.com/GPflow/GPflow) objects to build our GPflux model. This is a conscious decision. GPflow contains well-tested and stable implementations of key GP building blocks, such as kernels, likelihoods, inducing variables, mean functions, and so on. By relying on GPflow for these elements, we can keep the GPflux code lean and focused on what is important for building deep GP models.  

We are going to build a Sparse Variational Gaussian process (SVGP), for which we refer to <cite data-cite="gpflow2020"/> or <cite data-cite="leibfried2020tutorial"/> for an in-depth overview.
"""

# %% [markdown]
"""
We start by importing `gpflow` and `gpflux`, and then create a kernel and inducing variable. Both are GPflow objects. We pass both objects to a `GPLayer` which will represent a single GP layer in our deep GP. We also need to specify the total number of datapoints and the number of outputs.
"""

# %%
import gpflow
import gpflux
import tensorflow_probability as tfp

from gpflux.encoders import DirectlyParameterizedNormalDiag
from gpflux.layers import LatentVariableLayer

num_data = len(X)
num_inducing = 100
output_dim = Y.shape[1]
EPOCHS = int(1e3)


w_dim = 1
prior_means = np.zeros(w_dim)
prior_std = np.ones(w_dim) * 1e-2
encoder = DirectlyParameterizedNormalDiag(num_data, w_dim)
prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)
lv = LatentVariableLayer(prior, encoder)


kernel = gpflow.kernels.SquaredExponential(lengthscales=[.5, .5])
inducing_variable = gpflow.inducing_variables.InducingPoints(
    np.random.randn(100, 2)
)
gpflow.utilities.set_trainable(inducing_variable, False)
gp_layer = gpflux.layers.GPLayer(
    kernel, inducing_variable, num_data=num_data, num_latent_gps=output_dim, mean_function=gpflow.mean_functions.Zero()
)

# %% [markdown]
"""
We now create a `LikelihoodLayer` which encapsulates a `Likelihood` from GPflow. The likelihood layer is responsible for computing the variational expectation in the objective, and for dealing with our likelihood distribution $p(y | f)$. Other typical likelihoods are `Softmax`, `RobustMax`, and `Bernoulli`. Because we are dealing with the simple case of regression in this notebook, we use the `Gaussian` likelihood.
"""

# %%
likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))

# %% [markdown]
"""
We now pass our GP layer and likelihood layer into a GPflux `DeepGP` model. The `DeepGP` class is a specialisation of the Keras `Model` where we've added a few helper methods such as `predict_f`, `predict_y`, and `elbo`. By relying on Keras, we hope that  users will have an easy time familiarising themselves with the GPflux API. Our `DeepGP` model has the same public API calls: `fit`, `predict`, `evaluate`, and so on.
"""

# %%
gpflow.utilities.set_trainable(lv, False)
single_layer_dgp = gpflux.models.DeepGP([lv, gp_layer], likelihood_layer)
model = single_layer_dgp.as_training_model()
model.compile(tf.optimizers.Adam(0.01))

# %% [markdown]
"""
We train the model by calling `fit`. Keras handles minibatching the data, and keeps track of our loss during training.
"""

# %%
history = model.fit({"inputs": X, "targets": Y}, epochs=EPOCHS, verbose=1, batch_size=num_data)


def plot_history(history, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(history)
    ax.set_xlabel("Epochs")

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Objecitive single layer GP")
plot_history(history.history["loss"], ax=ax1)

# %% [markdown]
"""
We can now visualise the fit. We clearly see that a single layer GP with a simple stationary kernel has trouble fitting this dataset.
"""


# %%
def plot(model, X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    a = .1
    N_test = 200
    X_test = np.linspace(X.min() - a, X.max() + a, N_test).reshape(-1, 1)
    out = model(X_test)
    X_test = X_test.squeeze()

    for mu, var in [(out.y_mean, out.y_var), (out.f_mean, out.f_var)]:
        mu = mu.numpy().squeeze()
        var = var.numpy().squeeze()
        lower = mu - 2 * np.sqrt(var)
        upper = mu + 2 * np.sqrt(var)

        ax.plot(X_test, mu, "C1")
        ax.fill_between(X_test, lower, upper, color="C1", alpha=0.3)

    ax.plot(X, Y, "kx", alpha=0.5)
    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)


plot(single_layer_dgp.as_prediction_model(), X, Y, ax=ax2)

plt.show()

# %%
gpflow.utilities.set_trainable(lv, True)
single_layer_dgp = gpflux.models.DeepGP([lv, gp_layer], likelihood_layer)
model = single_layer_dgp.as_training_model()
model.compile(tf.optimizers.Adam(0.01))

# %% [markdown]
"""
We train the model by calling `fit`. Keras handles minibatching the data, and keeps track of our loss during training.
"""

# %%
history = model.fit({"inputs": X, "targets": Y}, epochs=EPOCHS, verbose=1, batch_size=num_data)


def plot_history(history, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(history)
    ax.set_xlabel("Epochs")

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Objecitive single layer GP")
plot_history(history.history["loss"], ax=ax1)

# %% [markdown]
"""
We can now visualise the fit. We clearly see that a single layer GP with a simple stationary kernel has trouble fitting this dataset.
"""


# %%
def plot(model, X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    a = .1
    N_test = 200
    X_test = np.linspace(X.min() - a, X.max() + a, N_test).reshape(-1, 1)
    out = model(X_test)
    X_test = X_test.squeeze()

    for mu, var in [(out.y_mean, out.y_var), (out.f_mean, out.f_var)]:
        mu = mu.numpy().squeeze()
        var = var.numpy().squeeze()
        lower = mu - 2 * np.sqrt(var)
        upper = mu + 2 * np.sqrt(var)

        ax.plot(X_test, mu, "C1")
        ax.fill_between(X_test, lower, upper, color="C1", alpha=0.3)

    ax.plot(X, Y, "kx", alpha=0.5)
    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)


plot(single_layer_dgp.as_prediction_model(), X, Y, ax=ax2)

plt.show()