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
# Demo visualisation

"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

import gpflow
import gpflux
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.layers import GPLayer
from gpflux.experiment_support.plotting import plot_layer

tf.keras.backend.set_floatx("float64")

tf.random.set_seed(42)

# %%
num_data = 200
D = 1
a, b = 0, 60
X = np.linspace(a, b, num_data).reshape(-1, 1)


# %%
def motorcycle_data():
    """ Return inputs and outputs for the motorcycle dataset. We normalise the outputs. """
    df = pd.read_csv("./data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    return X, Y


tX, tY = motorcycle_data()
plt.plot(tX, tY, "kx")
plt.xlabel("time")
plt.ylabel("Acceleration")

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
inducing_var1 = construct_basic_inducing_variables(M, D, D, share_variables=True, z_init=Z.copy())
kernel1 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=5, variance=10),
    output_dim=D,
    share_hyperparams=True,
)
layer1 = GPLayer(kernel1, inducing_var1, num_data, num_samples=num_samples)

# Layer 2
inducing_var2 = construct_basic_inducing_variables(M, D, D, share_variables=True, z_init=Z.copy())
kernel2 = construct_basic_kernel(
    gpflow.kernels.SquaredExponential(lengthscales=5),
    output_dim=D,
    share_hyperparams=True,
)
layer2 = GPLayer(
    kernel2,
    inducing_var2,
    num_data,
    num_samples=num_samples,
    mean_function=gpflow.mean_functions.Zero(),
)

gp_layers = [layer1, layer2]

# %% [markdown]
"""
## Propagating samples through the layers
"""

# %%
layer_input = X

# %%
means, covs, samples = [], [], []

for layer in gp_layers:
    layer.full_cov = True
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

# %%
likelihood_layer.likelihood

# %%
likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
dgp = gpflux.models.DeepGP(gp_layers, likelihood_layer)

# %%
for layer in gp_layers:
    layer.full_cov = False


# %%
@tf.function
def objective():
    return - dgp.elbo((tX, tY))


# %%
objective()

# %%
opt = tf.optimizers.Adam(learning_rate=0.001)


# %%
@tf.function
def step():
    opt.minimize(objective, dgp.trainable_variables)


# %%
objective()

# %%
# elbos = []
for _ in range(1000):
    elbos.append(-objective())
    step()


# %%
plt.figure()
plt.plot(elbos)
plt.ylim(-200, -100)

# %%
