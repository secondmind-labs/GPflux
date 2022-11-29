# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Sparse Orthogonal Variational Inference for Deep Gaussian Processes
#
# In this notebook, we explore the use of a new interpretation of sparse variational approximations for Gaussian processes using inducing points, which can lead to more scalable algorithms than previous methods. It is based on decomposing a Gaussian process as a sum of two independent processes: one spanned by a finite basis of inducing points and the other capturing the remaining variation <cite data-cite="shi2020sparseorthogonal"/>.

# +
import tensorflow as tf
import gpflow
import gpflux
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity


# -

# ## Load data
#
# The data comes from a motorcycle accident simulation [1] and shows some interesting behaviour. In particular the heteroscedastic nature of the noise.


def motorcycle_data():
    """Return inputs and outputs for the motorcycle dataset. We normalise the outputs."""
    import pandas as pd

    df = pd.read_csv("./data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    X /= X.max()
    return X, Y


# +
X, Y = motorcycle_data()
num_data, d_xim = X.shape

X_MARGIN, Y_MARGIN = 0.1, 0.5
fig, ax = plt.subplots()
ax.scatter(X, Y, marker="x", color="k")
ax.set_ylim(Y.min() - Y_MARGIN, Y.max() + Y_MARGIN)
ax.set_xlim(X.min() - X_MARGIN, X.max() + X_MARGIN)
# -

# ## Orthogonal Deep Gaussian process
#
# To tackle the problem we suggest a Deep Gaussian process with a latent variable in the first layer. The latent variable will be able to capture the
# heteroscedasticity, while the two-layered deep GP is able to model the sharp transitions.
#
# Note that a GPflux Deep Gaussian process by itself (i.e. without the latent variable layer) is not able to capture the heteroscedasticity of this dataset. This is a consequence of the noise-less hidden layers and the doubly-stochastic variational inference training procedure, as forumated in <cite data-cite="salimbeni2017doubly">. On the contrary, the original deep GP suggested by Damianou and Lawrence <cite data-cite="damianou2013deep">, using a different variational approximation for training, can model this dataset without a latent variable, as shown in [this blogpost](https://inverseprobability.com/talks/notes/deep-gps.html).

# ### Latent Variable Layer
#
# This layer concatenates the inputs with a latent variable. See Dutordoir, Salimbeni et al. Conditional Density with Gaussian processes (2018) <cite data-cite="dutordoir2018cde"/> for full details. We choose a one-dimensional input and a full parameterisation for the latent variables. This means that we do not need to train a recognition network, which is useful for fitting but can only be done in the case of small datasets, as is the case here.

w_dim = 1
prior_means = np.zeros(w_dim)
prior_std = np.ones(w_dim)
encoder = gpflux.encoders.DirectlyParameterizedNormalDiag(num_data, w_dim)
prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)
lv = gpflux.layers.LatentVariableLayer(prior, encoder)

# ### First GP layer
#
# GP Layer with two dimensional input because it acts on the inputs and the one-dimensional latent variable. We use a Squared Exponential kernel, a zero mean function, and inducing points, whose pseudo input locations are carefully chosen.

# +

kernel = gpflow.kernels.SquaredExponential(lengthscales=[0.05, 0.2], variance=1.0)
inducing_variable = gpflow.inducing_variables.InducingPoints(
    np.concatenate(
        [
            np.linspace(X.min(), X.max(), NUM_INDUCING).reshape(-1, 1),
            np.random.randn(NUM_INDUCING, 1),
        ],
        axis=1,
    )
)
gp_layer = gpflux.layers.GPLayer(
    kernel,
    inducing_variable,
    num_data=num_data,
    num_latent_gps=1,
    mean_function=gpflow.mean_functions.Zero(),
)
# -

# ### Second GP layer
#
# Final layer GP with Squared Exponential kernel

# +

kernel = gpflow.kernels.SquaredExponential()
inducing_variable = gpflow.inducing_variables.InducingPoints(
    np.random.randn(NUM_INDUCING, 1),
)
gp_layer2 = gpflux.layers.GPLayer(
    kernel,
    inducing_variable,
    num_data=num_data,
    num_latent_gps=1,
    mean_function=gpflow.mean_functions.Identity(),
)
gp_layer2.q_sqrt.assign(gp_layer.q_sqrt * 1e-5)

# +

likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))
gpflow.set_trainable(likelihood_layer, False)
dgp = gpflux.models.DeepGP([lv, gp_layer, gp_layer2], likelihood_layer)
gpflow.utilities.print_summary(dgp, fmt="notebook")
# -

# ### Fit
#
# We can now fit the model. Because of the `DirectlyParameterizedEncoder` it is important to set the batch size to the number of datapoints and turn off shuffle. This is so that we use the associated latent variable for each datapoint. If we would use an amortized encoder network this would not be necessary.

model = dgp.as_training_model()
model.compile(tf.optimizers.Adam(0.005))
history = model.fit(
    {"inputs": X, "targets": Y}, epochs=int(20e3), verbose=0, batch_size=num_data, shuffle=False
)

gpflow.utilities.print_summary(dgp, fmt="notebook")

# ### Prediction and plotting code

# +
Xs = np.linspace(X.min() - X_MARGIN, X.max() + X_MARGIN, num_data_test).reshape(-1, 1)


def predict_y_samples(prediction_model, Xs, num_samples=25):
    samples = []
    for i in tqdm(range(num_samples)):
        out = prediction_model(Xs)
        s = out.y_mean + out.y_var ** 0.5 * tf.random.normal(
            tf.shape(out.y_mean), dtype=out.y_mean.dtype
        )
        samples.append(s)
    return tf.concat(samples, axis=1)


def plot_samples(ax, N_samples=25):
    samples = predict_y_samples(dgp.as_prediction_model(), Xs, N_samples).numpy().T
    Xs_tiled = np.tile(Xs, [N_samples, 1])
    ax.scatter(Xs_tiled.flatten(), samples.flatten(), marker=".", alpha=0.2, color="C0")
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(min(Xs), max(Xs))
    ax.scatter(X, Y, marker=".", color="C1")


def plot_latent_variables(ax):
    for l in dgp.f_layers:
        if isinstance(l, gpflux.layers.LatentVariableLayer):
            m = l.encoder.means.numpy()
            s = l.encoder.stds.numpy()
            ax.errorbar(X.flatten(), m.flatten(), yerr=s.flatten(), fmt="o")
            return


# -

#

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_samples(ax1)
plot_latent_variables(ax2)


# Left we show the dataset and posterior samples of $y$. On the right we plot the mean and std. deviation of the latent variables corresponding to the datapoints.


def plot_mean_and_var(ax, samples=None, N_samples=5_000):
    if samples is None:
        samples = predict_y_samples(dgp.as_prediction_model(), Xs, N_samples).numpy().T

    m = np.mean(samples, 0).flatten()
    v = np.var(samples, 0).flatten()

    ax.plot(Xs.flatten(), m, "C1")
    for i in [1, 2]:
        lower = m - i * np.sqrt(v)
        upper = m + i * np.sqrt(v)
        ax.fill_between(Xs.flatten(), lower, upper, color="C1", alpha=0.3)
    ax.plot(X, Y, "kx", alpha=0.5)
    ax.set_ylim(Y.min() - Y_MARGIN, Y.max() + Y_MARGIN)
    ax.set_xlabel("time")
    ax.set_ylabel("acceleration")
    return samples


fig, ax = plt.subplots()
plot_mean_and_var(ax)

# The deep GP model can handle the heteroscedastic noise in the dataset as well as the sharp-ish transition at $0.3$.

# ## Conclusion
#
# In this notebook we created a two layer deep Gaussian process with a latent variable base layer to model a heteroscedastic dataset using GPflux.
#
#
# [1] Silverman, B. W. (1985) “Some aspects of the spline smoothing approach to non-parametric curve fitting”. Journal of the Royal Statistical Society series B 47, 1-52.

#
