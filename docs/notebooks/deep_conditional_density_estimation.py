import tensorflow as tf
import gpflow
import gpflux
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity


tf.keras.backend.set_floatx("float64")

"""
In this notebook we explore the use of Deep Gaussian processes and Latent Variables to model a dataset with heteroscedastic noise.
"""

####################### data

Ns = 200
Xs = np.linspace(-.1, 1.1, Ns).reshape(-1, 1)


def motorcycle_data():
    """ Return inputs and outputs for the motorcycle dataset. We normalise the outputs. """
    import pandas as pd
    df = pd.read_csv("./data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    X /= X.max()
    return X, Y


X, Y = motorcycle_data()
N, d_xim = X.shape

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X, Y, marker='x', color='k')
ax.set_ylim(-1, 2)
plt.savefig("data.png")

####################### model
w_dim = 1
num_inducing = 50

prior_means = np.zeros(w_dim)
prior_std = np.ones(w_dim)
encoder = gpflux.encoders.DirectlyParameterizedNormalDiag(N, w_dim)
prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)

lv = gpflux.layers.LatentVariableLayer(prior, encoder)

kernel = gpflow.kernels.SquaredExponential(lengthscales=[.05, .2], variance=1.)
inducing_variable = gpflow.inducing_variables.InducingPoints(
    # np.random.randn(num_inducing, 2)
    np.concatenate(
        [
            np.linspace(X.min(), X.max(), num_inducing).reshape(-1, 1),
            np.random.randn(num_inducing, 1),
        ],
        axis=1
    )
)
gp_layer = gpflux.layers.GPLayer(
    kernel, inducing_variable, num_data=N, num_latent_gps=1, mean_function=gpflow.mean_functions.Zero(),
)

kernel = gpflow.kernels.SquaredExponential(lengthscales=1.0, variance=.01)
inducing_variable = gpflow.inducing_variables.InducingPoints(
    np.linspace(-2, 2, num_inducing).reshape(-1, 1),
)
gp_layer2 = gpflux.layers.GPLayer(
    kernel, inducing_variable, num_data=N, num_latent_gps=1, mean_function=gpflow.mean_functions.Identity(),
)
gp_layer2.q_sqrt.assign(gp_layer.q_sqrt * 1e-5)

likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

dgp = gpflux.models.DeepGP([lv, gp_layer, gp_layer2], likelihood_layer)

gpflow.utilities.print_summary(dgp)

model = dgp.as_training_model()
model.compile(tf.optimizers.Adam(0.005))
history = model.fit({"inputs": X, "targets": Y}, epochs=int(7e3), verbose=1, batch_size=N, shuffle=False)


########################### plotting
def predict_y_samples(prediction_model, Xs, num_samples=25):
    samples = []
    for i in tqdm(range(num_samples)):
        out = prediction_model(Xs)
        s = out.y_mean + out.y_var ** .5 * tf.random.normal(tf.shape(out.y_mean), dtype=out.y_mean.dtype)
        samples.append(s)
    return tf.concat(samples, axis=1)


def plot_samples(ax, N_samples=25):
    samples = predict_y_samples(dgp.as_prediction_model(), Xs, N_samples).numpy().T
    Xs_tiled = np.tile(Xs, [N_samples, 1])
    ax.scatter(Xs_tiled.flatten(), samples.flatten(), marker='.', alpha=0.2, color='C0')
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(min(Xs), max(Xs))
    ax.scatter(X, Y, marker='.', color='C1')


def plot_latent_variables(ax):
    for l in dgp.f_layers:
        if isinstance(l, gpflux.layers.LatentVariableLayer):
            m = l.encoder.means.numpy()
            s = l.encoder.stds.numpy()
            ax.errorbar(X.flatten(), m.flatten(), yerr=s.flatten(), fmt='o')
            return


def plot_density(axes, N_samples=5_000, samples=None):
    if samples is None:
        samples = predict_y_samples(dgp.as_prediction_model(), Xs, N_samples).numpy().T
        print(samples.shape)

    if isinstance(axes, (list, np.ndarray)):
        ax = axes[0]
    else:
        ax = axes

    ax.scatter(X, Y, marker='.', color='C1')
    levels = np.linspace(-2.5, 2.5, 200)
    ax.set_ylim(min(levels), max(levels))
    ax.set_xlim(min(Xs), max(Xs))

    cs = np.zeros((len(Xs), len(levels)))
    for i, Ss in enumerate(samples.T):
        bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
        kde = KernelDensity(bandwidth=float(bandwidth))
        kde.fit(Ss.reshape(-1, 1))
        for j, level in enumerate(levels):
            cs[i, j] = kde.score(np.array(level).reshape(1, 1))
    ax.pcolormesh(Xs.flatten(), levels, np.exp(cs.T), cmap='Blues', vmin=0, vmax=3)  # , alpha=0.1)
    print(np.max(np.exp(cs.T)))
    print(np.min(np.exp(cs.T)))
    ax.scatter(X, Y, marker='x', color='k')

    if isinstance(axes, (list, np.ndarray)) and len(axes) > 1:
        ax = axes[1]
        ax.scatter(X, Y, marker='x', color='k')
        ax.set_ylim(min(levels), max(levels))
        ax.set_xlim(min(Xs), max(Xs))
        m = np.mean(samples, 0).flatten()
        v = np.var(samples, 0).flatten()
        ax.plot(Xs.flatten(), m, "C1")
        ax.plot(Xs.flatten(), m + 2 * v ** .5, "C1--")
        ax.plot(Xs.flatten(), m - 2 * v ** .5, "C1--")

    return samples


def plot_mean_and_var(ax, samples=None, N_samples=1_000):
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
    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("acceleration")
    return samples



fig, axes = plt.subplots(3, 2, figsize=(6, 6))
axes = np.array(axes).flatten()
plot_samples(axes[0])
plot_latent_variables(axes[1])
samples = plot_density(axes[[2, 3]])
axes[4].plot(history.history["loss"])

plt.savefig("cde.png")
plt.close()


fig, ax = plt.subplots()
plot_density(ax, samples=samples)
plt.savefig("cde2.png")
plt.close()

fig, ax = plt.subplots()
plot_mean_and_var(ax, samples=samples)
plt.savefig("cde3.png")
plt.close()