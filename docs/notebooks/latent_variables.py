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
from PIL import Image
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 3)
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 12})

import gpflow as gpf
from gpflow import default_float
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from gpflow.mean_functions import Zero
from gpflow.kernels import RBF, Matern12, Matern32, Matern52, SharedIndependent
from gpflow.likelihoods import Gaussian

from gpflux.layers import GPLayer, LatentVariableAugmentationLayer, LikelihoodLayer
from gpflux.encoders import DirectlyParameterizedNormalDiag
from gpflux.models import DeepGP, BayesianModel


# Plot limits
x_lim = [-6, 6]
y_lim = [-5, 5]
scatter_color = 'blue'
scatter_alpha = 0.01
edgecolors = None


# Load image and compute training data
image = np.array(Image.open("DGP.png"))
image[np.where(image != 0.0)] = 255.0
image = image[..., 0]
X, y = [], []
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] == 0.0:
            X_n, y_n = j, 40 - i - 20
            X.append(X_n)
            y.append(y_n)
X = np.array(X) / 20.0 - 2.5
y = np.array(y) / 7.0
X = X.astype(default_float())
y = y.astype(default_float())
X_train = X[..., None]
y_train = y[..., None]


# Generate subplot frames and raw data
fig, axs = plt.subplots(1, 5)
for i in range(5):
    axs[i].set_xlim(x_lim)
    axs[i].set_ylim(y_lim)
axs[0].plot(X, y, 'o', color='blue', alpha=0.2)
axs[0].set_title('Raw Data')
axs[0].set_xlabel('$X$')
axs[0].set_ylabel('$f(X)$')
axs[1].set_title('Shallow Sparse GP')
axs[1].set_xlabel('$X$')
axs[2].set_title('Latent-Variable Shallow Sparse GP')
axs[2].set_xlabel('$X$')
axs[3].set_title('Deep Sparse GP')
axs[3].set_xlabel('$X$')
axs[4].set_title('Latent-Variable Deep Sparse GP')
axs[4].set_xlabel('$X$')


# Configuration]
num_inducing_points = 100
Kernel = RBF
lengthscale = 1.0
outer_variance = 1.0
inner_variance = 2.0
likelihood_variance = 0.1
learning_rate = 0.01
training_epochs = 10000
patience = 20
factor = 0.95
verbose = 0
min_learning_rate = 1e-5
num_test_points = 1024
num_function_samples = 100
X_star = np.linspace(x_lim[0], x_lim[1], num_test_points)
X_star_test = X_star[..., None]


# Create inducing variables by hand
# Z = np.array([
#     -2.18, -2.06, -1.57, -1.27, -0.89,
#     -0.40, -0.29,  0.16,  0.65,  0.62,
#      1.07,  1.38,  1.94,  2.05,  2.36
# ]).astype(default_float())
# q = np.array([
#     -2.50, -0.78, -2.50,  1.81, -0.47,
#     -0.47, -2.55,  1.81,  1.10, -0.57,
#     -2.60,  1.56, -0.62,  1.86,  0.80
# ]).astype(default_float())
# Z_aug = np.random.rand(3 * 5,).astype(default_float())
# num_inducing_points = Z.shape[0]
# axs[0].plot(Z, q, 'o', color='black')


# Shallow sparse GP
Z1 = np.linspace(min(X), max(X), num=num_inducing_points)[..., None].astype(default_float())
# Z1 = Z[..., None]
feat1 = SharedIndependentInducingVariables(InducingPoints(Z1))
kern1 = SharedIndependent(Kernel(lengthscales=lengthscale, variance=outer_variance), output_dim=1)
layer1 = GPLayer(kern1, feat1, X.shape[0], mean_function=Zero(), white=False)
# layer1.q_mu.assign(value=q[..., None])

lik_layer = LikelihoodLayer(Gaussian(variance=likelihood_variance))

model = DeepGP([layer1], lik_layer, input_dim=1, output_dim=1)
model.compile(tf.optimizers.Adam(learning_rate=learning_rate))
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=patience, factor=factor, verbose=verbose, min_lr=min_learning_rate,
    )
]
_ = model.fit(
    x=(X_train, y_train),
    y=None,
    batch_size=X.shape[0],
    epochs=training_epochs,
    callbacks=callbacks
)

model.gp_layers[0].full_cov = True
model.gp_layers[0].returns_samples = True
for i in range(num_function_samples):
    sample = model.gp_layers[0](X_star_test)[..., 0]
    sample = sample + np.random.normal(
        scale=lik_layer.likelihood.variance ** 0.5,
        size=(sample.shape[0],)
    )
    axs[1].scatter(
        X_star,
        sample,
        color=scatter_color,
        marker='.',
        alpha=scatter_alpha,
        edgecolors=edgecolors
    )


# Remember parameters
Z1_numpy = feat1.inducing_variable.Z.numpy().copy()
kern1_lengthscale_numpy = kern1.kernel.lengthscales.numpy().copy()
kern1_variance_numpy = kern1.kernel.variance.numpy().copy()
q_mu1_numpy = layer1.q_mu.numpy().copy()
q_sqrt1_numpy = layer1.q_sqrt.numpy().copy()
lik_var_numpy = lik_layer.likelihood.variance.numpy().copy()


# Deep sparse GP
# Z1 = Z[..., None]
Z1 = np.linspace(min(X), max(X), num=num_inducing_points)[..., None].astype(default_float())
feat1 = SharedIndependentInducingVariables(InducingPoints(Z1))
kern1 = SharedIndependent(Kernel(lengthscales=lengthscale, variance=inner_variance), output_dim=1)
layer1 = GPLayer(kern1, feat1, X.shape[0], white=False)
layer1.q_sqrt.assign(value=layer1.q_sqrt.numpy() * 1e-5)
# gpf.set_trainable(feat1, False)
# gpf.set_trainable(kern1, False)
# gpf.set_trainable(layer1.q_mu, False)
# gpf.set_trainable(layer1.q_sqrt, False)

# Z2 = Z[..., None]
feat2 = SharedIndependentInducingVariables(InducingPoints(Z1_numpy.copy()))
kern2 = SharedIndependent(Kernel(lengthscales=kern1_lengthscale_numpy.copy(), variance=kern1_variance_numpy.copy()), output_dim=1)
layer2 = GPLayer(kern2, feat2, X.shape[0], mean_function=Zero(), white=False)
layer2.q_mu.assign(value=q_mu1_numpy.copy())
layer2.q_sqrt.assign(value=q_sqrt1_numpy.copy())
# gpf.set_trainable(feat2, False)
# gpf.set_trainable(layer2.q_mu, False)
# gpf.set_trainable(layer2.q_sqrt, False)

lik_layer = LikelihoodLayer(Gaussian(variance=likelihood_variance))
# gpf.set_trainable(lik_layer.likelihood, False)

model = DeepGP([layer1, layer2], lik_layer, input_dim=1, output_dim=1)
model.compile(tf.optimizers.Adam(learning_rate=learning_rate))
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=patience, factor=factor, verbose=verbose, min_lr=min_learning_rate,
    )
]
_ = model.fit(
    x=(X_train, y_train),
    y=None,
    batch_size=X.shape[0],
    epochs=training_epochs,
    callbacks=callbacks
)

model.gp_layers[0].full_cov = True
model.gp_layers[0].returns_samples = True
model.gp_layers[1].full_cov = True
model.gp_layers[1].returns_samples = True
for i in range(num_function_samples):
    sample = model.gp_layers[1](model.gp_layers[0](X_star_test))[..., 0]
    sample = sample + np.random.normal(
        scale=lik_layer.likelihood.variance ** 0.5,
        size=(sample.shape[0],)
    )
    # axs[3].plot(X_star, sample, color='blue', linewidth=0.02)
    axs[3].scatter(
        X_star,
        sample,
        color=scatter_color,
        marker='.',
        alpha=scatter_alpha,
        edgecolors=edgecolors
    )


# Latent-variable shallow sparse GP
encoder = DirectlyParameterizedNormalDiag(X.shape[0], latent_dim=1)
prior_mean = np.zeros(1, dtype=default_float())
prior_var = np.ones(1, dtype=default_float())
prior = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_var)
lv_layer = LatentVariableAugmentationLayer(encoder=encoder, prior=prior)

Z = np.linspace(min(X), max(X), num=num_inducing_points)[...].astype(default_float())
Z_aug = np.random.rand(num_inducing_points).astype(default_float())
Z1 = np.stack([Z, Z_aug]).transpose()
feat1 = SharedIndependentInducingVariables(InducingPoints(Z1))
kern1 = SharedIndependent(Kernel(lengthscales=lengthscale, variance=outer_variance), output_dim=1)
layer1 = GPLayer(kern1, feat1, X.shape[0], mean_function=Zero(), white=False)

lik_layer = LikelihoodLayer(Gaussian(variance=likelihood_variance))

x = tf.keras.Input((1,))
y = tf.keras.Input((1,))
f = lv_layer((x, y), training=True)
f = layer1(f, training=True)
model = BayesianModel(
    X_input=x,
    Y_input=y,
    F_output=f,
    likelihood_layer=lik_layer,
    num_data=X.shape[0]
)
model.compile(tf.optimizers.Adam(learning_rate=learning_rate))
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=patience, factor=factor, verbose=verbose, min_lr=min_learning_rate,
    )
]
_ = model.fit(
    x=(X_train, y_train),
    y=None,
    batch_size=X.shape[0],
    epochs=training_epochs
)

layer1.full_cov = True
layer1.returns_samples = True
for i in range(num_function_samples):
    sample = layer1(lv_layer((X_star_test, None), ))[..., 0]
    sample = sample + np.random.normal(
        scale=lik_layer.likelihood.variance ** 0.5,
        size=(sample.shape[0],)
    )
    # axs[2].plot(X_star, sample, color='blue', linewidth=0.02)
    axs[2].scatter(
        X_star,
        sample,
        color=scatter_color,
        marker='.',
        alpha=scatter_alpha,
        edgecolors=edgecolors
    )


# Remember parameters
Z1_numpy = feat1.inducing_variable.Z.numpy().copy()
kern1_lengthscale_numpy = kern1.kernel.lengthscales.numpy().copy()
kern1_variance_numpy = kern1.kernel.variance.numpy().copy()
q_mu1_numpy = layer1.q_mu.numpy().copy()
q_sqrt1_numpy = layer1.q_sqrt.numpy().copy()
lik_var_numpy = lik_layer.likelihood.variance.numpy().copy()


# Latent-variable deep sparse GP
encoder = DirectlyParameterizedNormalDiag(X.shape[0], latent_dim=1)
prior_mean = np.zeros(1, dtype=default_float())
prior_var = np.ones(1, dtype=default_float())
prior = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_var)
lv_layer = LatentVariableAugmentationLayer(encoder=encoder, prior=prior)

Z = np.linspace(min(X), max(X), num=num_inducing_points)[...].astype(default_float())
Z_aug = np.random.rand(num_inducing_points).astype(default_float())
Z1 = np.stack([Z, Z_aug]).transpose()
feat1 = SharedIndependentInducingVariables(InducingPoints(Z1))
kern1 = SharedIndependent(Kernel(lengthscales=lengthscale, variance=inner_variance), output_dim=1)
layer1 = GPLayer(kern1, feat1, X.shape[0], white=False)
layer1.q_sqrt.assign(value=layer1.q_sqrt.numpy() * 1e-5)

feat2 = SharedIndependentInducingVariables(InducingPoints(Z1_numpy.copy()))
kern2 = SharedIndependent(Kernel(lengthscales=kern1_lengthscale_numpy.copy(), variance=kern1_variance_numpy.copy()), output_dim=1)
layer2 = GPLayer(kern2, feat2, X.shape[0], mean_function=Zero(), white=False)
layer2.q_mu.assign(value=q_mu1_numpy.copy())
layer2.q_sqrt.assign(value=q_sqrt1_numpy.copy())

lik_layer = LikelihoodLayer(Gaussian(variance=likelihood_variance))

x = tf.keras.Input((1,))
y = tf.keras.Input((1,))
f = lv_layer((x, y), training=True)
f = layer1(f, training=True)
f = layer2(f, training=True)
model = BayesianModel(
    X_input=x,
    Y_input=y,
    F_output=f,
    likelihood_layer=lik_layer,
    num_data=X.shape[0]
)
model.compile(tf.optimizers.Adam(learning_rate=learning_rate))
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=patience, factor=factor, verbose=verbose, min_lr=min_learning_rate,
    )
]
_ = model.fit(
    x=(X_train, y_train),
    y=None,
    batch_size=X.shape[0],
    epochs=training_epochs
)

layer1.full_cov = True
layer1.returns_samples = True
layer2.full_cov = True
layer2.returns_samples = True
for i in range(num_function_samples):
    sample = layer2(layer1(lv_layer((X_star_test, None), )))[..., 0]
    sample = sample + np.random.normal(
        scale=lik_layer.likelihood.variance ** 0.5,
        size=(sample.shape[0],)
    )
    # axs[4].plot(X_star, sample, color='blue', linewidth=0.02)
    axs[4].scatter(
        X_star,
        sample,
        color=scatter_color,
        marker='.',
        alpha=scatter_alpha,
        edgecolors=edgecolors
    )


# show plot
plt.tight_layout()
plt.show()
