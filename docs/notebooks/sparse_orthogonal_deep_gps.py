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
#
# Sparse orthogonal VI is based on decomposing the GP prior as the sum of a low-rank approximation using inducing points, and a full-rank residual process. It's been observed how the standard SVGP methods can be reinterpreted under such decomposition. By introducing another set of inducing variables for the orthogonal complement, we can increase the number of inducing points at a much lower additional computational cost.

import gpflow
import gpflux
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
# GPflux provides provides a class `OrthGPLayer`, which implements a Sparse Orthogonal Variational multioutput Gaussian Process as a `tf.keras.layers.Layer`. In the following, we build a 2-layer orthogonal deep GP model using this new layer type, with a Gaussian likelihood in the output layer. A standard squared exponential kernel is used throughout the layers.

# +
from typing import Type
from gpflow.kernels import SquaredExponential, Stationary
from gpflow.mean_functions import Zero
from gpflow.likelihoods import Gaussian
from scipy.cluster.vq import kmeans2

from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)
from gpflux.layers import OrthGPLayer
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.models import OrthDeepGP


def build_kernel(input_dim: int, is_last_layer: bool, kernel: Type[Stationary]) -> Stationary:
    """
    Return a :class:`gpflow.kernels.Stationary` kernel with ARD lengthscales set to
    1.0 and a small kernel variance of 1e-6 if the kernel is part of a hidden layer;
    otherwise, the kernel variance is set to 1.0.

    :param input_dim: The input dimensionality of the layer.
    :param is_last_layer: Whether the kernel is part of the last layer in the Deep GP.
    :param kernel: the :class:`~gpflow.kernels.Stationary` type of the kernel
    """
    assert input_dim > 0, "Cannot have non positive input dimension"

    variance = 1e-6 if not is_last_layer else 1.0
    lengthscales = [1.0] * input_dim

    return kernel(lengthscales=lengthscales, variance=variance)


def build_orthogonal_deep_gp(
    num_layers: int, num_inducing_u: int, num_inducing_v: int, X: np.ndarray
) -> OrthDeepGP:
    """
    :param num_layers: the number of (hidden) layers
    :param num_inducing_u: The number of inducing points to use for the low-rank approximation
    :param num_inducing_v: The number of inducing points to use for the full-rank residual process
    :param X: the data
    """
    num_data, input_dim = X.shape
    X_running = X

    gp_layers = []
    centroids, _ = kmeans2(X, k=min(num_inducing_u + num_inducing_v, X.shape[0]), minit="points")

    centroids_u = centroids[:num_inducing_u, ...]
    centroids_v = centroids[num_inducing_u:, ...]

    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else input_dim

        inducing_var_u = construct_basic_inducing_variables(
            num_inducing=num_inducing_u,
            input_dim=D_in,
            share_variables=True,
            z_init=centroids_u,
        )

        inducing_var_v = construct_basic_inducing_variables(
            num_inducing=num_inducing_v,
            input_dim=D_in,
            share_variables=True,
            z_init=centroids_v,
        )

        kernel = construct_basic_kernel(
            kernels=build_kernel(D_in, is_last_layer, SquaredExponential),
            output_dim=D_out,
            share_hyperparams=True,
        )

        if is_last_layer:
            mean_function = Zero()
            q_sqrt_scaling = 1.0
        else:
            mean_function = construct_mean_function(X_running, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = cast(tf.Tensor, X_running).numpy()
            q_sqrt_scaling = 1e-5

        # NOTE: here we're using the specialised GPLayer
        layer = OrthGPLayer(
            kernel,
            inducing_var_u,
            inducing_var_v,
            num_data,
            mean_function=mean_function,
            name=f"orth_gp_{i_layer}",
            num_latent_gps=D_out,
        )
        layer.q_sqrt_u.assign(layer.q_sqrt_u * q_sqrt_scaling)
        layer.q_sqrt_v.assign(layer.q_sqrt_v * q_sqrt_scaling)
        gp_layers.append(layer)

    # NOTE: here we return an instance of a DeeGP type specialised for sparse orthogonal VI
    return OrthDeepGP(gp_layers, LikelihoodLayer(likelihood=Gaussian(variance=1e-2)))


# -

# ### Create the model
#
# We now instantiate one model using the above utility function. Note how we can use substantial more inducing points compared to model defined in other notebooks, for both the low-rank approximation and the full-rank residual process.

# +

orthogonal_dgp = build_orthogonal_deep_gp(num_layers=1, num_inducing_u=50, num_inducing_v=50, X=X)
gpflow.utilities.print_summary(orthogonal_dgp, fmt="notebook")
# -

# ### Model training

# +
# Fit the model on the training data

BATCH_SIZE = 32
NUM_EPOCHS = 1000

model = orthogonal_dgp.as_training_model()
model.compile(tf.optimizers.Adam(5e-2))


callbacks = [
    # Create callback that reduces the learning rate every time the ELBO plateaus
    tf.keras.callbacks.ReduceLROnPlateau("loss", factor=0.95, patience=10, min_lr=1e-6, verbose=0)
]
history = model.fit(
    {"inputs": X, "targets": Y},
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    verbose=0,
)
gpflow.utilities.print_summary(orthogonal_dgp, fmt="notebook")
# -

fig, ax = plt.subplots()
ax.plot(history.history["loss"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

# +
fig, ax = plt.subplots()
num_data_test = 200
X_test = np.linspace(X.min() - X_MARGIN, X.max() + X_MARGIN, num_data_test).reshape(-1, 1)
model = orthogonal_dgp.as_prediction_model()
out = model(X_test)

mu = out.y_mean.numpy().squeeze()
var = out.y_var.numpy().squeeze()
X_test = X_test.squeeze()

for i in [1, 2]:
    lower = mu - i * np.sqrt(var)
    upper = mu + i * np.sqrt(var)
    ax.fill_between(X_test, lower, upper, color="C1", alpha=0.3)

ax.set_ylim(Y.min() - Y_MARGIN, Y.max() + Y_MARGIN)
ax.set_xlim(X.min() - X_MARGIN, X.max() + X_MARGIN)
ax.plot(X, Y, "kx", alpha=0.5)
ax.plot(X_test, mu, "C1")
ax.set_xlabel("time")
ax.set_ylabel("acc")
# -

# ## Conclusion
#
# In this notebook we have shown how to create a variant of the deep gp model using the recently introduced sparse orthogonal variational inference of Gaussian processes in GPflux.
#
#
# ## References
#
# [1] Shi, J. et al. (2020) “Sparse Orthogonal Variational Inference for Gaussian Processes”. Proceedings of the 23rdInternational Conference on Artificial Intelligence and Statistics (AISTATS), 109.

#
