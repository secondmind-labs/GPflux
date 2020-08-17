# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import tensorflow as tf
import gpflow
import gpflux
import pio_mllib.optimizers
from gpflow.ci_utils import ci_niter

import matplotlib.pyplot as plt

# %%
tf.keras.backend.set_floatx("float64")

# %%
# %matplotlib inline

# %%
d = np.load("../tests/snelson1d.npz")
X, Y = data = d["X"], d["Y"]
num_data, input_dim = X.shape
_, output_dim = Y.shape

# %%
plt.figure()
plt.plot(X, Y, ".")
plt.show()


# %%
def create_layers():
    num_inducing = 13
    hidden_dim = 1

    init_kmeans = gpflux.initializers.KmeansInitializer(X, num_inducing)
    layer1 = gpflux.helpers.construct_gp_layer(
        num_data, num_inducing, input_dim, hidden_dim, initializer=init_kmeans
    )
    layer1.mean_function = (
        gpflow.mean_functions.Identity()
    )  # TODO: pass layer_type instead
    layer1.q_sqrt.assign(layer1.q_sqrt * 0.01)

    init_last_layer = gpflux.initializers.FeedForwardInitializer()
    layer2 = gpflux.helpers.construct_gp_layer(
        num_data, num_inducing, hidden_dim, output_dim, initializer=init_last_layer,
    )
    layer2.returns_samples = False  # TODO: pass layer_type instead

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    return layer1, layer2, likelihood_layer


# %%
def create_model(model_class):
    layer1, layer2, likelihood_layer = create_layers()

    inputs = tf.keras.Input((input_dim,))
    targets = tf.keras.Input((output_dim,))
    f1 = layer1(inputs)
    f2 = layer2(f1)
    outputs = likelihood_layer(f2, targets=targets)

    model = model_class(inputs=(inputs, targets), outputs=outputs)
    return model


# %%
batch_size = 2
num_epochs = ci_niter(200)

# %%
dgp = create_model(tf.keras.Model)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=5, factor=0.95, verbose=1, min_lr=1e-6,
    )
]

dgp.compile(tf.optimizers.Adam(learning_rate=0.1))

history = dgp.fit(
    x=data, y=None, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks
)

# %%
dgp_natgrad = create_model(gpflux.optimization.NatGradModel)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=5, factor=0.95, verbose=1, min_lr=1e-6,
    )
]

dgp_natgrad.compile(
    [
        pio_mllib.optimizers.MomentumNaturalGradient(gamma=0.05, beta1=0.9, beta2=0.99),
        pio_mllib.optimizers.MomentumNaturalGradient(gamma=0.05, beta1=0.9, beta2=0.99),
        tf.optimizers.Adam(learning_rate=0.1),
    ]
)

history_natgrad = dgp_natgrad.fit(
    x=data, y=None, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks
)

# %%
res = dgp((X, np.zeros_like(Y)))

# %%
plt.plot(X, Y, "x")
plt.errorbar(X.squeeze(), np.squeeze(res[0]), np.sqrt(np.squeeze(res[1])), ls="")
plt.show()

# %%
plt.plot(history.history["loss"], label="Adam")
plt.plot(history_natgrad.history["loss"], label="NatGrad")
plt.show()