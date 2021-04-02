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
# Why GPflux is a modern (deep) GP library

In this notebook we go over some of the features that make GPflux a modern GP library, such as out-of-the-box support for monitoring the optimisation and saving & serving (deep) GP models. Compared to [GPflow](www.gpflow.org), which GPflux relies on for most of the mathemetical routines, the API is more high-level. This makes typical machine learning tasks like mini-batching training and tests data, using learning rate schedulers, connecting your optimisation to TensorBoard, etc. easier.
"""

# %% [markdown]
"""
## Setting up the dataset and model

### Motorcycle: a toy one-dimensional dataset
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.keras.backend.set_floatx("float64")
tf.get_logger().setLevel("INFO")


# %%
def motorcycle_data():
    """
    The motorcycle dataset where the target are normalised to be N(0,1) distributed.
    :return: Input features with shape ``[N, 1]`` and corresponding targets with shape ``[N, 1]``.
    """
    df = pd.read_csv("./data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    return X, Y


X, Y = motorcycle_data()
plt.plot(X, Y, "kx");
plt.xlabel("time");
plt.ylabel("Acceleration");

# %% [markdown]
"""
### Two-layer deep GP

To keep this notebook focussed we are going to use a predefined deep GP architecture `gpflux.architectures.build_constant_input_dim_deep_gp` for create our simple two-layer model.
"""

# %%
import gpflux

from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

config = Config(
    num_inducing=25,
    inner_layer_qsqrt_factor=1e-5,
    likelihood_noise_variance=1e-2,
    whiten=True
)
deep_gp: DeepGP = build_constant_input_dim_deep_gp(X, 1, config=config)

# %% [markdown]
"""
## Training: mini-batching, callbacks, checkpoints and monitoring

When training a model, GPflux takes care of minibatching the dataset and accepts a range of callbacks that make modifying the learning rate or monitoring the optimisation, for example, very simple. 
"""

# %%
# From the `DeepGP` model we instantiate a training model which is a `tf.keras.Model`
training_model: tf.keras.Model = deep_gp.as_training_model()

# Following the Keras procedure we need to compile and pass a optimizer,
# before fitting the model to data
training_model.compile(optimizer=tf.optimizers.Adam(0.01))

callbacks = [
    # Create callback that reduces the learning rate every time the ELBO plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        'loss', factor=0.95, patience=3, min_lr=1e-6, verbose=0
    ),
    # Create a callback that writes logs (e.g., hyperparameters, KLs, etc.) to TensorBoard
    gpflux.callbacks.TensorBoard(),
    # Create a callback that saves the model's weights
    tf.keras.callbacks.ModelCheckpoint(
        filepath="ckpts/", save_weights_only=True, verbose=0)
]

history = training_model.fit(
    {"inputs": X, "targets": Y},
    batch_size=12,
    epochs=150,
    callbacks=callbacks,
    verbose=0,
)

# %% [markdown]
"""
Keras' fit returns a `history` object contains some information like the loss and the learning rate
"""

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
ax1.plot(history.history["loss"]);
ax1.set_xlabel("Iteration");
ax1.set_ylabel("Objective = neg. ELBO");

ax2.plot(history.history["lr"]);
ax2.set_xlabel("Iteration");
ax2.set_ylabel("Learning rate");

# %% [markdown]
"""
More insightful, however, are the TensorBoard logs. They contain the the objective and hyperparameters in the course of optimisation. This can be very handy to find out why things work or don't :D. The logs can be viewed in TensorBoard by running in the command line
```
$ tensorboard --logdir logs
```
"""


# %%
def plot(model, X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    a = 1.0
    N_test = 100
    X_test = np.linspace(X.min() - a, X.max() + a, N_test).reshape(-1, 1)
    out = model(X_test)

    mu = out.f_mean.numpy().squeeze()
    var = out.f_var.numpy().squeeze()
    X_test = X_test.squeeze()
    lower = mu - 2 * np.sqrt(var)
    upper = mu + 2 * np.sqrt(var)

    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)
    ax.plot(X, Y, "kx", alpha=0.5)
    ax.plot(X_test, mu, "C1")

    ax.fill_between(X_test, lower, upper, color="C1", alpha=0.3)

prediction_model = deep_gp.as_prediction_model()
plot(prediction_model, X, Y)

# %% [markdown]
"""
## Post-training: saving and serving the model

We can store the weights and reload them afterwards.
"""

# %%
prediction_model.save_weights("weights")

# %%
prediction_model_new = build_constant_input_dim_deep_gp(X, 1, config=config).as_prediction_model()
prediction_model_new.load_weights("weights")

# %%
plot(prediction_model_new, X, Y)

# %% [markdown]
"""
Indeed, this prediction corresponds to the one of the original model.
"""
