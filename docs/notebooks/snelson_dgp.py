"""
# Why GPflux is a modern (deep) GP library

In this notebook we go over some of the features that make GPflux a powerful, deep-learning-style GP library. We demonstrate the out-of-the-box support for monitoring during the course of optimisation, adapting the learning rate, and saving & serving (deep) GP models.
"""

"""
## Setting up the dataset and model

### Motorcycle: a toy one-dimensional dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os

tf.keras.backend.set_floatx("float64")  # we want to carry out GP calculations in 64 bit
tf.get_logger().setLevel("INFO")

class ToyData1D(object):
    def __init__(self, train_x, train_y, test_x, normalize=False, 
                 dtype=np.float64):
        self.train_x = np.array(train_x, dtype=dtype)[:, None]
        self.train_y = np.array(train_y, dtype=dtype)[:, None]
        self.n_train = self.train_x.shape[0]
        self.test_x = np.array(test_x, dtype=dtype)[:, None]
        self.x_min = np.min(test_x)
        self.x_max = np.max(test_x)
        self.n_test = self.test_x.shape[0]
        if normalize:
            self.normalize()

    def normalize(self):
        self.mean_x = np.mean(self.train_x, axis=0, keepdims=True)
        self.std_x = np.std(self.train_x, axis=0, keepdims=True) + 1e-6
        self.mean_y = np.mean(self.train_y, axis=0, keepdims=True)
        self.std_y = np.std(self.train_y, axis=0, keepdims=True) + 1e-6

        for x in [self.train_x, self.test_x]:
            x -= self.mean_x
            x /= self.std_x

        for x in [self.x_min, self.x_max]:
            x -= self.mean_x.squeeze()
            x /= self.std_x.squeeze()

        self.train_y -= self.mean_y
        self.train_y /= self.std_y

    
def load_snelson_data(n=100, dtype=np.float64):
    def _load_snelson(filename):
        with open(os.path.join("/home/sebastian.popescu/Desktop/my_code/GP_package/docs/notebooks","data", "snelson", filename), "r") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")],
                            dtype=dtype)

    np.random.seed(7)

    train_x = _load_snelson("train_inputs")
    train_y = _load_snelson("train_outputs")
    test_x = _load_snelson("test_inputs")
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm][:n]
    train_y = train_y[perm][:n]
    return ToyData1D(train_x, train_y, test_x=test_x)


toy = load_snelson_data(n=100)
X, Y = toy.train_x, toy.train_y

plt.plot(X, Y, "kx")
plt.xlabel("time")
plt.ylabel("Acceleration")


"""
### Two-layer deep GP

To keep this notebook focussed we are going to use a predefined deep GP architecture `gpflux.architectures.build_constant_input_dim_deep_gp` for creating our simple two-layer model.
"""

import gpflux

from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

config = Config(
    num_inducing=25, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-2, whiten=True
)
deep_gp: DeepGP = build_constant_input_dim_deep_gp(X, num_layers=2, config=config)


"""
## Training: mini-batching, callbacks, checkpoints and monitoring

When training a model, GPflux takes care of minibatching the dataset and accepts a range of callbacks that make it very simple to, for example, modify the learning rate or monitor the optimisation. 
"""


# From the `DeepGP` model we instantiate a training model which is a `tf.keras.Model`
training_model: tf.keras.Model = deep_gp.as_training_model()

# Following the Keras procedure we need to compile and pass a optimizer,
# before fitting the model to data
training_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01))

callbacks = [
    # Create callback that reduces the learning rate every time the ELBO plateaus
    tf.keras.callbacks.ReduceLROnPlateau("loss", factor=0.95, patience=3, min_lr=1e-6, verbose=0),
    # Create a callback that writes logs (e.g., hyperparameters, KLs, etc.) to TensorBoard
    gpflux.callbacks.TensorBoard(),
    # Create a callback that saves the model's weights
    tf.keras.callbacks.ModelCheckpoint(filepath="ckpts/", save_weights_only=True, verbose=0),
]

history = training_model.fit(
    {"inputs": X, "targets": Y},
    batch_size=12,
    epochs=200,
    callbacks=callbacks,
    verbose=1,
)


"""
The call to fit() returns a `history` object that contains information like the loss and the learning rate over the course of optimisation.
"""


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
ax1.plot(history.history["loss"])
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Objective = neg. ELBO")

ax2.plot(history.history["lr"])
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Learning rate")


"""
More insightful, however, are the TensorBoard logs. They contain the objective and hyperparameters over the course of optimisation. This can be very handy to find out why things work or don't :D. The logs can be viewed in TensorBoard by running in the command line
```
$ tensorboard --logdir logs
```
"""

def plot(model, X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    x_margin = 1.0
    N_test = 100
    X_test = np.linspace(X.min() - x_margin, X.max() + x_margin, N_test).reshape(-1, 1)
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
    plt.savefig('./snelson_preds.png')
    plt.close()

prediction_model = deep_gp.as_prediction_model()
plot(prediction_model, X, Y)

