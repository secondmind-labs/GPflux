import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
import gpflux

from notebooks.ci_utils import is_running_pytest
from gpflux.vish.helpers import (
    get_max_degree_closest_but_smaller_than_num_inducing,
    preprocess_data
)
from gpflux.vish.inducing_variables import SphericalHarmonicInducingVariable
from gpflux.vish.kernels import ArcCosine, Matern, Parameterised
from gpflux.vish.spherical_harmonics import SphericalHarmonicsCollection


tf.keras.backend.set_floatx('float64')


class Config:
    max_num_inducing = 20 if is_running_pytest() else 1024
    max_degree, num_inducing = (
        # Snelson is one dimensional + 2 bias dimensions
        get_max_degree_closest_but_smaller_than_num_inducing("matern", 3, max_num_inducing)
    )
    num_epochs = 20 if is_running_pytest() else 1000
    batch_size = 200
    lr = 1e-2


def get_snelson():
    path = os.path.join(".", "snelson1d.npz")
    data = np.load(path)
    return (data["X"], data["Y"])


def build_model(data) -> tf.keras.models.Model:
    truncation_level = 40
    X, Y = data
    num_data, dimension = X.shape
    kernel = Matern(
        dimension,
        truncation_level=truncation_level,
        nu=1.5,
        weight_variances=np.ones(dimension)
    )
    degrees = range(Config.max_degree)
    
    harmonics = SphericalHarmonicsCollection(dimension, degrees=degrees)
    inducing_variable = SphericalHarmonicInducingVariable(harmonics)

    gp_layer = gpflux.layers.GPLayer(
        kernel=kernel,
        inducing_variable=inducing_variable,
        mean_function=gpflow.mean_functions.Zero(),
        num_data=num_data,
        num_latent_gps=Y.shape[1],
        returns_samples=False,
        verify=False,
        white=False,
    )
    gp_layer._initialized = True

    likelihood_layer = gpflux.layers.LikelihoodLayer(
        gpflow.likelihoods.Gaussian(1.0)
    )

    inputs = tf.keras.Input((dimension,), name="inputs")
    targets = tf.keras.Input((1,), name="targets")

    f1 = gp_layer(inputs)
    outputs = likelihood_layer(f1, targets=targets)
    return tf.keras.Model(inputs=(inputs, targets), outputs=outputs)


import time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train_model(model, data_train, data_test):

    path = "."
    time_cb = TimeHistory()
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=5, factor=0.95, verbose=1, min_lr=1e-6,
        ),
        gpflux.callbacks.TensorBoard(path),
        time_cb
    ]
    try:
        data_test = ((data_test[0], data_test[1] * np.nan), data_test[1])
        history = model.fit(
            x=data_train,
            y=None,
            # y=data_train[1],
            batch_size=Config.batch_size,
            epochs=Config.num_epochs,
            # validation_data=data_test,
            callbacks=callbacks,
        )
        # plt.plot(history.history["loss"])
    except KeyboardInterrupt:
        print("Training stopped")
    print(time_cb.times)
    return history


if __name__ == "__main__":
    data, _, _ = preprocess_data(get_snelson())
    model = build_model(data)
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.lr)
    model.compile(optimizer=optimizer)
    model.summary()
    gpflow.utilities.print_summary(model)
    hist = train_model(model, data, data)

    import matplotlib.pyplot as plt

    X_data = data[0][:, :1]
    Y_data = data[1]
    N_test = 100
    X_new = np.linspace(X_data.min() - 0.5, X_data.max() + 0.5, N_test).reshape(-1, 1)  # [N_test, 1]
    X_new = np.c_[X_new, np.ones((N_test, 2))]  # [N_test, 3]
    Y_nan = np.ones((len(X_new), 1)) * np.nan
    data_new = (X_new, Y_nan)
    y_mean, y_var = model.predict(data_new)

    plt.ylim(-5, 5)
    plt.plot(X_data, Y_data, "o")
    plt.plot(X_new, y_mean, "C1")
    plt.plot(X_new, y_mean + np.sqrt(y_var + 1e-6), "C1--")
    plt.plot(X_new, y_mean - np.sqrt(y_var + 1e-6), "C1--")

    plt.show()
