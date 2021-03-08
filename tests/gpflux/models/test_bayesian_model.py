import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import default_float
from gpflow.likelihoods import Gaussian

from gpflux.layers import LatentVariableLayer, LikelihoodLayer
from tests.integration.test_latent_variable_integration import build_gp_layers  # noqa: F401

tf.keras.backend.set_floatx("float64")

MAXITER = int(80e3)
PLOTTER_INTERVAL = 60


def build_latent_layer(w_dim, x_dim, y_dim):
    def build_encoder():
        inputs = tf.keras.Input((x_dim + y_dim,))
        x1 = tf.keras.layers.Dense(100)(inputs)
        x2 = tf.keras.layers.Dense(20)(x1)
        mean = tf.keras.layers.Dense(w_dim, activation="linear", name="output_mean")(x2)
        std = tf.keras.layers.Dense(w_dim, activation="softplus", name="output_std")(x2)
        return tf.keras.Model(inputs=[inputs], outputs=[mean, std])

    def build_prior():
        mean = np.zeros(w_dim)
        std = np.ones(w_dim)
        return tfp.distributions.MultivariateNormalDiag(mean, std)

    return LatentVariableLayer(build_prior(), build_encoder())


def build_LVGPGP_model(x_dim, w_dim, y_dim, num_data):
    lv_layer = build_latent_layer(w_dim, x_dim, y_dim)
    layer_dims = [x_dim + w_dim, x_dim + w_dim, y_dim]
    gp_layers = build_gp_layers(layer_dims, num_data)
    likelihood_layer = LikelihoodLayer(Gaussian(0.1))
    return DeepGP([lv_layer] + gp_layers, likelihood_layer, num_data=num_data)


@pytest.fixture
def dgp_model(test_data):
    X, Y = test_data
    num_data, x_dim = X.shape
    _, y_dim = Y.shape
    w_dim = 1

    return build_LVGPGP_model(x_dim, w_dim, y_dim, num_data)
    # model = dgp.as_training_model()
    # model.compile()
    # model.fit({"inputs": X, "targets": Y}, epochs=1)
    # return model
