import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.likelihoods import Gaussian

from gpflux.layers import LatentVariableAugmentationLayer, LikelihoodLayer
from gpflux.models import BayesianModel
from tests.integration.test_latent_variable_integration import (  # noqa: F401
    build_gp_layers,
    test_data,
)

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

    return LatentVariableAugmentationLayer(build_encoder(), build_prior())


def build_LVGPGP_Bayesian_Model(x_dim, w_dim, y_dim, num_data):
    layer_dims = [x_dim + w_dim, x_dim + w_dim, y_dim]
    gp_layers = build_gp_layers(layer_dims, num_data)
    lv_layer = build_latent_layer(w_dim, x_dim, y_dim)
    likelihood_layer = LikelihoodLayer(Gaussian(0.1))

    x = tf.keras.Input((x_dim,))
    y = tf.keras.Input((y_dim,))

    f = lv_layer((x, y))
    for gp in gp_layers:
        f = gp(f)

    return BayesianModel(
        X_input=x, Y_input=y, F_output=f, likelihood_layer=likelihood_layer, num_data=num_data,
    )


@pytest.fixture
def bayesian_model(test_data):
    X, Y = test_data
    num_data, x_dim = X.shape
    _, y_dim = Y.shape
    w_dim = 1

    model = build_LVGPGP_Bayesian_Model(x_dim, w_dim, y_dim, num_data)
    model.compile()
    model.fit([X, Y], epochs=1)
    return model


def test_predict_f_shape(bayesian_model, test_data):
    X, Y = test_data
    num_data, y_dim = Y.shape
    f_mean, f_cov = bayesian_model.predict_f(X)
    assert f_mean.shape == (num_data, y_dim)
    assert f_cov.shape == (num_data, y_dim)


def test_predict_y_shape(bayesian_model, test_data):
    X, Y = test_data
    num_data, y_dim = Y.shape
    y_mean, y_cov = bayesian_model.predict_y(X)
    assert y_mean.shape == (num_data, y_dim)
    assert y_cov.shape == (num_data, y_dim)


def test_predict_f_samples_shape(bayesian_model, test_data):
    X, Y = test_data
    num_samples = 5
    num_data, y_dim = Y.shape
    f_samples = bayesian_model.predict_f_samples(X, num_samples=num_samples)
    assert f_samples.shape == (num_samples, num_data, y_dim)
