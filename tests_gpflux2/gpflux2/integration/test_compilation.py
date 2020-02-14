import tensorflow as tf
from tensorflow import keras

import numpy as np
import pytest

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero

from gpflux2.layers import GPLayer, LikelihoodLayer
from gpflux2.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux2.models import DeepGP

tf.keras.backend.set_floatx("float64")

#########################################
# Helpers
#########################################
def setup_dataset(input_dim: int, num_data: int):
    lim = [0, 100]
    kernel = RBF(lengthscale=20)
    sigma = 0.01
    X = np.random.uniform(low=lim[0], high=lim[1], size=(num_data, input_dim))
    cov = kernel(X) + np.eye(num_data) * sigma ** 2
    Y = np.random.multivariate_normal(np.zeros(num_data), cov)[:, None]
    Y = np.clip(Y, -0.5, 0.5).astype(np.float64)
    return X, Y


def build_gp_layers(layer_sizes, num_data):
    gp_layers = []
    for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:]):

        kernel = construct_basic_kernel(kernels=RBF(), output_dim=output_dim)
        inducing_vars = construct_basic_inducing_variables(
            num_inducing=25, input_dim=input_dim, output_dim=output_dim
        )

        layer = GPLayer(kernel, inducing_vars, num_data)
        gp_layers.append(layer)

    gp_layers[-1].mean_function = Zero()
    gp_layers[-1].use_samples = False

    return gp_layers


def build_keras_functional_deep_gp(layer_sizes, num_data):
    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood_layer = LikelihoodLayer(Gaussian())

    inputs = keras.Input(shape=(layer_sizes[0]))
    x = inputs
    for gp in gp_layers:
        x = gp(x)

    targets = keras.Input(shape=(layer_sizes[-1]))
    mean_and_var = likelihood_layer(x, targets=targets)

    return keras.Model(
        inputs=[inputs, targets], outputs=mean_and_var, name="deep_gp_fp"
    )


def build_keras_objected_oriented_deep_gp(layer_sizes, num_data):
    class KerasDeepGP(tf.keras.Model):
        def __init__(self, gp_layers, likelihood):
            super().__init__(name="deep_gp_oop")
            self.gp_layers = gp_layers
            self.likelihood = likelihood

        def call(self, data, training=None):
            inputs, targets = data
            for gp in self.gp_layers:
                inputs = gp(inputs)
            return self.likelihood(inputs, targets=targets)

    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood_layer = LikelihoodLayer(Gaussian())
    return KerasDeepGP(gp_layers, likelihood_layer)


def build_gpflux_deep_gp(layer_sizes, num_data):
    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood_layer = LikelihoodLayer(Gaussian())
    return DeepGP(gp_layers, likelihood_layer)


#########################################
# Tests
#########################################
MODEL_BUILDERS = [
    build_keras_functional_deep_gp,
    build_keras_objected_oriented_deep_gp,
    build_gpflux_deep_gp,
]


@pytest.mark.parametrize("deep_gp_model_builder", MODEL_BUILDERS)
def test_model_compilation(deep_gp_model_builder):
    """Test whether a keras functional style model compiles with a standard optimizer"""
    layer_sizes = [5, 5, 1]
    num_data = 200
    batch = 100

    (X, Y) = setup_dataset(layer_sizes[0], num_data)
    deep_gp_model = deep_gp_model_builder(layer_sizes, num_data)

    dataset_tuple = ((X, Y), Y)  # (inputs_to_model, targets_from_model)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset_tuple).batch(batch)

    optimizer = tf.keras.optimizers.Adam()

    tf.random.set_seed(0)
    np.random.seed(0)

    deep_gp_model.compile(optimizer=optimizer)
    history = deep_gp_model.fit(train_dataset, epochs=10)
    assert 4.545 < history.history["loss"][0] < 4.558
    assert 3.830 < history.history["loss"][-1] < 3.841

    test_batch, _ = next(iter(train_dataset))
    mean, var = deep_gp_model.predict(test_batch)
    assert mean.shape == (batch, 1)
    assert var.shape == (batch, 1)


@pytest.mark.parametrize("use_tf_function", [True, False])
@pytest.mark.parametrize("deep_gp_model_builder", MODEL_BUILDERS)
def test_model_eager(deep_gp_model_builder, use_tf_function):
    layer_sizes = [5, 5, 1]
    num_data = 200
    batch = 100
    deep_gp_model = deep_gp_model_builder(layer_sizes, num_data)
    (X, Y) = setup_dataset(layer_sizes[0], num_data)

    dataset_tuple = ((X, Y), Y)  # (inputs_to_model, targets_from_model)
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(dataset_tuple).repeat().batch(batch)
    )
    optimizer = tf.keras.optimizers.Adam()

    train_dataset_iter = iter(train_dataset)
    test_mini_batch, _ = next(train_dataset_iter)

    def objective(data_minibatch):
        y_pred = deep_gp_model(data_minibatch, training=True)
        return tf.reduce_sum(deep_gp_model.losses)


    def optimization_step(data_minibatch):
        optimizer.minimize(
            lambda: objective(data_minibatch), deep_gp_model.trainable_weights
        )

    if use_tf_function:
        objective = tf.function(objective, autograph=False)
        optimization_step = tf.function(optimization_step, autograph=False)

    tf.random.set_seed(0)
    np.random.seed(0)

    # TO DO: investigate why the difference between OOP vs FP models, and tf.function
    assert 4.46 < objective(test_mini_batch) < 4.59
    for i in range(20):
        data_mini_batch, _ = next(train_dataset_iter)

        optimization_step(data_mini_batch)
    assert 3.67 < objective(test_mini_batch) < 3.81

    test_batch, _ = next(iter(train_dataset))
    mean, var = deep_gp_model.predict(test_batch)
    assert mean.shape == (batch, 1)
    assert var.shape == (batch, 1)
