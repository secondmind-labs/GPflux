import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero

from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.layers import GPLayer, LikelihoodLayer, LikelihoodLoss
from gpflux.models import DeepGP

tf.keras.backend.set_floatx("float64")


#########################################
# Helpers
#########################################


def setup_dataset(input_dim: int, num_data: int):
    lim = [0, 100]
    kernel = RBF(lengthscales=20)
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
        layer.q_sqrt.assign(0.01 * layer.q_sqrt)
        gp_layers.append(layer)

    gp_layers[-1].mean_function = Zero()

    return gp_layers


def build_keras_functional_deep_gp(layer_sizes, num_data):
    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood = Gaussian()
    likelihood_layer = LikelihoodLayer(likelihood)

    inputs = keras.Input(shape=(layer_sizes[0]))
    x = inputs
    for gp in gp_layers:
        x = gp(x)

    mean_and_var = likelihood_layer(x)

    return (
        keras.Model(inputs=[inputs], outputs=mean_and_var, name="deep_gp_fp"),
        likelihood,
    )


def build_keras_objected_oriented_deep_gp(layer_sizes, num_data):
    class KerasDeepGP(tf.keras.Model):
        def __init__(self, gp_layers, likelihood):
            super().__init__(name="deep_gp_oop")
            self.gp_layers = gp_layers
            self.likelihood = likelihood

        def call(self, data, training=None, mask=None):
            for gp in self.gp_layers:
                data = gp(data)
            return self.likelihood(data)

    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood = Gaussian()
    likelihood_layer = LikelihoodLayer(likelihood)
    return KerasDeepGP(gp_layers, likelihood_layer), likelihood


def build_gpflux_deep_gp(layer_sizes, num_data):
    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood = Gaussian()
    likelihood_layer = LikelihoodLayer(likelihood)
    return DeepGP(gp_layers, likelihood_layer), likelihood


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
    tf.random.set_seed(0)
    np.random.seed(0)

    layer_sizes = [5, 5, 1]
    num_data = 200
    batch = 100

    (X, Y) = setup_dataset(layer_sizes[0], num_data)
    deep_gp_model, likelihood = deep_gp_model_builder(layer_sizes, num_data)

    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch)

    optimizer = tf.keras.optimizers.Adam()

    deep_gp_model.compile(optimizer=optimizer, loss=LikelihoodLoss(likelihood))
    history = deep_gp_model.fit(train_dataset, epochs=10)
    assert 4.545 < history.history["loss"][0] < 4.558
    assert 3.830 < history.history["loss"][-1] < 3.841

    test_batch, _ = next(iter(train_dataset))

    output = deep_gp_model(test_batch)
    mean, var = output.y_mean, output.y_var

    assert mean.shape == (batch, 1)
    assert var.shape == (batch, 1)


@pytest.mark.parametrize("use_tf_function", [True, False])
@pytest.mark.parametrize("deep_gp_model_builder", MODEL_BUILDERS)
def test_model_eager(deep_gp_model_builder, use_tf_function):
    tf.random.set_seed(0)
    np.random.seed(0)

    layer_sizes = [5, 5, 1]
    num_data = 200
    batch = 100
    deep_gp_model, likelihood = deep_gp_model_builder(layer_sizes, num_data)
    (X, Y) = setup_dataset(layer_sizes[0], num_data)

    dataset_tuple = (X, Y)  # (inputs_to_model, targets_from_model)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset_tuple).repeat().batch(batch)
    optimizer = tf.keras.optimizers.Adam()

    train_dataset_iter = iter(train_dataset)
    test_mini_batch = next(train_dataset_iter)

    deep_gp_model.compile(loss=LikelihoodLoss(likelihood))

    def objective(data_minibatch):
        predictions_minibatch = deep_gp_model(data_minibatch[0], training=True)
        model_loss = deep_gp_model.loss(data_minibatch[1], predictions_minibatch)
        return tf.reduce_sum(deep_gp_model.losses) + model_loss

    def optimization_step(data_minibatch):
        optimizer.minimize(lambda: objective(data_minibatch), deep_gp_model.trainable_weights)

    if use_tf_function:
        objective = tf.function(objective, autograph=False)
        optimization_step = tf.function(optimization_step, autograph=False)

    # TO DO: investigate why the difference between OOP vs FP models, and tf.function
    assert 4.46 < objective(test_mini_batch) < 4.59
    for i in range(20):
        data_mini_batch = next(train_dataset_iter)

        optimization_step(data_mini_batch)
    assert 3.67 < objective(test_mini_batch) < 3.81

    test_batch, _ = next(iter(train_dataset))
    output = deep_gp_model(test_batch)
    mean, var = output.f_mu, output.f_var

    assert mean.shape == (batch, 1)
    assert var.shape == (batch, 1)


def test_deep_gp_model_default_loss():
    tf.random.set_seed(0)
    np.random.seed(0)

    layer_sizes = [5, 5, 1]
    num_data = 200
    batch = 100
    deep_gp_model, likelihood = build_gpflux_deep_gp(layer_sizes, num_data)

    (X, Y) = setup_dataset(layer_sizes[0], num_data)
    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch)

    optimizer = tf.keras.optimizers.Adam()

    deep_gp_model.compile(optimizer=optimizer)
    history = deep_gp_model.fit(train_dataset, epochs=10)
    assert 4.545 < history.history["loss"][0] < 4.558
    assert 3.830 < history.history["loss"][-1] < 3.841

    test_batch, _ = next(iter(train_dataset))

    output = deep_gp_model(test_batch)
    mean, var = output.y_mean, output.y_var

    assert mean.shape == (batch, 1)
    assert var.shape == (batch, 1)
