#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero

from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.layers import GPLayer, LikelihoodLayer, TrackableLayer
from gpflux.losses import LikelihoodLoss
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

    inputs = keras.Input(shape=(layer_sizes[0]))
    x = inputs
    for gp in gp_layers:
        x = gp(x)

    container = TrackableLayer()
    container.likelihood = likelihood
    outputs = container(x)  # to track likelihood

    model = tf.keras.Model(inputs=[inputs], outputs=outputs, name="deep_gp_fp")
    loss = LikelihoodLoss(likelihood)
    return model, loss


def build_keras_objected_oriented_deep_gp(layer_sizes, num_data):
    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood = Gaussian()

    class KerasDeepGP(tf.keras.Model, TrackableLayer):
        def __init__(self, gp_layers, likelihood):
            super().__init__(name="deep_gp_oop")
            self.gp_layers = gp_layers
            self.likelihood = likelihood

        def call(self, data, training=None, mask=None):
            for gp in self.gp_layers:
                data = gp(data)
            return data

    model = KerasDeepGP(gp_layers, likelihood)
    loss = LikelihoodLoss(likelihood)
    return model, loss


def build_gpflux_deep_gp(layer_sizes, num_data):
    gp_layers = build_gp_layers(layer_sizes, num_data)
    likelihood = Gaussian()

    likelihood_layer = LikelihoodLayer(likelihood)
    model = DeepGP(gp_layers, likelihood_layer).as_training_model()
    return model, None


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
    deep_gp_model, loss = deep_gp_model_builder(layer_sizes, num_data)

    if loss is None:  # build_gpflux_deep_gp
        dataset = {"inputs": X, "targets": Y}
    else:
        dataset = (X, Y)

    train_dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch)

    optimizer = tf.keras.optimizers.Adam()

    deep_gp_model.compile(optimizer=optimizer, loss=loss)

    history = deep_gp_model.fit(train_dataset, epochs=10)
    assert 4.545 < history.history["loss"][0] < 4.558
    assert 3.829 < history.history["loss"][-1] < 3.841

    # Check outputs

    if loss is None:  # build_gpflux_deep_gp
        test_batch_dict = next(iter(train_dataset))
        test_batch_dict["targets"] = np.full_like(test_batch_dict["targets"].shape, np.nan)
        output = deep_gp_model(test_batch_dict)
        mean, var = output.f_mean, output.f_var
    else:
        test_batch, _ = next(iter(train_dataset))
        output = deep_gp_model(test_batch)
        mean, var = output.loc, output.scale.diag ** 2

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

    (X, Y) = setup_dataset(layer_sizes[0], num_data)
    deep_gp_model, loss = deep_gp_model_builder(layer_sizes, num_data)

    if loss is None:  # build_gpflux_deep_gp
        dataset = {"inputs": X, "targets": Y}
    else:
        dataset = (X, Y)

    train_dataset = tf.data.Dataset.from_tensor_slices(dataset).repeat().batch(batch)

    optimizer = tf.keras.optimizers.Adam()

    train_dataset_iter = iter(train_dataset)
    test_mini_batch = next(train_dataset_iter)

    deep_gp_model.compile(loss=loss)

    if loss is None:

        def objective(data_minibatch):
            _ = deep_gp_model(data_minibatch, training=True)
            return tf.reduce_sum(deep_gp_model.losses)

    else:

        def objective(data_minibatch):
            predictions_minibatch = deep_gp_model(data_minibatch[0], training=True)
            model_loss = deep_gp_model.loss(data_minibatch[1], predictions_minibatch)
            return tf.reduce_sum(deep_gp_model.losses) + model_loss

    if use_tf_function:
        objective = tf.function(objective)

    def optimization_step(data_minibatch):
        optimizer.minimize(lambda: objective(data_minibatch), deep_gp_model.trainable_weights)

    if use_tf_function:
        optimization_step = tf.function(optimization_step)

    # TO DO: investigate why the difference between OOP vs FP models, and tf.function
    assert 4.46 < objective(test_mini_batch) < 4.5902
    for i in range(20):
        data_mini_batch = next(train_dataset_iter)

        optimization_step(data_mini_batch)
    assert 3.67 < objective(test_mini_batch) < 3.81

    # Check outputs
    if loss is None:  # build_gpflux_deep_gp
        test_batch_dict = next(iter(train_dataset))
        test_batch_dict["targets"] = np.full_like(test_batch_dict["targets"].shape, np.nan)
        output = deep_gp_model(test_batch_dict)
        mean, var = output.f_mean, output.f_var
    else:
        test_batch, _ = next(iter(train_dataset))
        output = deep_gp_model(test_batch)
        mean, var = output.loc, output.scale.diag ** 2

    assert mean.shape == (batch, 1)
    assert var.shape == (batch, 1)
