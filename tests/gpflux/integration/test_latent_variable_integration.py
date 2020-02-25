import itertools

import numpy as np
import pytest

import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, ReLU

from gpflow.kernels import RBF
from gpflow.kullback_leiblers import gauss_kl
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero

from gpflux.layers import (
    LatentVariableAugmentationLayer,
    LatentVariableLayer,
    GPLayer,
    LikelihoodLayer,
)
from gpflux.encoders import DirectlyParameterizedNormalDiag
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.models import DeepGP


tf.keras.backend.set_floatx("float64")

############
# Utilities
############


@pytest.fixture
def test_data():
    x_dim, y_dim, w_dim = 2, 1, 2
    points = 200
    x_data = np.random.random((points, x_dim)) * 5
    w_data = np.random.random((points, w_dim))
    w_data[: (points // 2), :] = 0.2 * w_data[: (points // 2), :] + 5

    input_data = np.concatenate([x_data, w_data], axis=1)
    y_data = np.random.multivariate_normal(
        mean=np.zeros(points), cov=RBF(variance=0.1).K(input_data), size=y_dim
    ).T
    return x_data[:, :x_dim], y_data


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
    gp_layers[-1].returns_samples = False

    return gp_layers


def train_model(x_data, y_data, model, use_keras_compile):
    dataset_tuple = ((x_data, y_data), y_data)  # (inputs_to_model, targets_from_model)
    num_data = len(x_data)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset_tuple).batch(num_data)

    optimizer = tf.keras.optimizers.Adam()

    epochs = 20
    if use_keras_compile:
        model.compile(optimizer=optimizer)
        history = model.fit(train_dataset, epochs=epochs)
        loss = history.history["loss"]
    else:
        train_dataset_iter = iter(train_dataset)
        data_batch, _ = next(train_dataset_iter)

        def objective(data_minibatch):
            y_pred = model(data_minibatch, training=True)
            return tf.reduce_sum(model.losses)

        def optimization_step():
            optimizer.minimize(lambda: objective(data_batch), model.trainable_weights)

        loss = []
        for i in range(epochs):
            loss.append(objective(data_batch))
            optimization_step()
    return loss


############
# Tests
############


@pytest.mark.parametrize(
    "w_dim, use_keras_compile, do_augmentation",
    itertools.product([1, 2], [True, False], [True, False]),
)
def test_cde_direct_parametrization(
    test_data, w_dim, use_keras_compile, do_augmentation
):
    """Test a directly parameterized CDE, using functional API, both eager or compiled.
    Test that the losses decrease."""

    tf.random.set_seed(0)
    np.random.seed(0)

    # 1. Set up data
    x_data, y_data = test_data
    num_data, x_dim = x_data.shape

    # 2. Set up layers
    prior_means = np.zeros(w_dim)
    prior_std = np.ones(w_dim)
    encoder = DirectlyParameterizedNormalDiag(num_data, w_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)

    if do_augmentation:
        lv = LatentVariableAugmentationLayer(encoder, prior)
    else:
        lv = LatentVariableLayer(encoder, prior)
    [gp] = build_gp_layers([x_dim + w_dim, 1], num_data)
    likelihood = LikelihoodLayer(Gaussian())

    # 3. Build the model w functional API:
    inputs = keras.Input(shape=(x_dim))
    targets = keras.Input(shape=(1))

    if do_augmentation:
        x_and_w = lv([inputs, targets])
    else:
        encoder_data = tf.concat([inputs, targets], -1)
        w = lv(encoder_data)
        x_and_w = tf.concat([inputs, w], -1)
    f = gp(x_and_w)
    mean_and_var = likelihood(f, targets=targets)

    model = tf.keras.Model(inputs=[inputs, targets], outputs=mean_and_var)

    # 4. Train the model and check 2nd half of loss is lower than first
    loss_history = train_model(x_data, y_data, model, use_keras_compile)
    epochs = len(loss_history)
    assert np.all(loss_history[: (epochs // 2)] > loss_history[(epochs // 2) :])
