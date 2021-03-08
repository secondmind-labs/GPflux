import itertools

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero

from gpflux.encoders import DirectlyParameterizedNormalDiag
from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.layers import GPLayer, LatentVariableLayer, LikelihoodLayer
from gpflux.models import DeepGP

tf.keras.backend.set_floatx("float64")

############
# Utilities
############


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

    return gp_layers


def train_model(x_data, y_data, model, use_keras_compile):
    dataset_dict = {"inputs": x_data, "targets": y_data}
    num_data = len(x_data)

    optimizer = tf.keras.optimizers.Adam()

    epochs = 20

    if use_keras_compile:
        model.compile(optimizer=optimizer)
        history = model.fit(dataset_dict, batch_size=num_data, epochs=epochs)
        loss = history.history["loss"]

    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
        dataset_iter = iter(train_dataset.repeat().batch(num_data))

        def objective(data_minibatch):
            _ = model(data_minibatch, training=True)
            return tf.reduce_sum(model.losses)

        loss = []

        def optimization_step():
            data_batch = next(dataset_iter)
            optimizer.minimize(lambda: objective(data_batch), model.trainable_weights)
            loss.append(objective(data_batch))

        for _ in range(epochs):
            optimization_step()

    return loss


############
# Tests
############


@pytest.mark.parametrize("w_dim", [1, 2])
@pytest.mark.parametrize("use_keras_compile", [True, False])
def test_cde_direct_parametrization(test_data, w_dim, use_keras_compile):
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

    lv = LatentVariableLayer(prior, encoder)
    [gp] = build_gp_layers([x_dim + w_dim, 1], num_data)
    likelihood_layer = LikelihoodLayer(Gaussian())

    # 3. Build the model
    dgp = DeepGP([lv, gp], likelihood_layer)
    model = dgp.as_training_model()

    # 4. Train the model and check 2nd half of loss is lower than first
    loss_history = train_model(x_data, y_data, model, use_keras_compile)
    epochs = len(loss_history)
    assert np.all(loss_history[: (epochs // 2)] > loss_history[(epochs // 2) :])
