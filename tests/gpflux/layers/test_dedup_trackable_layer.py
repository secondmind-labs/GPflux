# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflux
from gpflow.utilities import parameter_dict
from utils import build_gp_layer
from tensorflow.python.util import object_identity


class CONFIG:
    # prime numbers for unique products
    input_dim = 5
    hidden_dim = 11
    output_dim = 3
    num_inducing = 13
    num_data = 7


def count_params(model: tf.keras.models.Model) -> int:
    """
    Counts the total number of scalar parameters in a Model.

    :param model: count the number of scalar parameters for `model`.
    :return: integer value with the total number of trainable scalar weights.
    """
    return int(sum(np.prod(p.shape.as_list()) for p in model.trainable_weights))


@pytest.fixture
def data() -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(CONFIG.num_data, CONFIG.input_dim)
    Y = np.random.randn(CONFIG.num_data, CONFIG.output_dim)
    return (X, Y)


@pytest.fixture
def model(data) -> tf.keras.models.Model:
    """
    Builds a two-layer deep GP model.
    """
    X, Y = data
    num_data = len(X)
    input_dim = X.shape[-1]

    layer1 = build_gp_layer(num_data, CONFIG.num_inducing, input_dim, CONFIG.hidden_dim)
    layer1.returns_samples = True

    output_dim = Y.shape[-1]
    layer2 = build_gp_layer(
        num_data, CONFIG.num_inducing, CONFIG.hidden_dim, output_dim
    )
    layer2.returns_samples = False

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    X = tf.keras.Input((input_dim,))
    f1 = layer1(X)
    f2 = layer2(f1)
    y = likelihood_layer(f2, targets=Y)
    return tf.keras.Model(inputs=X, outputs=y)


def _size_q_sqrt(num_inducing, output_dim):
    return num_inducing * (num_inducing + 1) / 2 * output_dim


def _size_q_mu(num_inducing, output_dim):
    return num_inducing * output_dim


_MODEL_PARAMS_AND_SIZE = {
    "._layers[1].kernel.kernel.variance": 1,
    "._layers[1].kernel.kernel.lengthscales": CONFIG.input_dim,
    "._layers[1].inducing_variable.inducing_variable.Z": (
        CONFIG.num_inducing * CONFIG.input_dim
    ),
    "._layers[1].q_mu": _size_q_mu(CONFIG.num_inducing, CONFIG.hidden_dim),
    "._layers[1].q_sqrt": _size_q_sqrt(CONFIG.num_inducing, CONFIG.hidden_dim),
    "._layers[2].kernel.kernel.variance": 1,
    "._layers[2].kernel.kernel.lengthscales": CONFIG.hidden_dim,
    "._layers[2].inducing_variable.inducing_variable.Z": (
        CONFIG.num_inducing * CONFIG.hidden_dim
    ),
    "._layers[2].q_mu": _size_q_mu(CONFIG.num_inducing, CONFIG.output_dim),
    "._layers[2].q_sqrt": _size_q_sqrt(CONFIG.num_inducing, CONFIG.output_dim),
    "._layers[3].likelihood.variance": 1,
}


def test_count_weights(model):
    """
    We build a relatively complex two-layer deep GP model and check
    that the number of scalar parameters (weights) in `model.trainable_weights`
    corresponds to a manual check.
    """
    assert count_params(model) == int(sum(_MODEL_PARAMS_AND_SIZE.values()))


def test_parameter_names(model):
    """
    Check that the parameter names returned by `gpflow.utilities.parameter_dict`
    match with our expectation.
    """
    params = parameter_dict(model)

    # We check for subset because `parameter_dict` returns multiple versions of the
    # variance.likelihood parameter with different names.
    # This is, for example, visible when using gpflow.utilities.print_summary(model).
    # However, we don't need to worry because the duplicates are correctly being filtered
    # out for any other operation that cannot deal with duplicate weights.
    # TODO(Vincent): Fix this in GPflow, possibly using `tensorflow.python.util.object_identity`.
    assert set(_MODEL_PARAMS_AND_SIZE.keys()).issubset(set(params.keys()))


def test_number_of_parameters(model):
    assert len(_MODEL_PARAMS_AND_SIZE) == len(model.trainable_weights)


def test_weights_equals_deduplicated_parameter_dict(model):
    """
    Checks GPflux's `model.trainable_weights` elements equals deduplicated
    GPflow's `gpflow.utilities.parameter_dict(model)`.
    """
    parameters = parameter_dict(model).values()
    variables = map(lambda p: p.unconstrained_variable, parameters)
    deduplicate_variables = object_identity.ObjectIdentitySet(variables)

    weights = model.trainable_weights
    assert len(weights) == len(deduplicate_variables)

    weights_set = object_identity.ObjectIdentitySet(weights)
    assert weights_set == deduplicate_variables
