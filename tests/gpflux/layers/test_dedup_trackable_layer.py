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
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

import gpflow
from gpflow.keras import tf_keras
from gpflow.utilities import parameter_dict

import gpflux
from gpflux.helpers import construct_gp_layer


class CONFIG:
    # prime numbers for unique products
    input_dim = 5
    hidden_dim = 11
    output_dim = 3
    num_inducing = 13
    num_data = 7


def count_params(model: tf_keras.models.Model) -> int:
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
def model(data) -> tf_keras.models.Model:
    """
    Builds a two-layer deep GP model.
    """
    X, Y = data
    num_data = len(X)
    input_dim = X.shape[-1]

    layer1 = construct_gp_layer(num_data, CONFIG.num_inducing, input_dim, CONFIG.hidden_dim)

    output_dim = Y.shape[-1]
    layer2 = construct_gp_layer(num_data, CONFIG.num_inducing, CONFIG.hidden_dim, output_dim)

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    X = tf_keras.Input((input_dim,))
    f1 = layer1(X)
    f2 = layer2(f1)
    y = likelihood_layer(f2)
    return tf_keras.Model(inputs=X, outputs=y)


def _size_q_sqrt(num_inducing, output_dim):
    return num_inducing * (num_inducing + 1) / 2 * output_dim


def _size_q_mu(num_inducing, output_dim):
    return num_inducing * output_dim


_MODEL_PARAMS_AND_SIZE = {
    "._self_tracked_trackables[1].kernel.kernel.variance": 1,
    "._self_tracked_trackables[1].kernel.kernel.lengthscales": CONFIG.input_dim,
    "._self_tracked_trackables[1].inducing_variable.inducing_variable.Z": (
        CONFIG.num_inducing * CONFIG.input_dim
    ),
    "._self_tracked_trackables[1].q_mu": _size_q_mu(CONFIG.num_inducing, CONFIG.hidden_dim),
    "._self_tracked_trackables[1].q_sqrt": _size_q_sqrt(CONFIG.num_inducing, CONFIG.hidden_dim),
    "._self_tracked_trackables[2].kernel.kernel.variance": 1,
    "._self_tracked_trackables[2].kernel.kernel.lengthscales": CONFIG.hidden_dim,
    "._self_tracked_trackables[2].inducing_variable.inducing_variable.Z": (
        CONFIG.num_inducing * CONFIG.hidden_dim
    ),
    "._self_tracked_trackables[2].q_mu": _size_q_mu(CONFIG.num_inducing, CONFIG.output_dim),
    "._self_tracked_trackables[2].q_sqrt": _size_q_sqrt(CONFIG.num_inducing, CONFIG.output_dim),
    "._self_tracked_trackables[3].likelihood.variance": 1,
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
    # We filter out the parameters of type ResourceVariable.
    # They have been added to the model by the `add_metric` call in the layer.
    parameters = [p for p in parameter_dict(model).values() if not isinstance(p, ResourceVariable)]
    variables = {id(p.unconstrained_variable) for p in parameters}

    weights = model.trainable_weights
    assert len(weights) == len(variables)

    weights_set = {id(w) for w in weights}
    assert weights_set == variables
