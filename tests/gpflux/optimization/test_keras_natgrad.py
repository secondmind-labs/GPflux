#
#  Copyright (c) 2021 The GPflux Contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Unit tests for the `gpflow.optimizers.NaturalGradient` optimizer within Keras models.
"""
import numpy as np
import pytest
import tensorflow as tf

import gpflow

import gpflux
from gpflux.layers.likelihood_layer import LikelihoodOutputs

tf.keras.backend.set_floatx("float64")


@pytest.fixture(name="base_model", scope="module")
def _base_model_fixture():
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = gpflow.inducing_variables.InducingPoints(np.random.random((10, 1)))
    likelihood = gpflow.likelihoods.Gaussian()
    num_data = 100

    mok = gpflow.kernels.SharedIndependent(kernel, output_dim=1)
    moiv = gpflow.inducing_variables.SharedIndependentInducingVariables(inducing_variable)
    gp_layer = gpflux.layers.GPLayer(
        mok, moiv, num_data, mean_function=gpflow.mean_functions.Zero()
    )
    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    return gpflux.models.DeepGP([gp_layer], likelihood_layer, input_dim=1, num_data=num_data)


def test_smoke_nat_grad_model_train_step(base_model):
    train_model = base_model.as_training_model()
    wrapper = gpflux.optimization.NatGradWrapper(train_model)
    wrapper.compile(optimizer=[gpflow.optimizers.NaturalGradient(1), tf.optimizers.Adam()])
    output = wrapper.train_step({"inputs": np.ones((1, 1)), "targets": np.ones((1, 1))})

    assert list(output.keys()) == ["loss", "gp_layer_prior_kl"]


def test_nat_grad_wrapper_layers(base_model):
    train_model = base_model.as_training_model()
    wrapper = gpflux.optimization.NatGradWrapper(train_model)
    assert wrapper.layers == train_model.layers


def _assert_likelihood_outputs_are_equal(l1: LikelihoodOutputs, l2: LikelihoodOutputs):
    assert isinstance(l1, LikelihoodOutputs)
    assert isinstance(l2, LikelihoodOutputs)

    assert list(vars(l1).keys()) == list(vars(l2).keys())

    for prop in vars(l1):
        if prop.startswith("_"):
            # skip private properties
            continue

        assert l1.__getattribute__(prop) == l2.__getattribute__(prop)


def test_nat_grad_wrapper_call(base_model):
    pred_model = base_model.as_prediction_model()
    wrapper = gpflux.optimization.NatGradWrapper(pred_model)
    input_values = tf.zeros((1, 1))

    actual_output = wrapper(input_values)
    expected_output = pred_model(input_values)

    _assert_likelihood_outputs_are_equal(actual_output, expected_output)
