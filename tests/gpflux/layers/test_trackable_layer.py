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
from typing import List

import pytest
import tensorflow as tf
from tensorflow.keras.layers import Layer

from gpflow import default_float
from gpflow.kernels import RBF, Matern12, Matern52

from gpflux.layers import TrackableLayer


class CompositeModule(tf.Module):
    def __init__(self, attributes):
        for i, a in enumerate(attributes):
            setattr(self, f"var_{i}", a)
        super().__init__()


class UntrackableCompositeLayer(Layer):
    def __init__(self, attributes):
        for i, a in enumerate(attributes):
            setattr(self, f"var_{i}", a)
        super().__init__()


class TrackableCompositeLayer(TrackableLayer):
    def __init__(self, attributes):
        for i, a in enumerate(attributes):
            setattr(self, f"var_{i}", a)
        super().__init__()


def setup_layer_modules_variables():
    variables = [
        tf.Variable(4.0, dtype=default_float()),
        tf.Variable(5.0, dtype=default_float(), trainable=False),
    ]
    modules = [
        RBF(),
        CompositeModule(attributes=[Matern12()]),
        UntrackableCompositeLayer(
            attributes=[
                tf.Variable(6.0, dtype=default_float()),
                tf.Variable(7.0, dtype=default_float(), trainable=False),
            ]
        ),
        [
            CompositeModule(attributes=[Matern52()]),
            CompositeModule(attributes=[Matern52()]),
        ],
    ]

    modules_variables = [
        modules[0].variance.unconstrained_variable,
        modules[0].lengthscales.unconstrained_variable,
        modules[1].var_0.variance.unconstrained_variable,
        modules[1].var_0.lengthscales.unconstrained_variable,
        modules[2].var_0,
        modules[2].var_1,
        modules[3][0].var_0.variance.unconstrained_variable,
        modules[3][0].var_0.lengthscales.unconstrained_variable,
        modules[3][1].var_0.variance.unconstrained_variable,
        modules[3][1].var_0.lengthscales.unconstrained_variable,
    ]

    attributes = variables + modules
    trackable_layer = TrackableCompositeLayer(attributes=attributes)

    flat_modules = modules[:3] + modules[3]
    return trackable_layer, variables, flat_modules, modules_variables


def to_tensor_set(tensor_set: List[tf.Tensor]):
    return set([t.experimental_ref() for t in tensor_set])


def test_submodule_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    assert to_tensor_set(trackable_layer.variables) == to_tensor_set(variables + module_variables)


def test_submodule_trainable_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    trainable_attributes = [v for v in variables + module_variables if v.trainable]
    assert trackable_layer.trainable_variables == trainable_attributes


def test_submodule_non_trainable_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    non_trainable_attributes = [v for v in variables + module_variables if not v.trainable]
    assert trackable_layer.non_trainable_variables == non_trainable_attributes


def test_trainable_weights():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    all_vars = variables + module_variables
    trainable_weights = [v for v in all_vars if v.trainable]
    assert to_tensor_set(trackable_layer.trainable_weights) == to_tensor_set(trainable_weights)


def test_non_trainable_weights():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    all_vars = variables + module_variables
    non_trainable_weights = [v for v in all_vars if not v.trainable]
    assert to_tensor_set(trackable_layer.non_trainable_weights) == to_tensor_set(
        non_trainable_weights
    )


def test_trainable_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    all_vars = variables + module_variables
    trainable_variables = [v for v in all_vars if v.trainable]
    assert to_tensor_set(trackable_layer.trainable_variables) == to_tensor_set(trainable_variables)


def test_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    all_vars = variables + module_variables
    assert to_tensor_set(trackable_layer.variables) == to_tensor_set(all_vars)


@pytest.mark.parametrize(
    "composite_class",
    [CompositeModule, UntrackableCompositeLayer],
)
def test_tensorflow_classes_trackable(composite_class):
    composite_object = composite_class([Matern52()])
    assert len(composite_object.trainable_variables) == 2
