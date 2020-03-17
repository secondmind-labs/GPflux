from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Layer

from gpflow.kernels import RBF, Matern12
from gpflow import default_float

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
    ]
    modules_variables = [
        modules[0].variance.unconstrained_variable,
        modules[0].lengthscales.unconstrained_variable,
        modules[1].var_0.variance.unconstrained_variable,
        modules[1].var_0.lengthscales.unconstrained_variable,
        modules[2].var_0,
        modules[2].var_1,
    ]

    attributes = variables + modules
    trackable_layer = TrackableCompositeLayer(attributes=attributes)
    return trackable_layer, variables, modules, modules_variables


def to_tensor_set(tensor_set: List[tf.Tensor]):
    return set([t.experimental_ref() for t in tensor_set])


def test_submodules():
    trackable_layer, variables, modules, _ = setup_layer_modules_variables()
    assert trackable_layer._submodules == modules


def test_submodule_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    assert trackable_layer.submodule_variables() == module_variables


def test_submodule_trainable_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    submodule_trainable_attributes = [v for v in module_variables if v.trainable]
    assert (
        trackable_layer.submodule_trainable_variables()
        == submodule_trainable_attributes
    )


def test_submodule_non_trainable_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    submodule_non_trainable_attributes = [
        v for v in module_variables if not v.trainable
    ]
    assert (
        trackable_layer.submodule_non_trainable_variables()
        == submodule_non_trainable_attributes
    )


def test_trainable_weights():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    all_vars = variables + module_variables
    trainable_weights = [v for v in all_vars if v.trainable]
    assert to_tensor_set(trackable_layer.trainable_weights) == to_tensor_set(
        trainable_weights
    )


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
    assert to_tensor_set(trackable_layer.trainable_variables) == to_tensor_set(
        trainable_variables
    )


def test_variables():
    (
        trackable_layer,
        variables,
        modules,
        module_variables,
    ) = setup_layer_modules_variables()
    all_vars = variables + module_variables
    assert to_tensor_set(trackable_layer.variables) == to_tensor_set(all_vars)


if __name__ == "__main__":
    test_submodules()
    test_submodule_variables()
    test_submodule_trainable_variables()
    test_submodule_non_trainable_variables()
    test_trainable_weights()
