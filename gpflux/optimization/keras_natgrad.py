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
"""
Support for the `gpflow.optimizers.NaturalGradient` optimizer within Keras models.
"""

from typing import Any, List, Mapping, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.python.util.object_identity import ObjectIdentitySet

import gpflow
from gpflow import Parameter
from gpflow.keras import tf_keras
from gpflow.models.model import MeanAndVariance
from gpflow.optimizers import NaturalGradient

from gpflux.layers.gp_layer import GPLayer

__all__ = [
    "NatGradModel",
    "NatGradWrapper",
]


class NatGradModel(tf_keras.Model):
    r"""
    This is a drop-in replacement for `tf.keras.Model` when constructing GPflux
    models using the functional Keras style, to make it work with the
    NaturalGradient optimizers for q(u) distributions in GP layers.

    You must set the `natgrad_layers` property before compiling the model. Set it
    to the list of all :class:`~gpflux.layers.GPLayer`\ s you want to train using
    natural gradients. You can also set it to `True` to include all of them.

    This model's :meth:`compile` method has to be passed a list of optimizers, which
    must be one `gpflow.optimizers.NaturalGradient` instance per natgrad-trained
    :class:`~gpflux.layers.GPLayer`, followed by a regular optimizer (e.g.
    `tf.keras.optimizers.Adam`) as the last element to handle all other
    parameters (hyperparameters, inducing point locations).
    """

    @property
    def natgrad_layers(self) -> List[GPLayer]:
        """
        The list of layers in this model that should be optimized using
        `~gpflow.optimizers.NaturalGradient`.

        :getter: Returns a list of the layers that should be trained using
            `~gpflow.optimizers.NaturalGradient`.
        :setter: Sets the layers that should be trained using
            `~gpflow.optimizers.NaturalGradient`. Can be an explicit list or a `bool`:
            If set to `True`, it will select all `GPLayer` instances in the model layers.
        """
        if not hasattr(self, "_natgrad_layers"):
            raise AttributeError(
                f"natgrad_layers must be set before training {self.__class__.__name__}"
            )  # pragma: no cover
        return self._natgrad_layers

    @natgrad_layers.setter
    def natgrad_layers(self, layers: Union[List[GPLayer], bool]) -> None:
        if isinstance(layers, bool):
            if layers:  # all (GP) layers
                self._natgrad_layers = [
                    layer for layer in self.layers if isinstance(layer, GPLayer)
                ]
            else:  # no layers
                self._natgrad_layers = []
        else:
            self._natgrad_layers = layers

    @property
    def natgrad_optimizers(self) -> List[gpflow.optimizers.NaturalGradient]:
        if not hasattr(self, "_all_optimizers"):
            raise AttributeError(
                "natgrad_optimizers accessed before optimizer being set"
            )  # pragma: no cover
        if self._all_optimizers is None:
            return None  # type: ignore
        return self._all_optimizers[:-1]

    @property
    def optimizer(self) -> tf_keras.optimizers.Optimizer:
        """
        HACK to cope with Keras's callbacks such as
        :class:`~tf.keras.callbacks.ReduceLROnPlateau`
        and
        :class:`~tf.keras.callbacks.LearningRateScheduler`
        having been hardcoded for a single optimizer.
        """
        if not hasattr(self, "_all_optimizers"):
            raise AttributeError("optimizer accessed before being set")
        if self._all_optimizers is None:
            return None  # type: ignore
        return self._all_optimizers[-1]

    @optimizer.setter
    def optimizer(
        self, optimizers: List[Union[NaturalGradient, tf_keras.optimizers.Optimizer]]
    ) -> None:
        if optimizers is None:
            # tf.keras.Model.__init__() sets self.optimizer = None
            self._all_optimizers = None
            return

        if optimizers is self.optimizer:
            # Keras re-sets optimizer with itself; this should not have any effect on the state
            return

        if not isinstance(optimizers, (tuple, list)):
            raise TypeError(
                "`optimizer` needs to be a list of NaturalGradient optimizers for "
                "each element of `natgrad_layers` followed by one optimizer for all "
                "other parameters, "
                f"but was {optimizers}"
            )  # pragma: no cover
        if isinstance(optimizers[-1], NaturalGradient):
            raise TypeError(
                "The last element of the optimizer list applies to the non-variational "
                "parameters and cannot be a NaturalGradient optimizer, "
                f"but was {optimizers[-1]}"
            )  # pragma: no cover
        if not all(isinstance(o, NaturalGradient) for o in optimizers[:-1]):
            raise TypeError(
                "The all-but-last elements of the optimizer list must be "
                "NaturalGradient instances, one for each element of natgrad_layers, "
                f"but were {optimizers[:-1]}"
            )  # pragma: no cover
        self._all_optimizers = optimizers

    def _split_natgrad_params_and_other_vars(
        self,
    ) -> Tuple[List[Tuple[Parameter, Parameter]], List[tf.Variable]]:
        # NOTE the structure of variational_params is directly linked to the _natgrad_step,
        # do not change out of sync
        variational_params = [(layer.q_mu, layer.q_sqrt) for layer in self.natgrad_layers]
        # NOTE could use a natgrad_parameters attribute on a layer or a
        # singledispatch function to make this more flexible for other layers

        # Collect all trainable variables that are not part of variational_params:
        variational_vars_set = ObjectIdentitySet(
            p.unconstrained_variable for vp in variational_params for p in vp
        )
        other_vars = [v for v in self.trainable_variables if v not in variational_vars_set]

        return variational_params, other_vars

    def _apply_backwards_pass(self, loss: tf.Tensor, tape: tf.GradientTape) -> None:
        print("Executing NatGradModel backwards pass")
        # TODO(Ti) This is to check tf.function() compilation works and this
        # won't be called repeatedly. Surprisingly, it gets called *twice*...
        # Leaving the print here until we've established why twice, or whether
        # it's irrelevant, and that it all works correctly in practice.

        num_natgrad_layers = len(self.natgrad_layers)
        num_natgrad_opt = len(self.natgrad_optimizers)
        if num_natgrad_opt != num_natgrad_layers:
            raise ValueError(
                f"Model has {num_natgrad_opt} NaturalGradient optimizers, "
                f"but {num_natgrad_layers} variational distributions in "
                "natgrad_layers"
            )  # pragma: no cover

        variational_params, other_vars = self._split_natgrad_params_and_other_vars()
        variational_params_vars = [
            (q_mu.unconstrained_variable, q_sqrt.unconstrained_variable)
            for (q_mu, q_sqrt) in variational_params
        ]

        variational_params_grads, other_grads = tape.gradient(
            loss, (variational_params_vars, other_vars)
        )

        for (natgrad_optimizer, (q_mu_grad, q_sqrt_grad), (q_mu, q_sqrt)) in zip(
            self.natgrad_optimizers, variational_params_grads, variational_params
        ):
            natgrad_optimizer._natgrad_apply_gradients(q_mu_grad, q_sqrt_grad, q_mu, q_sqrt)

        self.optimizer.apply_gradients(zip(other_grads, other_vars))

    def train_step(self, data: Any) -> Mapping[str, Any]:
        """
        The logic for one training step. For more details of the
        implementation, see TensorFlow's documentation of how to
        `customize what happens in Model.fit
        <https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit>`_.
        """
        from tensorflow.python.keras.engine import data_adapter

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self.__call__(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self._apply_backwards_pass(loss, tape=tape)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


class NatGradWrapper(NatGradModel):
    """
    Wraps a class-based Keras model (e.g. the return value of
    `gpflux.models.DeepGP.as_training_model`) to make it work with
    `gpflow.optimizers.NaturalGradient` optimizers. For more details, see
    `NatGradModel`.

    (Note that you can also directly pass `NatGradModel` to the
    :class:`~gpflux.models.DeepGP`'s
    :attr:`~gpflux.models.DeepGP.default_model_class` or
    :meth:`~gpflux.models.DeepGP.as_training_model`'s *model_class* arguments.)

    .. todo::

        This class will probably be removed in the future.
    """

    def __init__(self, base_model: tf_keras.Model, *args: Any, **kwargs: Any):
        """
        :param base_model: the class-based Keras model to be wrapped
        """
        super().__init__(*args, **kwargs)
        self.base_model = base_model

    @property
    def layers(self) -> List[tf_keras.layers.Layer]:
        if not hasattr(self, "base_model"):
            # required for super().__init__(), in which base_model has not been set yet
            return super().layers
        else:
            return self.base_model.layers

    def call(self, data: Any, training: Optional[bool] = None) -> Union[tf.Tensor, MeanAndVariance]:
        """
        Calls the model on new inputs. Simply passes through to the ``base_model``.
        """
        return self.base_model.call(data, training=training)
