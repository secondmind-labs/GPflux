# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import Callable, List, Tuple

import tensorflow as tf
from tensorflow.python.util.object_identity import ObjectIdentitySet

from gpflow import Parameter
from gpflow.optimizers import NaturalGradient
from gpflux.layers import GPLayer

__all__ = [
    "NatGradModel",
    "NatGradWrapper",
]


class NatGradModel(tf.keras.Model):
    """
    This is a drop-in replacement for tf.keras.Model when constructing GPflux
    models using the functional Keras style, to make it work with the
    NaturalGradient optimizers for q(u) distributions in GP layers.

    model.compile() has to be passed a list of optimizers, which must be one
        (Momentum)NaturalGradient optimizer instance per GPLayer, followed
        by a regular optimizer (e.g. Adam) as the last element for all other
        parameters (hyperparameters, inducing point locations).
    """

    def __init__(
        self, *args, other_loss_fn: Callable[[], tf.Tensor] = None, **kwargs,
    ):
        """
        :param other_loss_fn: experimental feature to allow specifying a
            different loss closure for computing gradients for Adam optimizer.
        """
        super().__init__(*args, **kwargs)
        self.other_loss_fn = other_loss_fn

    @property
    def natgrad_optimizers(self) -> List[NaturalGradient]:
        if not hasattr(self, "_all_optimizers"):
            raise AttributeError(
                "natgrad_optimizers accessed before optimizer being set"
            )  # pragma: no cover
        if self._all_optimizers is None:
            return None  # type: ignore
        return self._all_optimizers[:-1]

    @property
    def optimizer(self) -> tf.optimizers.Optimizer:
        """
        HACK to cope with Keras's callbacks such as ReduceLROnPlateau and
        LearningRateSchedule having been hardcoded for a single optimizer.
        """
        if not hasattr(self, "_all_optimizers"):
            raise AttributeError("optimizer accessed before being set")
        if self._all_optimizers is None:
            return None  # type: ignore
        return self._all_optimizers[-1]

    @optimizer.setter
    def optimizer(self, optimizers):
        if optimizers is None:
            # tf.keras.Model.__init__() sets self.optimizer = None
            self._all_optimizers = None
            return

        if optimizers is self.optimizer:
            # Keras re-sets optimizer with itself; this should not have any effect on the state
            return

        if not isinstance(optimizers, (tuple, list)):
            raise TypeError(
                "optimizer needs to be a list of NaturalGradient optimizers for "
                "each GPLayer followed by one optimizer for non-q(u) parameters, "
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
                "NaturalGradient instances, one for each GPLayer in the model, "
                f"but were {optimizers[:-1]}"
            )  # pragma: no cover
        self._all_optimizers = optimizers

    def _split_natgrad_params_and_other_vars(
        self,
    ) -> Tuple[List[Tuple[Parameter, Parameter]], List[tf.Variable]]:
        # NOTE the structure of variational_params is directly linked to the _natgrad_step,
        # do not change out of sync
        variational_params = [
            (layer.q_mu, layer.q_sqrt)
            for layer in self.layers
            if isinstance(layer, GPLayer)
        ]
        # NOTE could use a natgrad_parameters attribute on a layer or a
        # singledispatch function to make this more flexible for other layers

        # Collect all trainable variables that are not part of variational_params:
        variational_vars_set = ObjectIdentitySet(
            p.unconstrained_variable for vp in variational_params for p in vp
        )
        other_vars = [
            v for v in self.trainable_variables if v not in variational_vars_set
        ]

        return variational_params, other_vars

    def _backwards(self, tape: tf.GradientTape, loss: tf.Tensor) -> None:
        # TODO in TensorFlow 2.2, this method will be replaced by overwriting train_step() instead.
        # The `_backwards` shim still exists independently in TF 2.2.

        print("Calling _backwards")
        # TODO(Ti) This is to check tf.function() compilation works and this
        # won't be called repeatedly. Surprisingly, it gets called *twice*...
        # Leaving the print here until we've established why twice, or whether
        # it's irrelevant, and that it all works correctly in practice with
        # TensorFlow 2.2.

        variational_params, other_vars = self._split_natgrad_params_and_other_vars()

        variational_params_grads, other_grads = tape.gradient(
            loss, (variational_params, other_vars)
        )

        num_natgrad_opt = len(self.natgrad_optimizers)
        num_variational = len(variational_params)
        if len(self.natgrad_optimizers) != len(variational_params):
            raise ValueError(
                f"Model has {num_natgrad_opt} NaturalGradient optimizers, "
                f"but {num_variational} variational distributions"
            )  # pragma: no cover

        for (natgrad_optimizer, (q_mu_grad, q_sqrt_grad), (q_mu, q_sqrt)) in zip(
            self.natgrad_optimizers, variational_params_grads, variational_params
        ):
            natgrad_optimizer._natgrad_apply_gradients(
                q_mu_grad, q_sqrt_grad, q_mu, q_sqrt
            )

        if self.other_loss_fn is not None:
            with tf.GradientTape() as tape:
                other_loss = self.other_loss_fn()
            other_grads = tape.gradient(other_loss, other_vars)

        self.optimizer.apply_gradients(zip(other_grads, other_vars))

    def train_step(self, data):  # pragma: no cover
        """ For TensorFlow 2.2 """
        # TODO(Ti) once moving to TensorFlow 2.2, we need to clean up this
        # method and check it actually works

        from tensorflow.python.keras.engine import data_adapter

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        x, y, sample_weight = data_adapter.expand_1d((x, y, sample_weight))

        with tf.GradientTape() as tape:
            y_pred = self.__call__(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses
            )
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        self._backwards(tape, loss)

        return {m.name: m.result() for m in self.metrics}


class NatGradWrapper(NatGradModel):
    """
    Wraps a class-based Keras model (e.g. gpflux.models.DeepGP) to make it
    work with NaturalGradient optimizers. For more details, see NatGradModel.
    """

    def __init__(self, base_model: tf.keras.Model, *args, **kwargs):
        """
        :param base_model: the class-based Keras model to be wrapped
        """
        super().__init__(*args, **kwargs)
        self.base_model = base_model

    @property
    def layers(self):
        if not hasattr(self, "base_model"):
            # required for super().__init__(), in which base_model has not been set yet
            return super().layers
        else:
            return self.base_model.layers

    def elbo(self, data):
        # DeepGP-specific pass-through
        return self.base_model.elbo(data)

    def call(self, data, training=None):
        # pass-through for model call
        return self.base_model.call(data, training=training)