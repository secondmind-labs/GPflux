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
""" This module provides the base implementation for DeepGP models. """

import itertools
from typing import List, Optional, Tuple, Type, Union

import tensorflow as tf

import gpflow
from gpflow.base import Module, TensorType
from gpflow.keras import tf_keras

import gpflux
from gpflux.layers import LayerWithObservations, LikelihoodLayer
from gpflux.sampling.sample import Sample


class DeepGP(Module):
    """
    This class combines a sequential function model ``f(x) = fₙ(⋯ (f₂(f₁(x))))``
    and a likelihood ``p(y|f)``.

    Layers might depend on both inputs x and targets y during training by
    inheriting from :class:`~gpflux.layers.LayerWithObservations`; those will
    be passed the argument ``observations=[inputs, targets]``.

    When data is used with methods in this class (e.g. :meth:`predict_f` method), it needs to
    be with ``dtype`` corresponding to GPflow's default dtype as in :meth:`~gpflow.default_float()`.

    .. note:: This class is **not** a `tf.keras.Model` subclass itself. To access
       Keras features, call either :meth:`as_training_model` or :meth:`as_prediction_model`
       (depending on the use-case) to create a `tf.keras.Model` instance. See the method docstrings
       for more details.
    """

    inputs: tf_keras.Input
    targets: tf_keras.Input

    f_layers: List[tf_keras.layers.Layer]
    """ A list of all layers in this DeepGP (just :attr:`likelihood_layer` is separate). """

    likelihood_layer: gpflux.layers.LikelihoodLayer
    """ The likelihood layer. """

    default_model_class: Type[tf_keras.Model]
    """
    The default for the *model_class* argument of :meth:`as_training_model` and
    :meth:`as_prediction_model`. This must have the same semantics as `tf.keras.Model`,
    that is, it must accept a list of inputs and an output. This could be
    `tf.keras.Model` itself or `gpflux.optimization.NatGradModel` (but not, for
    example, `tf.keras.Sequential`).
    """

    num_data: int
    """
    The number of points in the training dataset. This information is used to
    obtain correct scaling between the data-fit and the KL term in the evidence
    lower bound (:meth:`elbo`).
    """

    def __init__(
        self,
        f_layers: List[tf_keras.layers.Layer],
        likelihood: Union[
            gpflux.layers.LikelihoodLayer, gpflow.likelihoods.Likelihood
        ],  # fully-qualified for autoapi
        *,
        input_dim: Optional[int] = None,
        target_dim: Optional[int] = None,
        default_model_class: Type[tf_keras.Model] = tf_keras.Model,
        num_data: Optional[int] = None,
    ):
        """
        :param f_layers: The layers ``[f₁, f₂, …, fₙ]`` describing the latent
            function ``f(x) = fₙ(⋯ (f₂(f₁(x))))``.
        :param likelihood: The layer for the likelihood ``p(y|f)``. If this is a
            GPflow likelihood, it will be wrapped in a :class:`~gpflux.layers.LikelihoodLayer`.
            Alternatively, you can provide a :class:`~gpflux.layers.LikelihoodLayer` explicitly.
        :param input_dim: The input dimensionality.
        :param target_dim: The target dimensionality.
        :param default_model_class: The default for the *model_class* argument of
            :meth:`as_training_model` and :meth:`as_prediction_model`;
            see the :attr:`default_model_class` attribute.
        :param num_data: The number of points in the training dataset; see the
            :attr:`num_data` attribute.
            If you do not specify a value for this parameter explicitly, it is automatically
            detected from the :attr:`~gpflux.layers.GPLayer.num_data` attribute in the GP layers.
        """
        self.inputs = tf_keras.Input((input_dim,), dtype=gpflow.default_float(), name="inputs")
        self.targets = tf_keras.Input((target_dim,), dtype=gpflow.default_float(), name="targets")
        self.f_layers = f_layers
        if isinstance(likelihood, gpflow.likelihoods.Likelihood):
            self.likelihood_layer = LikelihoodLayer(likelihood)
        else:
            self.likelihood_layer = likelihood
        self.default_model_class = default_model_class
        self.num_data = self._validate_num_data(f_layers, num_data)

    @staticmethod
    def _validate_num_data(
        f_layers: List[tf_keras.layers.Layer], num_data: Optional[int] = None
    ) -> int:
        """
        Check that the :attr:`~gpflux.layers.gp_layer.GPLayer.num_data`
        attributes of all layers in *f_layers* are consistent with each other
        and with the (optional) *num_data* argument.

        :returns: The validated number of datapoints.
        """
        for i, layer in enumerate(f_layers):
            layer_num_data = getattr(layer, "num_data", None)
            if num_data is None:
                num_data = layer_num_data
            else:
                if layer_num_data is not None and num_data != layer_num_data:
                    raise ValueError(
                        f"f_layers[{i}].num_data is inconsistent with num_data={num_data}"
                    )
        if num_data is None:
            raise ValueError("Could not determine num_data; please provide explicitly")
        return num_data

    @staticmethod
    def _validate_dtype(x: TensorType) -> None:
        """
        Check that data ``x`` is of correct ``dtype``, corresponding to GPflow's default dtype as
        defined by :meth:`~gpflow.default_float()`.

        :raise ValueError: If ``x`` is of incorrect ``dtype``.
        """
        if x.dtype != gpflow.default_float():
            raise ValueError(
                f"x needs to have dtype {gpflow.default_float()} (according to "
                f"gpflow.default_float()), however got x with {x.dtype} dtype."
            )

    def _evaluate_deep_gp(
        self,
        inputs: TensorType,
        targets: Optional[TensorType],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """
        Evaluate ``f(x) = fₙ(⋯ (f₂(f₁(x))))`` on the *inputs* argument.

        Layers that inherit from :class:`~gpflux.layers.LayerWithObservations`
        are passed the additional keyword argument ``observations=[inputs,
        targets]`` if *targets* contains a value, or ``observations=None`` when
        *targets* is `None`.
        """
        features = inputs

        # NOTE: we cannot rely on the `training` flag here, as the correct
        # symbolic graph needs to be constructed at "build" time (before either
        # fit() or predict() get called).
        if targets is not None:
            observations = [inputs, targets]
        else:
            # TODO would it be better to simply pass [inputs, None] in this case?
            observations = None

        for layer in self.f_layers:
            if isinstance(layer, LayerWithObservations):
                features = layer(features, observations=observations, training=training)
            else:
                features = layer(features, training=training)
        return features

    def _evaluate_likelihood(
        self,
        f_outputs: TensorType,
        targets: Optional[TensorType],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """
        Call the `likelihood_layer` on *f_outputs*, which adds the
        corresponding layer loss when training.
        """
        return self.likelihood_layer(f_outputs, targets=targets, training=training)

    def call(
        self,
        inputs: TensorType,
        targets: Optional[TensorType] = None,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        self._validate_dtype(inputs)
        if targets is not None:
            self._validate_dtype(targets)
        f_outputs = self._evaluate_deep_gp(inputs, targets=targets, training=training)
        y_outputs = self._evaluate_likelihood(f_outputs, targets=targets, training=training)
        return y_outputs

    def predict_f(self, inputs: TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :returns: The mean and variance (not the scale!) of ``f``, for compatibility with GPflow
           models.
        :raise ValueError: If ``x`` is of incorrect ``dtype``.

        .. note:: This method does **not** support ``full_cov`` or ``full_output_cov``.
        """
        self._validate_dtype(inputs)
        f_distribution = self._evaluate_deep_gp(inputs, targets=None)
        return f_distribution.loc, f_distribution.scale.diag ** 2

    def elbo(self, data: Tuple[TensorType, TensorType]) -> tf.Tensor:
        """
        :returns: The ELBO (not the per-datapoint loss!), for compatibility with GPflow models.
        """
        X, Y = data
        _ = self.call(X, Y, training=True)
        all_losses = [
            loss
            for layer in itertools.chain(self.f_layers, [self.likelihood_layer])
            for loss in layer.losses
        ]
        return -tf.reduce_sum(all_losses) * self.num_data

    def _get_model_class(self, model_class: Optional[Type[tf_keras.Model]]) -> Type[tf_keras.Model]:
        if model_class is not None:
            return model_class
        else:
            return self.default_model_class

    def as_training_model(
        self, model_class: Optional[Type[tf_keras.Model]] = None
    ) -> tf_keras.Model:
        r"""
        Construct a `tf.keras.Model` instance that requires you to provide both ``inputs``
        and ``targets`` to its call. This information is required for
        training the model, because the ``targets`` need to be passed to the `likelihood_layer` (and
        to :class:`~gpflux.layers.LayerWithObservations` instances such as
        :class:`~gpflux.layers.LatentVariableLayer`\ s, if present).

        When compiling the returned model, do **not** provide any additional
        losses (this is handled by the :attr:`likelihood_layer`).

        Train with

        .. code-block:: python

            model.compile(optimizer)  # do NOT pass a loss here
            model.fit({"inputs": X, "targets": Y}, ...)

        See `Keras's Endpoint layer pattern
        <https://keras.io/examples/keras_recipes/endpoint_layer_pattern/>`_
        for more details.

        .. note:: Use `as_prediction_model` if you want only to predict, and do not want to pass in
           a dummy array for the targets.

        :param model_class: The model class to use; overrides `default_model_class`.
        """
        model_class = self._get_model_class(model_class)
        outputs = self.call(self.inputs, self.targets)
        return model_class([self.inputs, self.targets], outputs)

    def as_prediction_model(
        self, model_class: Optional[Type[tf_keras.Model]] = None
    ) -> tf_keras.Model:
        """
        Construct a `tf.keras.Model` instance that requires only ``inputs``,
        which means you do not have to provide dummy target values when
        predicting at test points.

        Predict with

        .. code-block:: python

            model.predict(Xtest, ...)

        .. note:: The returned model will not support training; for that, use `as_training_model`.

        :param model_class: The model class to use; overrides `default_model_class`.
        """
        model_class = self._get_model_class(model_class)
        outputs = self.call(self.inputs)
        return model_class(self.inputs, outputs)


def sample_dgp(model: DeepGP) -> Sample:  # TODO: should this be part of a [Vanilla]DeepGP class?
    function_draws = [layer.sample() for layer in model.f_layers]
    # TODO: error check that all layers implement .sample()?

    class ChainedSample(Sample):
        """This class chains samples from consecutive layers."""

        def __call__(self, X: TensorType) -> tf.Tensor:
            for f in function_draws:
                X = f(X)
            return X

    return ChainedSample()
