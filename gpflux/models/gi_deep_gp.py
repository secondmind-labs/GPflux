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
""" This module provides an implementation of global inducing points for deep GPs. """

import itertools
from typing import List, Optional, Tuple, Type, Union

import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow import Parameter, default_float
from gpflow.base import Module, TensorType

import gpflux
from gpflux.layers import LayerWithObservations, LikelihoodLayer, SampleBasedGaussianLikelihoodLayer
from gpflux.sampling.sample import Sample


class GIDeepGP(Module):

    f_layers: List[tf.keras.layers.Layer]
    """ A list of all layers in this DeepGP (just :attr:`likelihood_layer` is separate). """

    likelihood_layer: gpflux.layers.LikelihoodLayer
    """ The likelihood layer. """

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
        f_layers: List[tf.keras.layers.Layer],
        num_inducing: int,
        *,
        likelihood_var: Optional[float] = 1.0,
        inducing_init: Optional[tf.Tensor] = None,
        inducing_shape: Optional[List[int]] = None,
        input_dim: Optional[int] = None,
        target_dim: Optional[int] = None,
        default_model_class: Type[tf.keras.Model] = tf.keras.Model,
        num_data: Optional[int] = None,
        num_train_samples: Optional[int] = 10,
        num_test_samples: Optional[int] = 100,
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
        self.inputs = tf.keras.Input((input_dim,), name="inputs")
        self.targets = tf.keras.Input((target_dim,), name="targets")
        self.f_layers = f_layers
        self.likelihood_layer = SampleBasedGaussianLikelihoodLayer(variance=likelihood_var)
        self.num_inducing = num_inducing
        self.default_model_class = default_model_class
        self.num_data = self._validate_num_data(f_layers, num_data)

        assert (inducing_init is not None) != (inducing_shape is not None)

        if inducing_init is None:
            self.inducing_data = Parameter(inducing_shape, dtype=default_float())
        else:
            self.inducing_data = Parameter(inducing_init, dtype=default_float())

        assert num_inducing == tf.shape(self.inducing_data)[0]
        self.rank = 1 + len(tf.shape(self.inducing_data))

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

    @staticmethod
    def _validate_num_data(
        f_layers: List[tf.keras.layers.Layer], num_data: Optional[int] = None
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

    def _inducing_add(self, inputs: TensorType):

        inducing_data = tf.tile(
            tf.expand_dims(self.inducing_data, 0),
            [tf.shape(inputs)[0], *[1]*(self.rank-1)]
        )
        x = tf.concat([inducing_data, inputs], 1)

        return x

    def _inducing_remove(self, inputs: TensorType):
        return inputs[:, self.num_inducing:]

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
            num_samples = self.num_train_samples
        else:
            # TODO would it be better to simply pass [inputs, None] in this case?
            observations = None
            num_samples = self.num_test_samples

        features = tf.tile(tf.expand_dims(features, 0),
                           [num_samples, *[1]*(self.rank-1)])
        features = self._inducing_add(features)

        for layer in self.f_layers:
            if isinstance(layer, LayerWithObservations):
                raise NotImplementedError("Latent variable layers not yet supported")
            else:
                features = layer(features, training=training)
        return self._inducing_remove(features)

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
    ) -> tfp.distributions.Distribution:
        f_outputs = self._evaluate_deep_gp(inputs, targets=targets, training=training)
        y_outputs = self._evaluate_likelihood(f_outputs, targets=targets, training=training)
        return y_outputs

    def predict_f(self, inputs: TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :returns: The mean and variance (not the scale!) of ``f``, for compatibility with GPflow
           models.

        .. note:: This method does **not** support ``full_cov`` or ``full_output_cov``.
        """
        raise NotImplementedError("TODO")

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

    def _get_model_class(self, model_class: Optional[Type[tf.keras.Model]]) -> Type[tf.keras.Model]:
        if model_class is not None:
            return model_class
        else:
            return self.default_model_class

    def as_training_model(
        self, model_class: Optional[Type[tf.keras.Model]] = None
    ) -> tf.keras.Model:
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
        self, model_class: Optional[Type[tf.keras.Model]] = None
    ) -> tf.keras.Model:
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

    def sample(self, inputs: TensorType, num_samples: int, consistent: bool = False) -> tf.Tensor:
        features = tf.tile(tf.expand_dims(inputs, 0),
                           [num_samples, *[1]*(self.rank-1)])
        features = self._inducing_add(features)

        for layer in self.f_layers:
            if isinstance(layer, LayerWithObservations):
                raise NotImplementedError("Latent variable layers not yet supported")
            else:
                features = layer(features, training=None, full_cov=consistent)

        return self._inducing_remove(features)


def sample_dgp(model: GIDeepGP) -> Sample:  # TODO: should this be part of a [Vanilla]DeepGP class?
    function_draws = [layer.sample() for layer in model.f_layers]
    # TODO: error check that all layers implement .sample()?

    class ChainedSample(Sample):
        """ This class chains samples from consecutive layers. """

        def __call__(self, X: TensorType) -> tf.Tensor:
            for f in function_draws:
                X = f(X)
            return X

    return ChainedSample()
