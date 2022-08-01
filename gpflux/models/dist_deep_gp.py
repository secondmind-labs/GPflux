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
from sklearn.pipeline import FeatureUnion

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import Module, TensorType
from gpflux.layers import GPLayer, LikelihoodLayer
from gpflow.likelihoods import Likelihood
from gpflux.layers import DistGPLayer
#from gpflux.sampling.sample import Sample

class DistDeepGP(Module):
    
    """
    This class combines a sequential function model ``f(x) = fₙ(⋯ (f₂(f₁(x))))``
    and a likelihood ``p(y|f)``
    
    Important note: works for both DeepGPs and DistDeepGPs
    """

    inputs: tf.keras.Input
    targets: tf.keras.Input
    f_layers: List[tf.keras.layers.Layer]
    likelihood_layer: LikelihoodLayer
    default_model_class: Type[tf.keras.Model]
    num_data: int

    def __init__(
        self,
        f_layers: List[tf.keras.layers.Layer],
        likelihood: Union[LikelihoodLayer, Likelihood],  # fully-qualified for autoapi
        *,
        num_data: Optional[int],
        input_dim: Optional[int] = None,
        target_dim: Optional[int] = None,
        default_model_class: Type[tf.keras.Model] = tf.keras.Model,
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
        self.likelihood_layer = likelihood
        self.default_model_class = default_model_class
        self.num_data = num_data

    def __repr__(self):
        return f'DistDGP(layers:"{len(self.f_layers)}",units:"{self.f_layers[0].num_latent_gps}",lik.:{self.likelihood_layer})'

    def _evaluate_dist_deep_gp(
        self,
        inputs: TensorType,
        targets: Optional[TensorType],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """
        Evaluate ``f(x) = fₙ(⋯ (f₂(f₁(x))))`` on the *inputs* argument.
        """
        features = inputs

        for count, layer in enumerate(self.f_layers):            
            features = layer(features, training=training)        
        return features


    def _evaluate_layer_wise_deep_gp(
        self,
        inputs: TensorType,
        *,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        """
        Evaluate ``f(x) = fₙ(⋯ (f₂(f₁(x))))`` on the *inputs* argument.
        """
        features = inputs
        hidden_layers = []

        for count, layer in enumerate(self.f_layers):
            
            features = layer(features, training=training)            
            moments = features.mean(), features.variance()
            hidden_layers.append(moments)

        return hidden_layers

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
        f_outputs = self._evaluate_dist_deep_gp(inputs, targets=targets, training=training)
        y_outputs = self._evaluate_likelihood(f_outputs, targets=targets, training=training)
        return y_outputs

    def predict_f(self, inputs: TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :returns: The mean and variance (not the scale!) of ``f``, for compatibility with GPflow
           models.

        .. note:: This method does **not** support ``full_cov`` or ``full_output_cov``.
        """
        f_distribution = self._evaluate_dist_deep_gp(inputs, targets=None)
        return f_distribution.loc, f_distribution.scale.diag ** 2

    def elbo(self, data: Tuple[TensorType, TensorType],
        *,
        training: Optional[bool] = True) -> tf.Tensor:
        """
        :returns: The ELBO (not the per-datapoint loss!), for compatibility with GPflow models.
        """
        X, Y = data
        _ = self.call(X, Y, training=training)
        all_losses = [
            loss
            for layer in itertools.chain(self.f_layers, [self.likelihood_layer])
            for loss in layer.losses
        ]
        return -tf.reduce_sum(all_losses) * self.num_data

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
        model_class = tf.keras.Model
        outputs = self.call(self.inputs, self.targets, True)
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
        model_class = tf.keras.Model
        outputs = self.call(self.inputs, False)
        return model_class(self.inputs, outputs)

"""
def sample_dgp(model: DeepGP) -> Sample:  # TODO: should this be part of a [Vanilla]DeepGP class?
    function_draws = [layer.sample() for layer in model.f_layers]
    # TODO: error check that all layers implement .sample()?

    class ChainedSample(Sample):
        #This class chains samples from consecutive layers.

        def __call__(self, X: TensorType) -> tf.Tensor:
            for f in function_draws:
                X = f(X)
            return X

    return ChainedSample()
"""