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
This module provides an implementation of global inducing points for deep GPs. See Ober &
Aitchison (2021) https://arxiv.org/abs/2005.08140

Note: this is a sample-based implementation, so we return samples from the DGP instead of predictive
means and variances.
"""

import itertools
from typing import List, Optional, Tuple, Type

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import Parameter, default_float
from gpflow.base import Module, TensorType

import gpflux
from gpflux.layers import LayerWithObservations, SampleBasedGaussianLikelihoodLayer


class GIDeepGP(Module):
    """
    This class combines a sequential function model ``f(x) = fₙ(⋯ (f₂(f₁(x))))``
    and a likelihood ``p(y|f)``. The model uses the inference method described in Ober & Aitchison
    (2021).

    Layers currently do NOT support inheriting from :class:`~gpflux.layers.LayerWithObservations`.

    .. note:: This class is **not** a `tf.keras.Model` subclass itself. To access
       Keras features, call either :meth:`as_training_model` or :meth:`as_prediction_model`
       (depending on the use-case) to create a `tf.keras.Model` instance. See the method docstrings
       for more details.
    """

    inputs: tf.keras.Input
    targets: tf.keras.Input

    f_layers: List[tf.keras.layers.Layer]
    """ A list of all layers in this DeepGP (just :attr:`likelihood_layer` is separate). """

    likelihood_layer: gpflux.layers.SampleBasedGaussianLikelihoodLayer
    """ The likelihood layer. Currently restricted to 
    :class:`gpflux.layers.SampleBasedGaussianLikelihoodLayer`."""

    default_model_class: Type[tf.keras.Model]
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

    num_train_samples: int
    """
    The number of samples from the posterior used for training. Usually lower than the number used
    for testing. Cannot be changed during training once the model has been compiled.
    """

    num_test_samples: int
    """
    The number of samples from the posterior used for testing. Usually greater than the number used
    for training.
    """

    num_inducing: int
    """
    The number of inducing points to be used. This is the same throughout the model.
    """

    inducing_data: Parameter
    """
    The inducing inputs for the model, propagated through the layers.
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
        :param num_inducing: The number of inducing points used in the approximate posterior. This
            must be the same for all layers of the model. If you do not specify a value for this
            parameter explicitly, it is automatically detected from the
            :attr:`~gpflux.layers.GIGPLayer.num_inducing` attribute in the GP layers.
        :param likelihood_var: The variance for the Gaussian likelihood ``p(y|f)``. This will be
            passed to a :class:`gpflux.layers.SampleBasedGaussianLikelihoodLayer` instance, which
            will be the likelihood.
        :param inducing_init: An initialization for the inducing inputs. Must have shape
            [num_inducing, input_dim], and cannot be provided along with `inducing_shape`.
        :param inducing_shape: The initialization shape of the inducing inputs. Used to initialize
            the inputs from N(0, 1) if `inducing_init` is not provided. Must be [num_inducing,
            input_dim]. Cannot be provided along with `inducing_init`.
        :param input_dim: The input dimensionality.
        :param target_dim: The target dimensionality.
        :param default_model_class: The default for the *model_class* argument of
            :meth:`as_training_model` and :meth:`as_prediction_model`;
            see the :attr:`default_model_class` attribute.
        :param num_data: The number of points in the training dataset; see the
            :attr:`num_data` attribute.
            If you do not specify a value for this parameter explicitly, it is automatically
            detected from the :attr:`~gpflux.layers.GIGPLayer.num_data` attribute in the GP layers.
        :param num_train_samples: The number of samples from the posterior used for training.
        :param num_test_samples: The number of samples from the posterior used for testing.
        """
        self.inputs = tf.keras.Input((input_dim,), name="inputs")
        self.targets = tf.keras.Input((target_dim,), name="targets")
        self.f_layers = f_layers
        self.likelihood_layer = SampleBasedGaussianLikelihoodLayer(variance=likelihood_var)
        self.num_inducing = self._validate_num_inducing(f_layers, num_inducing)
        self.default_model_class = default_model_class
        self.num_data = self._validate_num_data(f_layers, num_data)

        if (inducing_init is not None) != (inducing_shape is not None):
            raise ValueError(f"One of `inducing_init` or `inducing_shape` must be exclusively"
                             f"provided.")

        if inducing_init is None:
            self.inducing_data = Parameter(tf.random.normal(inducing_shape), dtype=default_float())
        else:
            self.inducing_data = Parameter(inducing_init, dtype=default_float())

        if num_inducing != tf.shape(self.inducing_data)[0]:
            raise ValueError(f"The number of inducing inputs {self.inducing_data.shape[0]} must "
                             f"equal num_inducing {num_inducing}.")
        if input_dim is not None and input_dim != tf.shape(self.inducing_data)[-1]:
            raise ValueError(f"The dimension of the inducing inputs {self.inducing_data.shape[-1]}"
                             f"must equal input_dim {input_dim}")

        self.rank = 1 + len(tf.shape(self.inducing_data))

        if self.rank != 3:
            raise ValueError(f"Currently the model only supports data of rank 2 ([N, D]); received"
                             f"rank {len(self.inducing_data.shape)} instead.")

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

    @staticmethod
    def _validate_num_inducing(
        f_layers: List[tf.keras.layers.Layer], num_inducing: Optional[int] = None
    ) -> int:
        """
        Check that the :attr:`~gpflux.layers.gp_layer.GPLayer.num_inducing`
        attributes of all layers in *f_layers* are consistent with each other
        and with the (optional) *num_inducing* argument.

        :returns: The validated number of inducing points.
        """
        for i, layer in enumerate(f_layers):
            layer_num_inducing = getattr(layer, "num_inducing", None)
            if num_inducing is None:
                num_inducing = layer_num_inducing
            else:
                if layer_num_inducing is not None and num_inducing != layer_num_inducing:
                    raise ValueError(
                        f"f_layers[{i}].num_inducing is inconsistent with num_inducing="
                        f"{num_inducing}"
                    )
        if num_inducing is None:
            raise ValueError("Could not determine num_inducing; please provide explicitly")
        return num_inducing

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
        """
        Adds the inducing points to the data to propagate through the model.

        :param inputs: input data
        :return: concatenated inducing points and datapoints
        """

        inducing_data = tf.tile(
            tf.expand_dims(self.inducing_data, 0),
            [tf.shape(inputs)[0], *[1]*(self.rank-1)]
        )
        x = tf.concat([inducing_data, inputs], 1)

        return x

    def _inducing_remove(self, inputs: TensorType):
        """
        Removes the inducing points from the combined inducing points and data tensor

        :param inputs: combined inducing point and data tensor
        :return: data only
        """
        return inputs[:, self.num_inducing:]

    def _evaluate_deep_gp(
        self,
        inputs: TensorType,
        targets: Optional[TensorType],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """
        Evaluate ``f(x) = fₙ(⋯ (f₂(f₁(x))))`` on the *inputs* argument. We must start by expanding
        the data by copying it :attr:`self.num_train_samples` times, then adding inducing points.
        These are then removed after the inducing points and data have been propagated through the
        model.

        Layers that inherit from :class:`~gpflux.layers.LayerWithObservations`
        are not yet supported.
        """
        features = inputs

        # NOTE: we cannot rely on the `training` flag here, as the correct
        # symbolic graph needs to be constructed at "build" time (before either
        # fit() or predict() get called).
        if targets is not None:
            num_samples = self.num_train_samples
        else:
            # TODO would it be better to simply pass [inputs, None] in this case?
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
        """
        Sample `num_samples` from the posterior at `inputs`. If `consistent` is True, we return
        consistent function samples.

        :param inputs: The input data at which we wish to obtain samples. Must have shape [N, D].
        :param num_samples: The number of samples to obtain.
        :param consistent: Whether to sample consistent samples.
        :return: samples with shape [S, N, L].
        """
        if inputs.shape[-1] != self.inducing_data.shape[-1]:
            raise ValueError(f"The trailing dimension of `inputs` must match the trailing dimension"
                             f"the model's inducing data: received {inputs.shape[-1]} but expected"
                             f"{self.inducing_data.shape[-1]}.")
        if len(inputs.shape) != 2:
            raise ValueError(f"Currently only inputs of rank 2 are supported; received rank"
                             f"{len(inputs.shape)}")

        features = tf.tile(tf.expand_dims(inputs, 0),
                           [num_samples, *[1]*(self.rank-1)])
        features = self._inducing_add(features)

        for layer in self.f_layers:
            if isinstance(layer, LayerWithObservations):
                raise NotImplementedError("Latent variable layers not yet supported")
            else:
                features = layer(features, training=None, full_cov=consistent)

        return self._inducing_remove(features)
