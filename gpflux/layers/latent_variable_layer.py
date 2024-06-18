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
""" This module implements a latent variable layer for deep GPs. """

import abc
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import default_float
from gpflow.base import TensorType
from gpflow.keras import tf_keras

from gpflux.layers.trackable_layer import TrackableLayer
from gpflux.types import ObservationType


class LayerWithObservations(TrackableLayer, metaclass=abc.ABCMeta):
    """
    By inheriting from this class, Layers indicate that their :meth:`call`
    method takes a second *observations* argument after the customary
    *layer_inputs* argument.

    This is used to distinguish which layers (unlike most standard Keras
    layers) require the original inputs and/or targets during training.
    For example, it is used by the amortized variational inference in the
    :class:`LatentVariableLayer`.
    """

    @abc.abstractmethod
    def call(
        self,
        layer_inputs: TensorType,
        observations: Optional[ObservationType] = None,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """
        The :meth:`call` method of `LayerWithObservations` subclasses should
        accept a second argument, *observations*. In training mode, this will
        be the ``[inputs, targets]`` of the training points; otherwise, it is `None`.
        """


class LatentVariableLayer(LayerWithObservations):
    """
    A latent variable layer, with amortized mean-field variational inference.

    The latent variable is distribution-agnostic, but assumes a variational posterior
    that is fully factorised and is of the same distribution family as the prior.

    This class is used by models as described in :cite:p:`dutordoir2018cde, salimbeni2019iwvi`.
    """

    prior: tfp.distributions.Distribution
    """ The prior distribution for the latent variables. """

    encoder: tf_keras.layers.Layer
    """
    An encoder that maps from a concatenation of inputs and targets to the
    parameters of the approximate posterior distribution of the corresponding
    latent variables.
    """

    compositor: tf_keras.layers.Layer
    """
    A layer that takes as input the two-element ``[layer_inputs, latent_variable_samples]`` list
    and combines the elements into a single output tensor.
    """

    def __init__(
        self,
        prior: tfp.distributions.Distribution,
        encoder: tf_keras.layers.Layer,
        compositor: Optional[tf_keras.layers.Layer] = None,
        name: Optional[str] = None,
    ):
        """
        :param prior: A distribution that represents the :attr:`prior` over the latent variable.
        :param encoder: A layer which is passed the concatenated observation inputs
            and targets, and returns the appropriate parameters for the approximate
            posterior distribution; see :attr:`encoder`.
        :param compositor: A layer that combines layer inputs and latent variable
            samples into a single tensor; see :attr:`compositor`. If you do not specify a value for
            this parameter, the default is
            ``tf.keras.layers.Concatenate(axis=-1, dtype=default_float())``. Note that you should
            set ``dtype`` of the layer to GPflow's default dtype as in
            :meth:`~gpflow.default_float()`.
        :param name: The name of this layer (passed through to `tf.keras.layers.Layer`).
        """

        super().__init__(dtype=default_float(), name=name)
        self.prior = prior
        self.distribution_class = prior.__class__
        self.encoder = encoder
        self.compositor = (
            compositor
            if compositor is not None
            else tf_keras.layers.Concatenate(axis=-1, dtype=default_float())
        )

    def call(
        self,
        layer_inputs: TensorType,
        observations: Optional[ObservationType] = None,
        training: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        r"""
        Sample the latent variables and compose them with the layer input.

        When training, draw a sample of the latent variable from the posterior,
        whose distribution is parameterised by the encoder mapping from the data.
        Also add a KL divergence [posterior∥prior] to the losses.

        When not training, draw a sample of the latent variable from the prior.

        :param layer_inputs: The output of the previous layer.
        :param observations: The ``[inputs, targets]``, with the shapes ``[batch size, Din]``
            and ``[batch size, Dout]`` respectively. This parameter should be passed only when in
            training mode.
        :param training: The training mode indicator.
        :param seed: A random seed for the sampling operation.
        :returns: Samples of the latent variable composed with the layer inputs through the
            :attr:`compositor`
        """
        if training:
            if observations is None:
                raise ValueError("LatentVariableLayer requires observations when training")

            samples, loss_per_datapoint = self._inference_latent_samples_and_loss(
                layer_inputs, observations, seed=seed
            )
        else:
            samples = self._prediction_latent_samples(layer_inputs, seed=seed)
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())

        self.add_loss(loss_per_datapoint)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_local_kl" if self.name else "local_kl"
        self.add_metric(loss_per_datapoint, name=name, aggregation="mean")

        return self.compositor([layer_inputs, samples])

    def _inference_posteriors(
        self,
        observations: ObservationType,
        training: Optional[bool] = None,
    ) -> tfp.distributions.Distribution:
        """
        Return the posterior distributions parametrised by the :attr:`encoder`, which gets called
        with the concatenation of the inputs and targets in the *observations* argument.

        .. todo:: We might want to change encoders to have a
            `tfp.layers.DistributionLambda` final layer that directly returns the
            appropriately parameterised distributions object.

        :param observations: The ``[inputs, targets]``, with the shapes ``[batch size, Din]``
            and ``[batch size, Dout]`` respectively.
        :param training: The training mode indicator (passed through to the :attr:`encoder`'s call).
        :returns: The posterior distributions object.
        """
        inputs, targets = observations
        encoder_inputs = tf.concat(observations, axis=-1)

        distribution_params = self.encoder(encoder_inputs, training=training)
        posteriors = self.distribution_class(*distribution_params, allow_nan_stats=False)

        return posteriors

    def _inference_latent_samples_and_loss(
        self, layer_inputs: TensorType, observations: ObservationType, seed: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Sample latent variables during the *training* forward pass, hence requiring
        the observations. Also return the KL loss per datapoint.

        :param layer_inputs: The output of the previous layer _(unused)_.
        :param observations: The ``[inputs, targets]``, with the shapes ``[batch size, Din]``
            and ``[batch size, Dout]`` respectively.
        :param seed: A random seed for the sampling operation.
        :returns: The samples and the loss-per-datapoint.
        """
        posteriors = self._inference_posteriors(observations, training=True)
        samples = posteriors.sample(seed=seed)  # [N, Dw]
        # closed-form expectation E_q[log(q/p)] = KL[q∥p]:
        local_kls = self._local_kls(posteriors)
        loss_per_datapoint = tf.reduce_mean(local_kls, name="local_kls")
        return samples, loss_per_datapoint

    def _prediction_latent_samples(
        self, layer_inputs: TensorType, seed: Optional[int] = None
    ) -> tf.Tensor:
        """
        Sample latent variables during the *prediction* forward pass, only
        depending on the shape of this layer's inputs.

        :param layer_inputs: The output of the previous layer (for determining batch shape).
        :param seed: A random seed for the sampling operation.
        :returns: The samples.
        """
        batch_shape = tf.shape(layer_inputs)[:-1]
        samples = self.prior.sample(batch_shape, seed=seed)
        return samples

    def _local_kls(self, posteriors: tfp.distributions.Distribution) -> tf.Tensor:
        """
        Compute the KL divergences [posteriors∥prior].

        :param posteriors: A distribution that represents the approximate posteriors.
        :returns: The KL divergences from the prior for each of the posteriors.
        """
        return posteriors.kl_divergence(self.prior)
