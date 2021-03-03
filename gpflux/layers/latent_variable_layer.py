# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""Latent variable layer for deep GPs"""

import abc
from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import default_float
from gpflow.base import TensorType

from gpflux.layers.trackable_layer import TrackableLayer
from gpflux.types import ShapeType

ObservationType = List[TensorType]


class LayerWithObservations(TrackableLayer, metaclass=abc.ABCMeta):
    """
    Layers that inherit from this class should accept `observations` as a second
    argument in the `call` method after the default inputs argument for each layer.
    """

    @abc.abstractmethod
    def call(
        self,
        layer_inputs: TensorType,
        observations: Optional[ObservationType] = None,
        training: Optional[bool] = None,
    ) -> TensorType:
        """
        Subclasses of LayerWithObservations should accept a second argument
        `observations` in their call() method. In training mode, this will be
        the `[inputs, targets]` of the training points; otherwise, `None`.
        """


class LatentVariableLayer(LayerWithObservations):
    """
    A latent variable layer, with amortized mean-field VI.

    The latent variable is distribution agnostic, but assumes a variational posterior
    that is fully factorised and is of the same distribution family as the prior.
    """

    def __init__(
        self,
        prior: tfp.distributions.Distribution,
        encoder: tf.keras.layers.Layer,
        compositor: Optional[
            Union[tf.keras.layers.Layer, Callable[[List[TensorType]], TensorType]]
        ] = None,
        name: Optional[str] = None,
    ):
        """
        :param prior: Distribution representing the prior over the latent variable
        :param encoder: Layer which gets passed the concatenated observation inputs
            and targets and returns the appropriate parameters to the approximate
            posterior distribution.
        :param compositor: Layer/callable that takes as input a list
            `[layer_inputs, latent_variables]` and combines it into a single tensor.
            If not specified (default), will use tf.keras.layers.Concatenate(axis=-1).
        """

        super().__init__(dtype=default_float(), name=name)
        self.prior = prior
        self.distribution_class = prior.__class__
        self.encoder = encoder
        self.compositor = (
            compositor if compositor is not None else tf.keras.layers.Concatenate(axis=-1)
        )

    def samples_and_posteriors(
        self,
        observations: ObservationType,
        num_samples: Optional[int] = None,
        training: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> Tuple[TensorType, tfp.distributions.Distribution]:
        # TODO: do we really need this function?
        # could change to encoder directly returning a Distribution object
        # (e.g. using DistributionLambda)?
        """
        Draw samples from the posterior disributions, given data for the encoder layer,
        and also return those distributions.

        :param recognition_data: data input for the encoder
        :param num_samples: number of samples
        :param training: training flag (for encoder)
        :param seed: random seed to sample with

        :return: (samples, posteriors)
        """
        inputs, targets = observations
        encoder_inputs = tf.concat(observations, axis=-1)
        distribution_params = self.encoder(encoder_inputs, training=training)

        posteriors = self.distribution_class(*distribution_params, allow_nan_stats=False)

        if num_samples is None:
            samples = posteriors.sample(seed=seed)  # [N, D]
        else:
            samples = posteriors.sample(num_samples, seed=seed)  # [S, N, D]

        return samples, posteriors

    def sample_prior(self, sample_shape: ShapeType, seed: Optional[int] = None) -> TensorType:
        # TODO: if we get rid of samples_and_posteriors (see there), can remove this one as well
        """
        Draw samples from the prior.

        :param sample_shape: shape to draw from prior
        """
        return self.prior.sample(sample_shape, seed=seed)

    def _inference_latent_samples_and_loss(
        self, layer_inputs: TensorType, observations: ObservationType, seed: Optional[int] = None
    ) -> Tuple[TensorType, TensorType]:
        """
        Sampling latent variables during training forward pass, hence requiring
        the observations. Also returns KL loss per datapoint.
        """
        samples, posteriors = self.samples_and_posteriors(
            observations, num_samples=None, training=True, seed=seed
        )
        # closed-form expectation E_q[log(q/p)] = KL[q∥p]:
        local_kls = self.local_kls(posteriors)
        loss_per_datapoint = tf.reduce_mean(local_kls, name="local_kls")
        return samples, loss_per_datapoint

    def _prediction_latent_samples(
        self, layer_inputs: TensorType, seed: Optional[int] = None
    ) -> TensorType:
        """
        Sampling latent variables during prediction forward pass, only
        depending on layer_inputs.
        """
        batch_shape = tf.shape(layer_inputs)[:-1]
        samples = self.sample_prior(batch_shape, seed=seed)
        return samples

    def call(
        self,
        layer_inputs: TensorType,
        observations: Optional[ObservationType] = None,
        training: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> TensorType:
        """
        When training: draw a sample of the latent variable from the posterior,
        whose distribution is parameterized by the encoder mapping from the data.
        Also adds a KL divergence [posterior∥prior] to the losses.

        When not training: draw a sample of the latent variable from the prior.

        :param recognition_data: the data inputs to the encoder network (if training).
            if not training - a single tensor, with leading dimensions indicating the
            number of samples to draw from the prior
        :param training: training mode indicator

        :return: samples of the latent variable
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

    def local_kls(self, posteriors: tfp.distributions.Distribution) -> TensorType:
        """
        The KL divergences [approximate posterior∥prior]

        :param posteriors: a distribution representing the approximate posteriors

        :return: tensor with the KL divergences for each of the posteriors
        """
        return posteriors.kl_divergence(self.prior)
