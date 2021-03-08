# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A Keras Layer that wraps a likelihood, while containing the necessary operations
for training"""
from typing import Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

from gpflow import default_float
from gpflow.base import TensorType
from gpflow.likelihoods import Likelihood

from gpflux.layers.trackable_layer import TrackableLayer


class LikelihoodLayer(TrackableLayer):
    """
    Keras layer that wraps a GPflow likelihood. This layer expects a
    MultivariateNormalDiag as its input, describing q(f). When training,
    computes the negative variational expectation -E_{q(f)}[log p(y|f)] and
    adds it as a layer loss. When not training, computes mean and variance
    of y under q(f).

    NOTE: You should _either_ use this `LikelihoodLayer` (together with
    `gpflux.models.DeepGP`) _or_ `LikelihoodLoss` (together with a
    `tf.keras.models.Sequential` model). Do NOT use both at once.
    """

    def __init__(self, likelihood: Likelihood):
        super().__init__(dtype=default_float())
        self.likelihood = likelihood

    def call(
        self,
        inputs: tfp.distributions.MultivariateNormalDiag,
        targets: Optional[TensorType] = None,
        training: bool = None,
    ) -> Union[tf.Tensor, "LikelihoodOutputs"]:
        # TODO: add support for non-distribution inputs? or other distributions?
        assert isinstance(inputs, tfp.distributions.MultivariateNormalDiag)
        F_mean = inputs.loc
        F_var = inputs.scale.diag ** 2

        if training:
            assert targets is not None
            # TODO: re-use LikelihoodLoss to remove code duplication
            loss_per_datapoint = tf.reduce_mean(
                -self.likelihood.variational_expectations(F_mean, F_var, targets)
            )
            Y_mean = Y_var = None
        else:
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
            Y_mean, Y_var = self.likelihood.predict_mean_and_var(F_mean, F_var)

        self.add_loss(loss_per_datapoint)

        # TODO: turn this layer into a DistributionLambda as well and
        # return correct distribution, not a mean/variance tuple
        return LikelihoodOutputs(F_mean, F_var, Y_mean, Y_var)


class LikelihoodOutputs(tf.Module, metaclass=TensorMetaClass):
    """
    Class to encapsulate the outputs of the likelihood layer.

    This class contains the mean and variance of the marginal distribution of the final latent
    GP layer, as well as the mean and variance of the likelihood.

    This class includes the `TensorMetaClass` to make objects behave as a tf.Tensor. This is
    necessary so it can be returned from the `DistributionLambda` Keras layer.
    """

    def __init__(
        self,
        f_mean: TensorType,
        f_var: TensorType,
        y_mean: Optional[TensorType],
        y_var: Optional[TensorType],
    ):
        super().__init__(name="likelihood_outputs")

        self.f_mean = f_mean
        self.f_var = f_var
        self.y_mean = y_mean
        self.y_var = y_var

    def _value(
        self, dtype: tf.dtypes.DType = None, name: str = None, as_ref: bool = False
    ) -> tf.Tensor:
        return self.f_mean

    @property
    def shape(self) -> tf.Tensor:
        return self.f_mean.shape

    @property
    def dtype(self) -> tf.dtypes.DType:
        return self.f_mean.dtype
