# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""
GPflow likelihood interface for Keras losses
"""
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.base import TensorType


class LikelihoodLoss(tf.keras.losses.Loss):
    """
    Keras loss that wraps a GPflow likelihood. When the prediction is a
    distribution q(f), returns the negative variational expectation
    -E_{q(f)}[log p(y|f)], otherwise just the log probability log p(y|f).

    When training a Keras model using this loss function, the value of this
    loss is not logged explicitly (the layer-specific losses are logged, as is
    the overall model loss). To output this loss value explicitly, wrap this
    class in a `Metric` and add it to the model metrics.

    NOTE: You should _either_ use this `LikelihoodLoss` (together with a
    `tf.keras.models.Sequential` model) _or_ `LikelihoodLayer` (together with
    `gpflux.models.DeepGP`). Do NOT use both at once.
    """

    def __init__(self, likelihood: gpflow.likelihoods.Likelihood):
        """
        :param likelihood: the GPflow likelihood object to use.
            **NOTE** if you want likelihood parameters to be trainable, you
            need to include the likelihood as an attribute on a TrackableLayer
            subclass that is part of your model.
        """
        super().__init__()

        self.likelihood = likelihood

    def call(
        self,
        y_true: TensorType,
        f_prediction: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
    ) -> TensorType:
        """
        Note that we deviate from the Keras Loss interface by calling the
        second argument `f_prediction` rather than `y_pred`.
        """
        if isinstance(f_prediction, tfp.distributions.MultivariateNormalDiag):

            F_mu = f_prediction.loc
            F_var = f_prediction.scale.diag ** 2
            return -self.likelihood.variational_expectations(F_mu, F_var, y_true)
        else:  # Tensor
            f_samples = f_prediction
            return -self.likelihood.log_prob(f_samples, y_true)
